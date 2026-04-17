import argparse
import glob
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct"
FRAME_DIR = "/mnt/ssd/code/Qwen3_vl/frames"
DEFAULT_BUNDLE_PATH = "/mnt/ssd/code/Qwen3_vl/qwen3vl_real_linear_case.pt"


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def init_dist():
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = getenv_int("MASTER_PORT", 29501)
    rank = getenv_int("RANK", 0)
    world_size = getenv_int("WORLD_SIZE", 1)

    print(
        f"[pre-init] host={socket.gethostname()} rank={rank} "
        f"world_size={world_size} master={master_addr}:{master_port}"
    )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size


def build_inputs(processor, frame_paths):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{p}" for p in frame_paths],
                    "sample_fps": 1,
                },
                {
                    "type": "text",
                    "text": "请用中文简要描述这个视频的主要内容。",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs.pop("token_type_ids", None)
    return inputs


def get_linear_module(model, layer_idx: int, linear_name: str):
    layer = model.model.language_model.layers[layer_idx]
    module_map = {
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
    }
    if linear_name not in module_map:
        raise ValueError(f"不支持的 linear_name={linear_name}")
    return module_map[linear_name]


def default_tp_mode_for_linear(linear_name: str) -> str:
    if linear_name in {"gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"}:
        return "column"
    if linear_name in {"down_proj", "o_proj"}:
        return "row"
    raise ValueError(f"无法为 linear_name={linear_name} 推断 tp mode。")


def dtype_from_name(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"不支持的 dtype 名称: {dtype_name}")
    return getattr(torch, dtype_name)


def resolve_save_dtype(save_dtype_arg: str, reference_tensor: torch.Tensor) -> torch.dtype:
    if save_dtype_arg == "auto":
        return reference_tensor.dtype
    return dtype_from_name(save_dtype_arg)


def resolve_comm_dtype(comm_dtype_arg: str, compute_dtype: torch.dtype) -> torch.dtype:
    if comm_dtype_arg == "auto":
        # 对 gloo + CPU 通信默认使用 float32，更稳，也能避免 bfloat16 collectives 带来的误差。
        return torch.float32 if compute_dtype == torch.bfloat16 else compute_dtype
    return dtype_from_name(comm_dtype_arg)


def cast_cpu_tensor(tensor: torch.Tensor, save_dtype: torch.dtype) -> torch.Tensor:
    return tensor.detach().clone().to("cpu", dtype=save_dtype)


def prepare_bundle(args):
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {FRAME_DIR}")
    frame_paths = frame_paths[: args.num_frames]

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    ).eval()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(model.device)

    linear_module = get_linear_module(model, args.layer_idx, args.linear_name)
    captured = {}

    def pre_hook(_module, module_inputs):
        captured["input"] = module_inputs[0].detach().clone()

    def forward_hook(_module, _module_inputs, module_output):
        captured["output"] = module_output.detach().clone()

    pre_handle = linear_module.register_forward_pre_hook(pre_hook)
    forward_handle = linear_module.register_forward_hook(forward_hook)
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        pre_handle.remove()
        forward_handle.remove()

    if "input" not in captured or "output" not in captured:
        raise RuntimeError("没有捕获到目标 linear 的输入/输出。")

    original_input = captured["input"]
    original_output = captured["output"]
    original_weight = linear_module.weight.detach().clone()
    original_bias = None if linear_module.bias is None else linear_module.bias.detach().clone()

    sanity_output = F.linear(original_input, original_weight, original_bias)
    sanity_max_diff = (sanity_output - original_output).abs().max().item()
    sanity_mean_diff = (sanity_output - original_output).abs().mean().item()

    save_dtype = resolve_save_dtype(args.save_dtype, original_input)
    bundle = {
        "layer_idx": args.layer_idx,
        "linear_name": args.linear_name,
        "tp_mode": args.tp_mode or default_tp_mode_for_linear(args.linear_name),
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "original_input_dtype": str(original_input.dtype),
        "original_weight_dtype": str(original_weight.dtype),
        "original_output_dtype": str(original_output.dtype),
        "original_input_device": str(original_input.device),
        "original_weight_device": str(original_weight.device),
        "original_output_device": str(original_output.device),
        "input": cast_cpu_tensor(original_input, save_dtype),
        "weight": cast_cpu_tensor(original_weight, save_dtype),
        "bias": None if original_bias is None else cast_cpu_tensor(original_bias, save_dtype),
        "output": cast_cpu_tensor(original_output, save_dtype),
        "frame_paths": frame_paths,
        "sanity_max_diff": sanity_max_diff,
        "sanity_mean_diff": sanity_mean_diff,
    }

    save_path = Path(args.bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)

    print(f"[prepare] bundle saved to {save_path}")
    print(
        f"[prepare] layer={args.layer_idx} linear={args.linear_name} "
        f"tp_mode={bundle['tp_mode']} save_dtype={bundle['save_dtype']}"
    )
    print(
        f"[prepare] input_shape={tuple(bundle['input'].shape)} "
        f"weight_shape={tuple(bundle['weight'].shape)} "
        f"output_shape={tuple(bundle['output'].shape)}"
    )
    print(
        f"[prepare] input_device={bundle['original_input_device']} "
        f"weight_device={bundle['original_weight_device']} "
        f"output_device={bundle['original_output_device']}"
    )
    print(
        f"[prepare] sanity max_diff={bundle['sanity_max_diff']} "
        f"mean_diff={bundle['sanity_mean_diff']}"
    )


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境里 torch.cuda.is_available() 为 False。")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def all_gather_cat_cpu(
    local_tensor: torch.Tensor,
    world_size: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    local_cpu = local_tensor.detach().to("cpu", dtype=comm_dtype)
    gather_list = [torch.empty_like(local_cpu) for _ in range(world_size)]
    dist.all_gather(gather_list, local_cpu)
    return torch.cat(gather_list, dim=-1).to(device=target_device, dtype=target_dtype)


def all_reduce_sum_cpu(
    local_tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    reduced = local_tensor.detach().to("cpu", dtype=comm_dtype)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.to(device=target_device, dtype=target_dtype)


def run_column_parallel(x, weight, bias, reference_output, rank, world_size, comm_dtype):
    out_features, _ = weight.shape
    if out_features % world_size != 0:
        raise ValueError(f"out_features={out_features} 不能被 world_size={world_size} 整除。")

    shard_out = out_features // world_size
    start = rank * shard_out
    end = (rank + 1) * shard_out

    weight_shard = weight[start:end, :].contiguous()
    bias_shard = None if bias is None else bias[start:end].contiguous()

    local_output = F.linear(x, weight_shard, bias_shard)
    ref_slice = reference_output[..., start:end].contiguous()
    local_max_diff = (local_output - ref_slice).abs().max().item()
    local_mean_diff = (local_output - ref_slice).abs().mean().item()
    gathered_output = all_gather_cat_cpu(local_output, world_size, x.device, x.dtype, comm_dtype)

    max_diff = (gathered_output - reference_output).abs().max().item()
    mean_diff = (gathered_output - reference_output).abs().mean().item()
    return local_output, gathered_output, ref_slice, local_max_diff, local_mean_diff, max_diff, mean_diff


def run_row_parallel(x, weight, bias, reference_output, rank, world_size, comm_dtype):
    _, in_features = weight.shape
    if in_features % world_size != 0:
        raise ValueError(f"in_features={in_features} 不能被 world_size={world_size} 整除。")

    shard_in = in_features // world_size
    start = rank * shard_in
    end = (rank + 1) * shard_in

    x_shard = x[..., start:end].contiguous()
    weight_shard = weight[:, start:end].contiguous()

    local_partial = F.linear(x_shard, weight_shard, bias=None)
    reduced_output = all_reduce_sum_cpu(local_partial, x.device, x.dtype, comm_dtype)
    reduced_output_cpu = reduced_output.detach().to("cpu", dtype=comm_dtype)
    if rank == 0 and bias is not None:
        reduced_output_cpu = reduced_output_cpu + bias.to("cpu", dtype=comm_dtype)
    dist.broadcast(reduced_output_cpu, src=0)
    reduced_output = reduced_output_cpu.to(device=x.device, dtype=x.dtype)

    max_diff = (reduced_output - reference_output).abs().max().item()
    mean_diff = (reduced_output - reference_output).abs().mean().item()
    return x_shard, local_partial, reduced_output, max_diff, mean_diff


def run_tp(args):
    rank, world_size = init_dist()
    device = get_device(args.device)

    bundle = torch.load(args.bundle_path, map_location="cpu")
    tp_mode = args.tp_mode or bundle["tp_mode"]
    compute_dtype_name = bundle["save_dtype"] if args.compute_dtype == "auto" else args.compute_dtype
    compute_dtype = dtype_from_name(compute_dtype_name)
    comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)

    x = bundle["input"].to(device=device, dtype=compute_dtype)
    weight = bundle["weight"].to(device=device, dtype=compute_dtype)
    bias = None if bundle["bias"] is None else bundle["bias"].to(device=device, dtype=compute_dtype)
    reference_output = bundle["output"].to(device=device, dtype=compute_dtype)

    print(
        f"[config] rank={rank} device={device} world_size={world_size} "
        f"layer={bundle['layer_idx']} linear={bundle['linear_name']} tp_mode={tp_mode}"
    )
    print(
        f"[config] input_shape={tuple(x.shape)} weight_shape={tuple(weight.shape)} "
        f"output_shape={tuple(reference_output.shape)} dtype={x.dtype} "
        f"bundle_dtype={bundle['save_dtype']} original_input_dtype={bundle['original_input_dtype']} "
        f"original_input_device={bundle.get('original_input_device', 'unknown')} comm_dtype={comm_dtype}"
    )

    direct_output = F.linear(x, weight, bias)
    direct_max_diff = (direct_output - reference_output).abs().max().item()
    direct_mean_diff = (direct_output - reference_output).abs().mean().item()
    print(
        f"[direct] rank={rank} direct_linear_vs_reference max_diff={direct_max_diff} "
        f"mean_diff={direct_mean_diff}"
    )

    if tp_mode == "column":
        local_output, gathered_output, ref_slice, local_max_diff, local_mean_diff, max_diff, mean_diff = run_column_parallel(
            x, weight, bias, reference_output, rank, world_size, comm_dtype
        )
        out_features = weight.shape[0]
        shard_out = out_features // world_size
        start = rank * shard_out
        end = (rank + 1) * shard_out
        direct_slice = direct_output[..., start:end].contiguous()
        local_vs_direct_max_diff = (local_output - direct_slice).abs().max().item()
        local_vs_direct_mean_diff = (local_output - direct_slice).abs().mean().item()
        gathered_vs_direct_max_diff = (gathered_output - direct_output).abs().max().item()
        gathered_vs_direct_mean_diff = (gathered_output - direct_output).abs().mean().item()
        print(
            f"[column] rank={rank} local_output_shape={tuple(local_output.shape)} "
            f"local_vs_reference_slice max_diff={local_max_diff} mean_diff={local_mean_diff} "
            f"local_vs_direct_slice max_diff={local_vs_direct_max_diff} mean_diff={local_vs_direct_mean_diff} "
            f"gathered_output_shape={tuple(gathered_output.shape)} "
            f"tp_vs_reference max_diff={max_diff} mean_diff={mean_diff} "
            f"tp_vs_direct max_diff={gathered_vs_direct_max_diff} mean_diff={gathered_vs_direct_mean_diff}"
        )
    elif tp_mode == "row":
        x_shard, local_partial, reduced_output, max_diff, mean_diff = run_row_parallel(
            x, weight, bias, reference_output, rank, world_size, comm_dtype
        )
        reduced_vs_direct_max_diff = (reduced_output - direct_output).abs().max().item()
        reduced_vs_direct_mean_diff = (reduced_output - direct_output).abs().mean().item()
        print(
            f"[row] rank={rank} x_shard_shape={tuple(x_shard.shape)} "
            f"local_partial_shape={tuple(local_partial.shape)} "
            f"reduced_output_shape={tuple(reduced_output.shape)} "
            f"tp_vs_reference max_diff={max_diff} mean_diff={mean_diff} "
            f"tp_vs_direct max_diff={reduced_vs_direct_max_diff} mean_diff={reduced_vs_direct_mean_diff}"
        )
    else:
        raise ValueError(f"不支持的 tp_mode={tp_mode}")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[done] rank={rank}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 真实 Linear 层的最小手工 TP 验证。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="从真实 Qwen3-VL Linear 层导出一份输入/权重/输出样本。")
    prepare_parser.add_argument("--layer-idx", type=int, default=11)
    prepare_parser.add_argument(
        "--linear-name",
        choices=["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
        default="gate_proj",
    )
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    prepare_parser.add_argument("--tp-mode", choices=["column", "row"], default=None)

    tp_parser = subparsers.add_parser("tp", help="用 gloo 双进程验证导出的真实 Linear 样本。")
    tp_parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    tp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tp_parser.add_argument("--tp-mode", choices=["column", "row"], default=None)
    tp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_bundle(args)
    elif args.command == "tp":
        run_tp(args)


if __name__ == "__main__":
    main()
