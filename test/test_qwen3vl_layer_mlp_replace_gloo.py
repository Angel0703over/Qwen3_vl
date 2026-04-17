import argparse
import glob
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.activations import ACT2FN

from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct"
FRAME_DIR = "/mnt/ssd/code/Qwen3_vl/frames"
DEFAULT_BUNDLE_PATH = "/mnt/ssd/code/Qwen3_vl/qwen3vl_layer_mlp_replace_case.pt"


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
        return torch.float32 if compute_dtype == torch.bfloat16 else compute_dtype
    return dtype_from_name(comm_dtype_arg)


def cast_cpu_tensor(tensor: torch.Tensor | None, save_dtype: torch.dtype):
    if tensor is None:
        return None
    return tensor.detach().clone().to("cpu", dtype=save_dtype)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境里 torch.cuda.is_available() 为 False。")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def all_reduce_sum_cpu(
    local_tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    reduced = local_tensor.detach().to("cpu", dtype=comm_dtype)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.to(device=target_device, dtype=target_dtype)


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

    layer = model.model.language_model.layers[args.layer_idx]
    mlp_module = layer.mlp
    post_attention_layernorm = layer.post_attention_layernorm
    captured = {}

    def post_attn_ln_pre_hook(_module, module_inputs):
        captured["mlp_residual"] = module_inputs[0].detach().clone()

    def mlp_pre_hook(_module, module_inputs):
        captured["mlp_input"] = module_inputs[0].detach().clone()

    def layer_forward_hook(_module, _module_inputs, module_output):
        captured["layer_output"] = module_output.detach().clone()

    post_attn_ln_pre_handle = post_attention_layernorm.register_forward_pre_hook(post_attn_ln_pre_hook)
    mlp_pre_handle = mlp_module.register_forward_pre_hook(mlp_pre_hook)
    layer_forward_handle = layer.register_forward_hook(layer_forward_hook)
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        post_attn_ln_pre_handle.remove()
        mlp_pre_handle.remove()
        layer_forward_handle.remove()

    if "mlp_input" not in captured or "mlp_residual" not in captured or "layer_output" not in captured:
        raise RuntimeError("没有捕获到目标 layer 的 MLP residual / MLP 输入 / layer 输出。")

    mlp_input = captured["mlp_input"].detach().clone()
    mlp_residual = captured["mlp_residual"].detach().clone()
    layer_output = captured["layer_output"].detach().clone()

    gate_weight = mlp_module.gate_proj.weight.detach().clone()
    up_weight = mlp_module.up_proj.weight.detach().clone()
    down_weight = mlp_module.down_proj.weight.detach().clone()
    gate_bias = None if mlp_module.gate_proj.bias is None else mlp_module.gate_proj.bias.detach().clone()
    up_bias = None if mlp_module.up_proj.bias is None else mlp_module.up_proj.bias.detach().clone()
    down_bias = None if mlp_module.down_proj.bias is None else mlp_module.down_proj.bias.detach().clone()
    hidden_act = mlp_module.config.hidden_act

    with torch.inference_mode():
        mlp_direct = mlp_module(mlp_input)
        residual = mlp_residual
        direct_layer_output = residual + mlp_direct
    sanity_max_diff = (direct_layer_output - layer_output).abs().max().item()
    sanity_mean_diff = (direct_layer_output - layer_output).abs().mean().item()

    save_dtype = resolve_save_dtype(args.save_dtype, mlp_input)
    bundle = {
        "layer_idx": args.layer_idx,
        "module_name": "layer_mlp_replace",
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "hidden_act": hidden_act,
        "mlp_residual": cast_cpu_tensor(mlp_residual, save_dtype),
        "original_input_dtype": str(mlp_input.dtype),
        "original_output_dtype": str(layer_output.dtype),
        "original_input_device": str(mlp_input.device),
        "original_output_device": str(layer_output.device),
        "mlp_input": cast_cpu_tensor(mlp_input, save_dtype),
        "layer_output": cast_cpu_tensor(layer_output, save_dtype),
        "gate_weight": cast_cpu_tensor(gate_weight, save_dtype),
        "up_weight": cast_cpu_tensor(up_weight, save_dtype),
        "down_weight": cast_cpu_tensor(down_weight, save_dtype),
        "gate_bias": cast_cpu_tensor(gate_bias, save_dtype),
        "up_bias": cast_cpu_tensor(up_bias, save_dtype),
        "down_bias": cast_cpu_tensor(down_bias, save_dtype),
        "frame_paths": frame_paths,
        "sanity_max_diff": sanity_max_diff,
        "sanity_mean_diff": sanity_mean_diff,
    }

    save_path = Path(args.bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)

    print(f"[prepare] bundle saved to {save_path}")
    print(
        f"[prepare] layer={args.layer_idx} hidden_act={hidden_act} save_dtype={bundle['save_dtype']}"
    )
    print(
        f"[prepare] mlp_residual_shape={tuple(bundle['mlp_residual'].shape)} "
        f"mlp_input_shape={tuple(bundle['mlp_input'].shape)} "
        f"gate_weight_shape={tuple(bundle['gate_weight'].shape)} "
        f"up_weight_shape={tuple(bundle['up_weight'].shape)} "
        f"down_weight_shape={tuple(bundle['down_weight'].shape)} "
        f"layer_output_shape={tuple(bundle['layer_output'].shape)}"
    )
    print(
        f"[prepare] input_device={bundle['original_input_device']} "
        f"output_device={bundle['original_output_device']}"
    )
    print(
        f"[prepare] sanity max_diff={bundle['sanity_max_diff']} "
        f"mean_diff={bundle['sanity_mean_diff']}"
    )


def run_tp(args):
    rank, world_size = init_dist()
    device = get_device(args.device)

    bundle = torch.load(args.bundle_path, map_location="cpu")
    compute_dtype_name = bundle["save_dtype"] if args.compute_dtype == "auto" else args.compute_dtype
    compute_dtype = dtype_from_name(compute_dtype_name)
    comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)
    act_fn = ACT2FN[bundle["hidden_act"]]

    mlp_residual = bundle["mlp_residual"].to(device=device, dtype=compute_dtype)
    mlp_input = bundle["mlp_input"].to(device=device, dtype=compute_dtype)
    reference_layer_output = bundle["layer_output"].to(device=device, dtype=compute_dtype)
    gate_weight = bundle["gate_weight"].to(device=device, dtype=compute_dtype)
    up_weight = bundle["up_weight"].to(device=device, dtype=compute_dtype)
    down_weight = bundle["down_weight"].to(device=device, dtype=compute_dtype)
    gate_bias = None if bundle["gate_bias"] is None else bundle["gate_bias"].to(device=device, dtype=compute_dtype)
    up_bias = None if bundle["up_bias"] is None else bundle["up_bias"].to(device=device, dtype=compute_dtype)
    down_bias = None if bundle["down_bias"] is None else bundle["down_bias"].to(device=device, dtype=compute_dtype)

    print(
        f"[config] rank={rank} device={device} world_size={world_size} "
        f"layer={bundle['layer_idx']} hidden_act={bundle['hidden_act']}"
    )
    print(
        f"[config] mlp_residual_shape={tuple(mlp_residual.shape)} "
        f"mlp_input_shape={tuple(mlp_input.shape)} "
        f"gate_weight_shape={tuple(gate_weight.shape)} "
        f"up_weight_shape={tuple(up_weight.shape)} "
        f"down_weight_shape={tuple(down_weight.shape)} "
        f"layer_output_shape={tuple(reference_layer_output.shape)} dtype={mlp_input.dtype} "
        f"bundle_dtype={bundle['save_dtype']} original_input_dtype={bundle['original_input_dtype']} "
        f"original_input_device={bundle['original_input_device']} comm_dtype={comm_dtype}"
    )

    direct_gate = F.linear(mlp_input, gate_weight, gate_bias)
    direct_up = F.linear(mlp_input, up_weight, up_bias)
    direct_fused = act_fn(direct_gate) * direct_up
    direct_mlp_output = F.linear(direct_fused, down_weight, down_bias)
    direct_layer_output = mlp_residual + direct_mlp_output

    direct_max_diff = (direct_layer_output - reference_layer_output).abs().max().item()
    direct_mean_diff = (direct_layer_output - reference_layer_output).abs().mean().item()
    print(
        f"[direct] rank={rank} direct_layer_mlp_replace_vs_reference max_diff={direct_max_diff} "
        f"mean_diff={direct_mean_diff}"
    )

    intermediate_size = gate_weight.shape[0]
    if intermediate_size % world_size != 0:
        raise ValueError(f"intermediate_size={intermediate_size} 不能被 world_size={world_size} 整除。")

    shard_intermediate = intermediate_size // world_size
    start = rank * shard_intermediate
    end = (rank + 1) * shard_intermediate

    gate_weight_shard = gate_weight[start:end, :].contiguous()
    up_weight_shard = up_weight[start:end, :].contiguous()
    gate_bias_shard = None if gate_bias is None else gate_bias[start:end].contiguous()
    up_bias_shard = None if up_bias is None else up_bias[start:end].contiguous()

    local_gate = F.linear(mlp_input, gate_weight_shard, gate_bias_shard)
    local_up = F.linear(mlp_input, up_weight_shard, up_bias_shard)
    local_fused = act_fn(local_gate) * local_up

    direct_gate_slice = direct_gate[..., start:end].contiguous()
    direct_up_slice = direct_up[..., start:end].contiguous()
    direct_fused_slice = direct_fused[..., start:end].contiguous()

    gate_local_vs_direct_max = (local_gate - direct_gate_slice).abs().max().item()
    gate_local_vs_direct_mean = (local_gate - direct_gate_slice).abs().mean().item()
    up_local_vs_direct_max = (local_up - direct_up_slice).abs().max().item()
    up_local_vs_direct_mean = (local_up - direct_up_slice).abs().mean().item()
    fused_local_vs_direct_max = (local_fused - direct_fused_slice).abs().max().item()
    fused_local_vs_direct_mean = (local_fused - direct_fused_slice).abs().mean().item()

    down_weight_shard = down_weight[:, start:end].contiguous()
    local_down_partial = F.linear(local_fused, down_weight_shard, bias=None)
    reduced_output = all_reduce_sum_cpu(local_down_partial, mlp_input.device, mlp_input.dtype, comm_dtype)
    reduced_output_cpu = reduced_output.detach().to("cpu", dtype=comm_dtype)
    if rank == 0 and down_bias is not None:
        reduced_output_cpu = reduced_output_cpu + down_bias.to("cpu", dtype=comm_dtype)
    dist.broadcast(reduced_output_cpu, src=0)
    tp_mlp_output = reduced_output_cpu.to(device=mlp_input.device, dtype=mlp_input.dtype)
    tp_layer_output = mlp_residual + tp_mlp_output

    tp_vs_reference_max = (tp_layer_output - reference_layer_output).abs().max().item()
    tp_vs_reference_mean = (tp_layer_output - reference_layer_output).abs().mean().item()
    tp_vs_direct_max = (tp_layer_output - direct_layer_output).abs().max().item()
    tp_vs_direct_mean = (tp_layer_output - direct_layer_output).abs().mean().item()

    print(
        f"[layer-mlp] rank={rank} local_gate_shape={tuple(local_gate.shape)} "
        f"local_up_shape={tuple(local_up.shape)} local_fused_shape={tuple(local_fused.shape)} "
        f"local_down_partial_shape={tuple(local_down_partial.shape)} tp_layer_output_shape={tuple(tp_layer_output.shape)}"
    )
    print(
        f"[layer-mlp] rank={rank} gate_local_vs_direct max_diff={gate_local_vs_direct_max} "
        f"mean_diff={gate_local_vs_direct_mean} "
        f"up_local_vs_direct max_diff={up_local_vs_direct_max} mean_diff={up_local_vs_direct_mean} "
        f"fused_local_vs_direct max_diff={fused_local_vs_direct_max} mean_diff={fused_local_vs_direct_mean}"
    )
    print(
        f"[layer-mlp] rank={rank} tp_vs_reference max_diff={tp_vs_reference_max} "
        f"mean_diff={tp_vs_reference_mean} "
        f"tp_vs_direct max_diff={tp_vs_direct_max} mean_diff={tp_vs_direct_mean}"
    )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[done] rank={rank}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 真实 decoder layer 的 MLP 替换 TP 验证。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="从真实 decoder layer 导出 MLP 输入和 layer 输出。")
    prepare_parser.add_argument("--layer-idx", type=int, default=11)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    tp_parser = subparsers.add_parser("tp", help="用 gloo 双进程验证 decoder layer 的 MLP 替换。")
    tp_parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    tp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_bundle(args)
    elif args.command == "tp":
        run_tp(args)


if __name__ == "__main__":
    main()
