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
from transformers.masking_utils import create_causal_mask

from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct"
FRAME_DIR = "/mnt/ssd/code/Qwen3_vl/frames"
DEFAULT_BUNDLE_PATH = "/mnt/ssd/code/Qwen3_vl/qwen3vl_full_layer_case.pt"


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


def cast_cpu_tensor(tensor: torch.Tensor | None, save_dtype: torch.dtype | None = None):
    if tensor is None:
        return None
    out = tensor.detach().clone().to("cpu")
    if save_dtype is not None:
        out = out.to(dtype=save_dtype)
    return out


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


def build_explicit_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask_2d: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    min_value = torch.finfo(dtype).min

    causal_mask = torch.full((batch_size, 1, seq_len, seq_len), min_value, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    if attention_mask_2d is not None:
        attention_mask_2d = attention_mask_2d.to(device=device)
        key_padding_mask = (attention_mask_2d[:, None, None, :] == 0)
        causal_mask = causal_mask.masked_fill(key_padding_mask, min_value)

    return causal_mask


def rms_norm_last_dim(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + eps)
    return weight * hidden_states_fp32.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_key_value_groups: int,
    scaling: float,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def attention_forward_from_tensors(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
    o_weight: torch.Tensor,
    o_bias: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    scaling: float,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = F.linear(hidden_states, q_weight, q_bias).view(hidden_shape)
    query_states = rms_norm_last_dim(query_states, q_norm_weight, rms_norm_eps).transpose(1, 2)

    key_states = F.linear(hidden_states, k_weight, k_bias).view(
        *input_shape, num_key_value_heads, head_dim
    )
    key_states = rms_norm_last_dim(key_states, k_norm_weight, rms_norm_eps).transpose(1, 2)

    value_states = F.linear(hidden_states, v_weight, v_bias).view(
        *input_shape, num_key_value_heads, head_dim
    ).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output, _ = eager_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        num_key_value_groups=num_attention_heads // num_key_value_heads,
        scaling=scaling,
    )

    attn_output_flat = attn_output.reshape(*input_shape, -1).contiguous()
    output = F.linear(attn_output_flat, o_weight, o_bias)
    return output


def mlp_forward_from_tensors(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor | None,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    hidden_act: str,
):
    act_fn = ACT2FN[hidden_act]
    gate_out = F.linear(hidden_states, gate_weight, gate_bias)
    up_out = F.linear(hidden_states, up_weight, up_bias)
    fused = act_fn(gate_out) * up_out
    return F.linear(fused, down_weight, down_bias)


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
        attn_implementation="eager",
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
    captured = {}

    def input_ln_pre_hook(_module, module_inputs):
        captured["layer_input"] = module_inputs[0].detach().clone()

    def input_ln_forward_hook(_module, _module_inputs, module_output):
        captured["attn_input"] = module_output.detach().clone()

    def attn_pre_hook(_module, module_args, module_kwargs):
        position_embeddings = module_kwargs.get("position_embeddings")
        if position_embeddings is None and len(module_args) > 1:
            position_embeddings = module_args[1]
        if position_embeddings is None:
            raise RuntimeError("没有在 self_attn pre-hook 中拿到 position_embeddings。")

        attention_mask = module_kwargs.get("attention_mask")
        if attention_mask is None and len(module_args) > 2:
            attention_mask = module_args[2]
        if attention_mask is not None:
            captured["attention_mask"] = attention_mask.detach().clone()

        cos, sin = position_embeddings
        captured["cos"] = cos.detach().clone()
        captured["sin"] = sin.detach().clone()

    def attn_forward_hook(_module, _module_inputs, module_output):
        if isinstance(module_output, tuple):
            captured["attn_output"] = module_output[0].detach().clone()
        else:
            captured["attn_output"] = module_output.detach().clone()

    def post_attn_ln_forward_hook(_module, _module_inputs, module_output):
        captured["mlp_input"] = module_output.detach().clone()

    def layer_forward_hook(_module, _module_inputs, module_output):
        captured["layer_output"] = module_output.detach().clone()

    input_ln_pre_handle = layer.input_layernorm.register_forward_pre_hook(input_ln_pre_hook)
    input_ln_forward_handle = layer.input_layernorm.register_forward_hook(input_ln_forward_hook)
    attn_pre_handle = layer.self_attn.register_forward_pre_hook(attn_pre_hook, with_kwargs=True)
    attn_forward_handle = layer.self_attn.register_forward_hook(attn_forward_hook)
    post_attn_ln_forward_handle = layer.post_attention_layernorm.register_forward_hook(post_attn_ln_forward_hook)
    layer_forward_handle = layer.register_forward_hook(layer_forward_hook)
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        input_ln_pre_handle.remove()
        input_ln_forward_handle.remove()
        attn_pre_handle.remove()
        attn_forward_handle.remove()
        post_attn_ln_forward_handle.remove()
        layer_forward_handle.remove()

    required = {
        "layer_input",
        "attn_input",
        "cos",
        "sin",
        "attn_output",
        "mlp_input",
        "layer_output",
    }
    if not required.issubset(captured):
        missing = required - set(captured)
        raise RuntimeError(f"没有捕获到完整 layer replay 的必要输入: {missing}")

    layer_input = captured["layer_input"].detach().clone()
    attn_input = captured["attn_input"].detach().clone()
    attention_mask = captured.get("attention_mask")
    if attention_mask is None:
        attention_mask = create_causal_mask(
            config=model.model.language_model.config,
            inputs_embeds=attn_input,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=None,
            position_ids=None,
        )
    if attention_mask is None:
        attention_mask = build_explicit_causal_mask(attn_input, inputs.get("attention_mask"))
    attention_mask = attention_mask.detach().clone()
    cos = captured["cos"].detach().clone()
    sin = captured["sin"].detach().clone()
    attn_output = captured["attn_output"].detach().clone()
    mlp_input = captured["mlp_input"].detach().clone()
    reference_layer_output = captured["layer_output"].detach().clone()

    attn = layer.self_attn
    mlp = layer.mlp
    q_weight = attn.q_proj.weight.detach().clone()
    q_bias = None if attn.q_proj.bias is None else attn.q_proj.bias.detach().clone()
    k_weight = attn.k_proj.weight.detach().clone()
    k_bias = None if attn.k_proj.bias is None else attn.k_proj.bias.detach().clone()
    v_weight = attn.v_proj.weight.detach().clone()
    v_bias = None if attn.v_proj.bias is None else attn.v_proj.bias.detach().clone()
    o_weight = attn.o_proj.weight.detach().clone()
    o_bias = None if attn.o_proj.bias is None else attn.o_proj.bias.detach().clone()
    q_norm_weight = attn.q_norm.weight.detach().clone()
    k_norm_weight = attn.k_norm.weight.detach().clone()

    gate_weight = mlp.gate_proj.weight.detach().clone()
    gate_bias = None if mlp.gate_proj.bias is None else mlp.gate_proj.bias.detach().clone()
    up_weight = mlp.up_proj.weight.detach().clone()
    up_bias = None if mlp.up_proj.bias is None else mlp.up_proj.bias.detach().clone()
    down_weight = mlp.down_proj.weight.detach().clone()
    down_bias = None if mlp.down_proj.bias is None else mlp.down_proj.bias.detach().clone()

    input_ln_weight = layer.input_layernorm.weight.detach().clone()
    input_ln_eps = layer.input_layernorm.variance_epsilon
    post_attn_ln_weight = layer.post_attention_layernorm.weight.detach().clone()
    post_attn_ln_eps = layer.post_attention_layernorm.variance_epsilon
    hidden_act = mlp.config.hidden_act

    with torch.inference_mode():
        direct_attn_input = rms_norm_last_dim(layer_input, input_ln_weight, input_ln_eps)
        direct_attn_output = attention_forward_from_tensors(
            direct_attn_input,
            attention_mask,
            cos,
            sin,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            q_norm_weight,
            k_norm_weight,
            attn.q_norm.variance_epsilon,
            attn.config.num_attention_heads,
            attn.config.num_key_value_heads,
            attn.head_dim,
            attn.scaling,
        )
        direct_after_attn = layer_input + direct_attn_output
        direct_mlp_input = rms_norm_last_dim(direct_after_attn, post_attn_ln_weight, post_attn_ln_eps)
        direct_mlp_output = mlp_forward_from_tensors(
            direct_mlp_input,
            gate_weight,
            gate_bias,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            hidden_act,
        )
        direct_layer_output = direct_after_attn + direct_mlp_output

    sanity_max_diff = (direct_layer_output - reference_layer_output).abs().max().item()
    sanity_mean_diff = (direct_layer_output - reference_layer_output).abs().mean().item()

    save_dtype = resolve_save_dtype(args.save_dtype, layer_input)
    bundle = {
        "layer_idx": args.layer_idx,
        "module_name": "full_layer",
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "hidden_act": hidden_act,
        "original_input_dtype": str(layer_input.dtype),
        "original_output_dtype": str(reference_layer_output.dtype),
        "original_input_device": str(layer_input.device),
        "original_output_device": str(reference_layer_output.device),
        "layer_input": cast_cpu_tensor(layer_input, save_dtype),
        "attn_input": cast_cpu_tensor(attn_input, save_dtype),
        "attention_mask": cast_cpu_tensor(attention_mask, None),
        "cos": cast_cpu_tensor(cos, save_dtype),
        "sin": cast_cpu_tensor(sin, save_dtype),
        "attn_output": cast_cpu_tensor(attn_output, save_dtype),
        "mlp_input": cast_cpu_tensor(mlp_input, save_dtype),
        "layer_output": cast_cpu_tensor(reference_layer_output, save_dtype),
        "q_weight": cast_cpu_tensor(q_weight, save_dtype),
        "q_bias": cast_cpu_tensor(q_bias, save_dtype),
        "k_weight": cast_cpu_tensor(k_weight, save_dtype),
        "k_bias": cast_cpu_tensor(k_bias, save_dtype),
        "v_weight": cast_cpu_tensor(v_weight, save_dtype),
        "v_bias": cast_cpu_tensor(v_bias, save_dtype),
        "o_weight": cast_cpu_tensor(o_weight, save_dtype),
        "o_bias": cast_cpu_tensor(o_bias, save_dtype),
        "q_norm_weight": cast_cpu_tensor(q_norm_weight, save_dtype),
        "k_norm_weight": cast_cpu_tensor(k_norm_weight, save_dtype),
        "gate_weight": cast_cpu_tensor(gate_weight, save_dtype),
        "gate_bias": cast_cpu_tensor(gate_bias, save_dtype),
        "up_weight": cast_cpu_tensor(up_weight, save_dtype),
        "up_bias": cast_cpu_tensor(up_bias, save_dtype),
        "down_weight": cast_cpu_tensor(down_weight, save_dtype),
        "down_bias": cast_cpu_tensor(down_bias, save_dtype),
        "input_ln_weight": cast_cpu_tensor(input_ln_weight, save_dtype),
        "input_ln_eps": input_ln_eps,
        "post_attn_ln_weight": cast_cpu_tensor(post_attn_ln_weight, save_dtype),
        "post_attn_ln_eps": post_attn_ln_eps,
        "rms_norm_eps": attn.q_norm.variance_epsilon,
        "num_attention_heads": attn.config.num_attention_heads,
        "num_key_value_heads": attn.config.num_key_value_heads,
        "head_dim": attn.head_dim,
        "scaling": attn.scaling,
        "attn_implementation": attn.config._attn_implementation,
        "frame_paths": frame_paths,
        "sanity_max_diff": sanity_max_diff,
        "sanity_mean_diff": sanity_mean_diff,
    }

    save_path = Path(args.bundle_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, save_path)

    print(f"[prepare] bundle saved to {save_path}")
    print(
        f"[prepare] layer={args.layer_idx} save_dtype={bundle['save_dtype']} "
        f"num_heads={bundle['num_attention_heads']} num_kv_heads={bundle['num_key_value_heads']} "
        f"head_dim={bundle['head_dim']} attn_impl={bundle['attn_implementation']} hidden_act={hidden_act}"
    )
    print(
        f"[prepare] layer_input_shape={tuple(bundle['layer_input'].shape)} "
        f"attn_input_shape={tuple(bundle['attn_input'].shape)} "
        f"mlp_input_shape={tuple(bundle['mlp_input'].shape)} "
        f"layer_output_shape={tuple(bundle['layer_output'].shape)}"
    )
    print(
        f"[prepare] attention_mask_shape={tuple(bundle['attention_mask'].shape)} "
        f"q_weight_shape={tuple(bundle['q_weight'].shape)} "
        f"gate_weight_shape={tuple(bundle['gate_weight'].shape)}"
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

    layer_input = bundle["layer_input"].to(device=device, dtype=compute_dtype)
    attention_mask = bundle["attention_mask"].to(device=device)
    cos = bundle["cos"].to(device=device, dtype=compute_dtype)
    sin = bundle["sin"].to(device=device, dtype=compute_dtype)
    reference_layer_output = bundle["layer_output"].to(device=device, dtype=compute_dtype)

    q_weight = bundle["q_weight"].to(device=device, dtype=compute_dtype)
    q_bias = None if bundle["q_bias"] is None else bundle["q_bias"].to(device=device, dtype=compute_dtype)
    k_weight = bundle["k_weight"].to(device=device, dtype=compute_dtype)
    k_bias = None if bundle["k_bias"] is None else bundle["k_bias"].to(device=device, dtype=compute_dtype)
    v_weight = bundle["v_weight"].to(device=device, dtype=compute_dtype)
    v_bias = None if bundle["v_bias"] is None else bundle["v_bias"].to(device=device, dtype=compute_dtype)
    o_weight = bundle["o_weight"].to(device=device, dtype=compute_dtype)
    o_bias = None if bundle["o_bias"] is None else bundle["o_bias"].to(device=device, dtype=compute_dtype)
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=compute_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=compute_dtype)
    gate_weight = bundle["gate_weight"].to(device=device, dtype=compute_dtype)
    gate_bias = None if bundle["gate_bias"] is None else bundle["gate_bias"].to(device=device, dtype=compute_dtype)
    up_weight = bundle["up_weight"].to(device=device, dtype=compute_dtype)
    up_bias = None if bundle["up_bias"] is None else bundle["up_bias"].to(device=device, dtype=compute_dtype)
    down_weight = bundle["down_weight"].to(device=device, dtype=compute_dtype)
    down_bias = None if bundle["down_bias"] is None else bundle["down_bias"].to(device=device, dtype=compute_dtype)
    input_ln_weight = bundle["input_ln_weight"].to(device=device, dtype=compute_dtype)
    post_attn_ln_weight = bundle["post_attn_ln_weight"].to(device=device, dtype=compute_dtype)

    hidden_act = bundle["hidden_act"]
    num_attention_heads = bundle["num_attention_heads"]
    num_key_value_heads = bundle["num_key_value_heads"]
    head_dim = bundle["head_dim"]
    rms_norm_eps = bundle["rms_norm_eps"]
    scaling = bundle["scaling"]
    input_ln_eps = bundle["input_ln_eps"]
    post_attn_ln_eps = bundle["post_attn_ln_eps"]

    if num_attention_heads % world_size != 0 or num_key_value_heads % world_size != 0:
        raise ValueError(
            f"当前实现要求 num_attention_heads={num_attention_heads} 和 num_key_value_heads={num_key_value_heads} "
            f"都能被 world_size={world_size} 整除。"
        )
    if gate_weight.shape[0] % world_size != 0:
        raise ValueError(f"intermediate_size={gate_weight.shape[0]} 不能被 world_size={world_size} 整除。")

    local_q_heads = num_attention_heads // world_size
    local_kv_heads = num_key_value_heads // world_size
    intermediate_size = gate_weight.shape[0]
    shard_intermediate = intermediate_size // world_size

    print(
        f"[config] rank={rank} device={device} world_size={world_size} layer={bundle['layer_idx']} "
        f"num_heads={num_attention_heads} num_kv_heads={num_key_value_heads} "
        f"local_q_heads={local_q_heads} local_kv_heads={local_kv_heads} head_dim={head_dim} "
        f"hidden_act={hidden_act} attn_impl={bundle.get('attn_implementation', 'unknown')}"
    )
    print(
        f"[config] layer_input_shape={tuple(layer_input.shape)} layer_output_shape={tuple(reference_layer_output.shape)} "
        f"dtype={layer_input.dtype} bundle_dtype={bundle['save_dtype']} "
        f"original_input_dtype={bundle['original_input_dtype']} original_input_device={bundle['original_input_device']} "
        f"comm_dtype={comm_dtype}"
    )

    direct_attn_input = rms_norm_last_dim(layer_input, input_ln_weight, input_ln_eps)
    direct_attn_output = attention_forward_from_tensors(
        direct_attn_input,
        attention_mask,
        cos,
        sin,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        q_norm_weight,
        k_norm_weight,
        rms_norm_eps,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        scaling,
    )
    direct_after_attn = layer_input + direct_attn_output
    direct_mlp_input = rms_norm_last_dim(direct_after_attn, post_attn_ln_weight, post_attn_ln_eps)
    direct_mlp_output = mlp_forward_from_tensors(
        direct_mlp_input,
        gate_weight,
        gate_bias,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        hidden_act,
    )
    direct_layer_output = direct_after_attn + direct_mlp_output

    direct_max_diff = (direct_layer_output - reference_layer_output).abs().max().item()
    direct_mean_diff = (direct_layer_output - reference_layer_output).abs().mean().item()
    print(
        f"[direct] rank={rank} direct_full_layer_vs_reference max_diff={direct_max_diff} "
        f"mean_diff={direct_mean_diff}"
    )

    # TP attention
    tp_attn_input = direct_attn_input
    q_start = rank * local_q_heads * head_dim
    q_end = (rank + 1) * local_q_heads * head_dim
    kv_start = rank * local_kv_heads * head_dim
    kv_end = (rank + 1) * local_kv_heads * head_dim

    local_q_proj = F.linear(
        tp_attn_input,
        q_weight[q_start:q_end, :].contiguous(),
        None if q_bias is None else q_bias[q_start:q_end].contiguous(),
    )
    local_q = rms_norm_last_dim(
        local_q_proj.view(*tp_attn_input.shape[:-1], local_q_heads, head_dim),
        q_norm_weight,
        rms_norm_eps,
    ).transpose(1, 2)

    local_k_proj = F.linear(
        tp_attn_input,
        k_weight[kv_start:kv_end, :].contiguous(),
        None if k_bias is None else k_bias[kv_start:kv_end].contiguous(),
    )
    local_k = rms_norm_last_dim(
        local_k_proj.view(*tp_attn_input.shape[:-1], local_kv_heads, head_dim),
        k_norm_weight,
        rms_norm_eps,
    ).transpose(1, 2)

    local_v = F.linear(
        tp_attn_input,
        v_weight[kv_start:kv_end, :].contiguous(),
        None if v_bias is None else v_bias[kv_start:kv_end].contiguous(),
    ).view(*tp_attn_input.shape[:-1], local_kv_heads, head_dim).transpose(1, 2)

    local_q, local_k = apply_rotary_pos_emb(local_q, local_k, cos, sin)
    local_attn_output, _ = eager_attention_forward(
        local_q,
        local_k,
        local_v,
        attention_mask,
        num_key_value_groups=local_q_heads // local_kv_heads,
        scaling=scaling,
    )
    local_attn_flat = local_attn_output.reshape(*tp_attn_input.shape[:-1], -1).contiguous()
    local_o_partial = F.linear(
        local_attn_flat,
        o_weight[:, q_start:q_end].contiguous(),
        bias=None,
    )
    reduced_attn = all_reduce_sum_cpu(local_o_partial, layer_input.device, layer_input.dtype, comm_dtype)
    reduced_attn_cpu = reduced_attn.detach().to("cpu", dtype=comm_dtype)
    if rank == 0 and o_bias is not None:
        reduced_attn_cpu = reduced_attn_cpu + o_bias.to("cpu", dtype=comm_dtype)
    dist.broadcast(reduced_attn_cpu, src=0)
    tp_attn_output = reduced_attn_cpu.to(device=layer_input.device, dtype=layer_input.dtype)
    tp_after_attn = layer_input + tp_attn_output

    # TP MLP
    tp_mlp_input = rms_norm_last_dim(tp_after_attn, post_attn_ln_weight, post_attn_ln_eps)
    inter_start = rank * shard_intermediate
    inter_end = (rank + 1) * shard_intermediate
    local_gate = F.linear(
        tp_mlp_input,
        gate_weight[inter_start:inter_end, :].contiguous(),
        None if gate_bias is None else gate_bias[inter_start:inter_end].contiguous(),
    )
    local_up = F.linear(
        tp_mlp_input,
        up_weight[inter_start:inter_end, :].contiguous(),
        None if up_bias is None else up_bias[inter_start:inter_end].contiguous(),
    )
    local_fused = ACT2FN[hidden_act](local_gate) * local_up
    local_down_partial = F.linear(
        local_fused,
        down_weight[:, inter_start:inter_end].contiguous(),
        bias=None,
    )
    reduced_mlp = all_reduce_sum_cpu(local_down_partial, layer_input.device, layer_input.dtype, comm_dtype)
    reduced_mlp_cpu = reduced_mlp.detach().to("cpu", dtype=comm_dtype)
    if rank == 0 and down_bias is not None:
        reduced_mlp_cpu = reduced_mlp_cpu + down_bias.to("cpu", dtype=comm_dtype)
    dist.broadcast(reduced_mlp_cpu, src=0)
    tp_mlp_output = reduced_mlp_cpu.to(device=layer_input.device, dtype=layer_input.dtype)
    tp_layer_output = tp_after_attn + tp_mlp_output

    tp_vs_reference_max = (tp_layer_output - reference_layer_output).abs().max().item()
    tp_vs_reference_mean = (tp_layer_output - reference_layer_output).abs().mean().item()
    tp_vs_direct_max = (tp_layer_output - direct_layer_output).abs().max().item()
    tp_vs_direct_mean = (tp_layer_output - direct_layer_output).abs().mean().item()

    print(
        f"[full-layer] rank={rank} tp_attn_input_shape={tuple(tp_attn_input.shape)} "
        f"local_attn_flat_shape={tuple(local_attn_flat.shape)} "
        f"tp_after_attn_shape={tuple(tp_after_attn.shape)} "
        f"tp_mlp_input_shape={tuple(tp_mlp_input.shape)} "
        f"local_fused_shape={tuple(local_fused.shape)} "
        f"tp_layer_output_shape={tuple(tp_layer_output.shape)}"
    )
    print(
        f"[full-layer] rank={rank} tp_vs_reference max_diff={tp_vs_reference_max} "
        f"mean_diff={tp_vs_reference_mean} "
        f"tp_vs_direct max_diff={tp_vs_direct_max} mean_diff={tp_vs_direct_mean}"
    )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[done] rank={rank}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 完整 decoder layer 的最小手工 TP replay。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="从真实 decoder layer 导出完整 replay 样本。")
    prepare_parser.add_argument("--layer-idx", type=int, default=11)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    tp_parser = subparsers.add_parser("tp", help="用 gloo 双进程验证完整 decoder layer replay。")
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
