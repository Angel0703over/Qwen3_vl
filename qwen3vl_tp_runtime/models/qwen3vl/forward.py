"""Qwen3-VL text decoder forward and TP replay kernels."""

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from qwen3vl_tp_runtime.hexgen_core.distributed import all_gather_cpu, broadcast_cpu
from qwen3vl_tp_runtime.models.qwen3vl.ops import apply_rope, attn_eager, rms_norm


def compose_layer_bundle(layer_bundle: dict, runtime_bundle: dict) -> dict:
    # 多层执行时，每层参数独立存放，共享的运行时张量在这里补齐。
    bundle = dict(layer_bundle)
    bundle["attention_mask"] = runtime_bundle["attention_mask"]
    bundle["cos"] = runtime_bundle["cos"]
    bundle["sin"] = runtime_bundle["sin"]
    return bundle


def apply_deepstack(
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    visual_embeds: torch.Tensor | None,
) -> torch.Tensor:
    if visual_pos_masks is None or visual_embeds is None:
        return hidden_states

    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    hidden_states = hidden_states.clone()
    local_this = hidden_states[visual_pos_masks, :] + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


def get_deepstack_embeds(stage_bundle: dict, layer_idx: int) -> torch.Tensor | None:
    deepstack_by_layer = stage_bundle.get("deepstack_by_layer")
    if deepstack_by_layer is None:
        return None
    return deepstack_by_layer.get(layer_idx)


def _resolve_tp_math_dtype(hidden_states: torch.Tensor, math_mode: str) -> tuple[torch.dtype, torch.dtype]:
    orig_dtype = hidden_states.dtype
    if math_mode == "orig":
        return orig_dtype, orig_dtype
    if math_mode == "float32":
        math_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        return orig_dtype, math_dtype
    raise ValueError(f"不支持的 TP math_mode: {math_mode!r}")


def _cast_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device, dtype=dtype)


def _slice_local_past_states(
    past_states: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    full_num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    tensor_name: str,
) -> torch.Tensor | None:
    if past_states is None:
        return None
    if past_states.dim() != 4:
        raise ValueError(
            f"{tensor_name} 需要是 4 维张量，当前拿到 shape={tuple(past_states.shape)}"
        )
    if full_num_heads % world_size != 0:
        raise ValueError(
            f"{tensor_name} 对应的 head 数 {full_num_heads} 不能被 TP world_size={world_size} 整除。"
        )
    if past_states.shape[1] != full_num_heads:
        raise ValueError(
            f"{tensor_name} 的 head 维不匹配，"
            f"shape={tuple(past_states.shape)} expected_heads={full_num_heads}"
        )
    shard = full_num_heads // world_size
    start = rank * shard
    end = start + shard
    return past_states[:, start:end, ...].to(device=device, dtype=dtype)


def forward_attention(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_attention(hidden_states, bundle)["attn_output"]


def _concat_past_key_value(
    current_states: torch.Tensor,
    past_states: torch.Tensor | None,
) -> torch.Tensor:
    if past_states is None:
        return current_states
    past_states = past_states.to(device=current_states.device, dtype=current_states.dtype)
    if past_states.dim() != current_states.dim():
        raise ValueError(
            "past_states 和 current_states 维度不一致，"
            f"past_shape={tuple(past_states.shape)} current_shape={tuple(current_states.shape)}"
        )
    if past_states.shape[:-2] != current_states.shape[:-2] or past_states.shape[-1] != current_states.shape[-1]:
        raise ValueError(
            "past_states 和 current_states 的 batch/head/head_dim 不一致，"
            f"past_shape={tuple(past_states.shape)} current_shape={tuple(current_states.shape)}"
        )
    return torch.cat([past_states, current_states], dim=-2)


def trace_attention(hidden_states: torch.Tensor, bundle: dict) -> dict:
    input_shape = hidden_states.shape[:-1]
    head_dim = bundle["head_dim"]
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]

    query_states = F.linear(hidden_states, bundle["q_weight"], bundle["q_bias"]).view(*input_shape, -1, head_dim)
    query_states = rms_norm(query_states, bundle["q_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    key_states = F.linear(hidden_states, bundle["k_weight"], bundle["k_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    )
    key_states = rms_norm(key_states, bundle["k_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    value_states = F.linear(hidden_states, bundle["v_weight"], bundle["v_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    ).transpose(1, 2)

    query_states, key_states = apply_rope(query_states, key_states, bundle["cos"], bundle["sin"])
    attn_output, _ = attn_eager(
        query_states,
        key_states,
        value_states,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
    )
    attn_context = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = F.linear(attn_context, bundle["o_weight"], bundle["o_bias"])
    return {
        "attn_context": attn_context,
        "attn_output": attn_output,
    }


def forward_attention_cached(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_attention_cached(hidden_states, bundle)["attn_output"]


def trace_attention_cached(hidden_states: torch.Tensor, bundle: dict) -> dict:
    return _trace_attention_cached_core(
        hidden_states,
        bundle,
        past_key=bundle.get("past_key"),
        past_value=bundle.get("past_value"),
    )


def _trace_attention_cached_core(
    hidden_states: torch.Tensor,
    bundle: dict,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
) -> dict:
    input_shape = hidden_states.shape[:-1]
    head_dim = bundle["head_dim"]
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]

    query_states = F.linear(hidden_states, bundle["q_weight"], bundle["q_bias"]).view(*input_shape, -1, head_dim)
    query_states = rms_norm(query_states, bundle["q_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    key_states = F.linear(hidden_states, bundle["k_weight"], bundle["k_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    )
    key_states = rms_norm(key_states, bundle["k_norm_weight"], bundle["rms_norm_eps"]).transpose(1, 2)

    value_states = F.linear(hidden_states, bundle["v_weight"], bundle["v_bias"]).view(
        *input_shape, num_kv_heads, head_dim
    ).transpose(1, 2)

    query_states, key_states = apply_rope(query_states, key_states, bundle["cos"], bundle["sin"])
    full_key_states = _concat_past_key_value(key_states, past_key)
    full_value_states = _concat_past_key_value(value_states, past_value)

    attn_output, _ = attn_eager(
        query_states,
        full_key_states,
        full_value_states,
        bundle["attention_mask"],
        num_key_value_groups=num_heads // num_kv_heads,
        scaling=bundle["scaling"],
    )
    attn_context = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = F.linear(attn_context, bundle["o_weight"], bundle["o_bias"])
    return {
        "attn_context": attn_context,
        "attn_output": attn_output,
        "current_key": key_states,
        "current_value": value_states,
        "full_key": full_key_states,
        "full_value": full_value_states,
    }


def forward_attention_cached_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> torch.Tensor:
    return trace_attention_cached_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
    )["attn_output"]


def trace_attention_cached_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> dict:
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]
    head_dim = bundle["head_dim"]
    orig_dtype, math_dtype = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device

    if num_heads % world_size != 0 or num_kv_heads % world_size != 0:
        raise ValueError("当前 TP attention 要求 num_heads 和 num_kv_heads 都能被 world_size 整除。")

    local_q_heads = num_heads // world_size
    local_kv_heads = num_kv_heads // world_size
    q_head_start = rank * local_q_heads
    q_head_end = (rank + 1) * local_q_heads
    kv_head_start = rank * local_kv_heads
    kv_head_end = (rank + 1) * local_kv_heads

    x = hidden_states.to(dtype=orig_dtype)
    q_weight = bundle["q_weight"].to(device=device, dtype=orig_dtype)
    q_bias = _cast_optional_tensor(
        bundle["q_bias"],
        device=device,
        dtype=orig_dtype,
    )
    k_weight = bundle["k_weight"].to(device=device, dtype=orig_dtype)
    k_bias = _cast_optional_tensor(
        bundle["k_bias"],
        device=device,
        dtype=orig_dtype,
    )
    v_weight = bundle["v_weight"].to(device=device, dtype=orig_dtype)
    v_bias = _cast_optional_tensor(
        bundle["v_bias"],
        device=device,
        dtype=orig_dtype,
    )
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=orig_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=orig_dtype)

    full_q_proj = F.linear(x, q_weight, q_bias)
    local_q_proj = full_q_proj.view(*hidden_states.shape[:-1], num_heads, head_dim)[..., q_head_start:q_head_end, :]
    local_q = rms_norm(
        local_q_proj,
        q_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_k_proj = F.linear(x, k_weight, k_bias)
    local_k_proj = full_k_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim)[..., kv_head_start:kv_head_end, :]
    local_k = rms_norm(
        local_k_proj,
        k_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_v_proj = F.linear(x, v_weight, v_bias)
    local_v = (
        full_v_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim)[..., kv_head_start:kv_head_end, :]
        .transpose(1, 2)
    )

    local_q, local_k = apply_rope(local_q, local_k, bundle["cos"], bundle["sin"])
    local_past_key = _slice_local_past_states(
        bundle.get("past_key"),
        rank=rank,
        world_size=world_size,
        full_num_heads=num_kv_heads,
        device=device,
        dtype=orig_dtype,
        tensor_name="past_key",
    )
    local_past_value = _slice_local_past_states(
        bundle.get("past_value"),
        rank=rank,
        world_size=world_size,
        full_num_heads=num_kv_heads,
        device=device,
        dtype=orig_dtype,
        tensor_name="past_value",
    )
    local_full_k = _concat_past_key_value(local_k, local_past_key)
    local_full_v = _concat_past_key_value(local_v, local_past_value)

    local_attn_output, _ = attn_eager(
        local_q,
        local_full_k,
        local_full_v,
        bundle["attention_mask"],
        num_key_value_groups=local_q_heads // local_kv_heads,
        scaling=bundle["scaling"],
        compute_dtype=None,
        score_output_dtype=None,
        probabilities_dtype=orig_dtype,
        output_dtype=orig_dtype,
    )
    local_attn_context = local_attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    gathered_context = all_gather_cpu(
        local_attn_context,
        device,
        math_dtype,
        comm_dtype,
        group=tp_group,
    )
    full_attn_context = torch.cat(gathered_context, dim=-1)
    full_o_weight = bundle["o_weight"].to(device=device, dtype=math_dtype)
    full_o_bias = _cast_optional_tensor(
        bundle["o_bias"],
        device=device,
        dtype=math_dtype,
    )
    leader_output = None
    if rank == 0:
        leader_output = F.linear(full_attn_context, full_o_weight, full_o_bias)
    attn_output = broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
    )
    gathered_current_k = all_gather_cpu(
        local_k,
        device,
        orig_dtype,
        comm_dtype,
        group=tp_group,
    )
    gathered_current_v = all_gather_cpu(
        local_v,
        device,
        orig_dtype,
        comm_dtype,
        group=tp_group,
    )
    gathered_full_k = all_gather_cpu(
        local_full_k,
        device,
        orig_dtype,
        comm_dtype,
        group=tp_group,
    )
    gathered_full_v = all_gather_cpu(
        local_full_v,
        device,
        orig_dtype,
        comm_dtype,
        group=tp_group,
    )
    return {
        "attn_context": local_attn_context,
        "attn_output": attn_output,
        "current_key": torch.cat(gathered_current_k, dim=1),
        "current_value": torch.cat(gathered_current_v, dim=1),
        "full_key": torch.cat(gathered_full_k, dim=1),
        "full_value": torch.cat(gathered_full_v, dim=1),
    }


def forward_attention_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> torch.Tensor:
    return trace_attention_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
    )["attn_output"]


def trace_attention_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> dict:
    num_heads = bundle["num_attention_heads"]
    num_kv_heads = bundle["num_key_value_heads"]
    head_dim = bundle["head_dim"]
    orig_dtype, math_dtype = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device

    if num_heads % world_size != 0 or num_kv_heads % world_size != 0:
        raise ValueError("当前 TP attention 要求 num_heads 和 num_kv_heads 都能被 world_size 整除。")

    local_q_heads = num_heads // world_size
    local_kv_heads = num_kv_heads // world_size
    q_head_start = rank * local_q_heads
    q_head_end = (rank + 1) * local_q_heads
    kv_head_start = rank * local_kv_heads
    kv_head_end = (rank + 1) * local_kv_heads

    x = hidden_states.to(dtype=orig_dtype)
    q_weight = bundle["q_weight"].to(device=device, dtype=orig_dtype)
    q_bias = _cast_optional_tensor(
        bundle["q_bias"],
        device=device,
        dtype=orig_dtype,
    )
    k_weight = bundle["k_weight"].to(device=device, dtype=orig_dtype)
    k_bias = _cast_optional_tensor(
        bundle["k_bias"],
        device=device,
        dtype=orig_dtype,
    )
    v_weight = bundle["v_weight"].to(device=device, dtype=orig_dtype)
    v_bias = _cast_optional_tensor(
        bundle["v_bias"],
        device=device,
        dtype=orig_dtype,
    )
    q_norm_weight = bundle["q_norm_weight"].to(device=device, dtype=orig_dtype)
    k_norm_weight = bundle["k_norm_weight"].to(device=device, dtype=orig_dtype)

    full_q_proj = F.linear(
        x,
        q_weight,
        q_bias,
    )
    local_q_proj = full_q_proj.view(*hidden_states.shape[:-1], num_heads, head_dim)[..., q_head_start:q_head_end, :]
    local_q = rms_norm(
        local_q_proj,
        q_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_k_proj = F.linear(
        x,
        k_weight,
        k_bias,
    )
    local_k_proj = full_k_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim)[..., kv_head_start:kv_head_end, :]
    local_k = rms_norm(
        local_k_proj,
        k_norm_weight,
        bundle["rms_norm_eps"],
    ).transpose(1, 2)

    full_v_proj = F.linear(
        x,
        v_weight,
        v_bias,
    )
    local_v = (
        full_v_proj.view(*hidden_states.shape[:-1], num_kv_heads, head_dim)[..., kv_head_start:kv_head_end, :]
        .transpose(1, 2)
    )

    local_q, local_k = apply_rope(local_q, local_k, bundle["cos"], bundle["sin"])
    local_attn_output, _ = attn_eager(
        local_q,
        local_k,
        local_v,
        bundle["attention_mask"],
        num_key_value_groups=local_q_heads // local_kv_heads,
        scaling=bundle["scaling"],
        compute_dtype=None,
        score_output_dtype=None,
        probabilities_dtype=orig_dtype,
        output_dtype=orig_dtype,
    )
    local_attn_context = local_attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    gathered_context = all_gather_cpu(
        local_attn_context,
        device,
        math_dtype,
        comm_dtype,
        group=tp_group,
    )
    full_attn_context = torch.cat(gathered_context, dim=-1)
    full_o_weight = bundle["o_weight"].to(device=device, dtype=math_dtype)
    full_o_bias = _cast_optional_tensor(
        bundle["o_bias"],
        device=device,
        dtype=math_dtype,
    )
    leader_output = None
    if rank == 0:
        leader_output = F.linear(full_attn_context, full_o_weight, full_o_bias)
    attn_output = broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
    )
    return {
        "attn_context": local_attn_context,
        "attn_output": attn_output,
    }


def forward_mlp(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    act_fn = ACT2FN[bundle["hidden_act"]]
    gate_out = F.linear(hidden_states, bundle["gate_weight"], bundle["gate_bias"])
    up_out = F.linear(hidden_states, bundle["up_weight"], bundle["up_bias"])
    fused = act_fn(gate_out) * up_out
    return F.linear(fused, bundle["down_weight"], bundle["down_bias"])


def trace_mlp(hidden_states: torch.Tensor, bundle: dict) -> dict:
    act_fn = ACT2FN[bundle["hidden_act"]]
    gate_out = F.linear(hidden_states, bundle["gate_weight"], bundle["gate_bias"])
    up_out = F.linear(hidden_states, bundle["up_weight"], bundle["up_bias"])
    fused = act_fn(gate_out) * up_out
    mlp_output = F.linear(fused, bundle["down_weight"], bundle["down_bias"])
    return {
        "gate_out": gate_out,
        "up_out": up_out,
        "fused": fused,
        "mlp_output": mlp_output,
    }


def trace_mlp_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> dict:
    intermediate_size = bundle["gate_weight"].shape[0]
    orig_dtype, math_dtype = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device
    if intermediate_size % world_size != 0:
        raise ValueError("当前 TP MLP 要求 intermediate_size 能被 world_size 整除。")

    act_fn = ACT2FN[bundle["hidden_act"]]
    shard = intermediate_size // world_size
    start = rank * shard
    end = (rank + 1) * shard

    x = hidden_states.to(dtype=orig_dtype)
    gate_weight = bundle["gate_weight"].to(device=device, dtype=orig_dtype)
    gate_bias = _cast_optional_tensor(
        bundle["gate_bias"],
        device=device,
        dtype=orig_dtype,
    )
    up_weight = bundle["up_weight"].to(device=device, dtype=orig_dtype)
    up_bias = _cast_optional_tensor(
        bundle["up_bias"],
        device=device,
        dtype=orig_dtype,
    )
    down_weight = bundle["down_weight"].to(device=device, dtype=math_dtype)
    down_bias = _cast_optional_tensor(
        bundle["down_bias"],
        device=device,
        dtype=math_dtype,
    )

    full_gate_out = F.linear(x, gate_weight, gate_bias)
    full_up_out = F.linear(x, up_weight, up_bias)
    gate_out = full_gate_out[..., start:end]
    up_out = full_up_out[..., start:end]
    fused_out = act_fn(gate_out) * up_out
    gathered_fused = all_gather_cpu(
        fused_out,
        device,
        math_dtype,
        comm_dtype,
        group=tp_group,
    )
    full_fused = torch.cat(gathered_fused, dim=-1)
    leader_output = None
    if rank == 0:
        leader_output = F.linear(full_fused, down_weight, down_bias)
    mlp_output = broadcast_cpu(
        hidden_states,
        leader_output,
        src=tp_src_rank,
        comm_dtype=comm_dtype,
        group=tp_group,
    )

    return {
        "gate_out": gate_out,
        "up_out": up_out,
        "fused": fused_out,
        "mlp_output": mlp_output,
    }


def forward_mlp_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    math_mode: str = "orig",
) -> torch.Tensor:
    return trace_mlp_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=math_mode,
    )["mlp_output"]


def forward_decoder_layer(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_output = forward_attention(attn_input, bundle)
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_output = forward_mlp(mlp_input, bundle)
    return after_attn + mlp_output


def forward_decoder_layer_cached(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_output = forward_attention_cached(attn_input, bundle)
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_output = forward_mlp(mlp_input, bundle)
    return after_attn + mlp_output


def forward_decoder_layer_cached_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_output = forward_attention_cached_tp(
        attn_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=attn_math_mode,
    )
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_output = forward_mlp_tp(
        mlp_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=mlp_math_mode,
    )
    return after_attn + mlp_output


def trace_decoder_layer(layer_input: torch.Tensor, bundle: dict) -> dict:
    # 返回中间张量，方便逐层定位数值偏差来源。
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_trace = trace_attention(attn_input, bundle)
    attn_context = attn_trace["attn_context"]
    attn_output = attn_trace["attn_output"]
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_trace = trace_mlp(mlp_input, bundle)
    mlp_output = mlp_trace["mlp_output"]
    layer_output = after_attn + mlp_output

    return {
        "layer_input": layer_input,
        "attn_input": attn_input,
        "attn_context": attn_context,
        "attn_output": attn_output,
        "after_attn": after_attn,
        "mlp_input": mlp_input,
        "gate_out": mlp_trace["gate_out"],
        "up_out": mlp_trace["up_out"],
        "fused": mlp_trace["fused"],
        "mlp_output": mlp_output,
        "layer_output": layer_output,
    }


def trace_decoder_layer_cached(layer_input: torch.Tensor, bundle: dict) -> dict:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_trace = trace_attention_cached(attn_input, bundle)
    attn_context = attn_trace["attn_context"]
    attn_output = attn_trace["attn_output"]
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_trace = trace_mlp(mlp_input, bundle)
    mlp_output = mlp_trace["mlp_output"]
    layer_output = after_attn + mlp_output

    return {
        "layer_input": layer_input,
        "attn_input": attn_input,
        "attn_context": attn_context,
        "attn_output": attn_output,
        "current_key": attn_trace["current_key"],
        "current_value": attn_trace["current_value"],
        "full_key": attn_trace["full_key"],
        "full_value": attn_trace["full_value"],
        "after_attn": after_attn,
        "mlp_input": mlp_input,
        "gate_out": mlp_trace["gate_out"],
        "up_out": mlp_trace["up_out"],
        "fused": mlp_trace["fused"],
        "mlp_output": mlp_output,
        "layer_output": layer_output,
    }


def trace_decoder_layer_cached_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> dict:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_trace = trace_attention_cached_tp(
        attn_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=attn_math_mode,
    )
    attn_context = attn_trace["attn_context"]
    attn_output = attn_trace["attn_output"]
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_trace = trace_mlp_tp(
        mlp_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=mlp_math_mode,
    )
    mlp_output = mlp_trace["mlp_output"]
    layer_output = after_attn + mlp_output

    return {
        "layer_input": layer_input,
        "attn_input": attn_input,
        "attn_context": attn_context,
        "attn_output": attn_output,
        "current_key": attn_trace["current_key"],
        "current_value": attn_trace["current_value"],
        "full_key": attn_trace["full_key"],
        "full_value": attn_trace["full_value"],
        "after_attn": after_attn,
        "mlp_input": mlp_input,
        "gate_out": mlp_trace["gate_out"],
        "up_out": mlp_trace["up_out"],
        "fused": mlp_trace["fused"],
        "mlp_output": mlp_output,
        "layer_output": layer_output,
    }


def forward_decoder_layer_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_output = forward_attention_tp(
        attn_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=attn_math_mode,
    )
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_output = forward_mlp_tp(
        mlp_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=mlp_math_mode,
    )
    return after_attn + mlp_output


def trace_decoder_layer_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> dict:
    attn_input = rms_norm(layer_input, bundle["input_ln_weight"], bundle["input_ln_eps"])
    attn_trace = trace_attention_tp(
        attn_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=attn_math_mode,
    )
    attn_context = attn_trace["attn_context"]
    attn_output = attn_trace["attn_output"]
    after_attn = layer_input + attn_output

    mlp_input = rms_norm(after_attn, bundle["post_attn_ln_weight"], bundle["post_attn_ln_eps"])
    mlp_trace = trace_mlp_tp(
        mlp_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        math_mode=mlp_math_mode,
    )
    mlp_output = mlp_trace["mlp_output"]
    layer_output = after_attn + mlp_output

    return {
        "layer_input": layer_input,
        "attn_input": attn_input,
        "attn_context": attn_context,
        "attn_output": attn_output,
        "after_attn": after_attn,
        "mlp_input": mlp_input,
        "gate_out": mlp_trace["gate_out"],
        "up_out": mlp_trace["up_out"],
        "fused": mlp_trace["fused"],
        "mlp_output": mlp_output,
        "layer_output": layer_output,
    }


def forward_layer_range(hidden_states: torch.Tensor, range_bundle: dict) -> torch.Tensor:
    output = hidden_states
    for layer_bundle in range_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, range_bundle)
        output = forward_decoder_layer(output, runtime_bundle)
    return output


def forward_layer_range_tp(
    hidden_states: torch.Tensor,
    range_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    for layer_bundle in range_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, range_bundle)
        output = forward_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
    return output


def forward_text_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]

        output = forward_decoder_layer(output, runtime_bundle)
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))

    return output


def forward_text_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]

        output = forward_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))

    return output


def forward_text_embeddings(input_ids: torch.Tensor, bundle: dict) -> torch.Tensor:
    return F.embedding(input_ids, bundle["embed_tokens_weight"])


def trace_text_prefill_logits(layer_input: torch.Tensor, bundle: dict) -> dict:
    stage_output = forward_text_stage(layer_input, bundle)
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_prefill_logits(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_prefill_logits(layer_input, bundle)["logits"]


def forward_text_decode_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        output = forward_decoder_layer_cached(output, runtime_bundle)
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))
    return output


def forward_text_decode_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        output = forward_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))
    return output


def trace_text_decode_logits(layer_input: torch.Tensor, bundle: dict) -> dict:
    stage_output = forward_text_decode_stage(layer_input, bundle)
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_decode_logits(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_decode_logits(layer_input, bundle)["logits"]


def trace_text_decode_stage_with_runtime_cache(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    current_cache = cache_by_layer or {}
    updated_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = int(layer_bundle["layer_idx"])
        past_key, past_value = current_cache.get(
            layer_idx,
            (runtime_bundle.get("past_key"), runtime_bundle.get("past_value")),
        )
        runtime_bundle["past_key"] = past_key
        runtime_bundle["past_value"] = past_value

        layer_trace = trace_decoder_layer_cached(output, runtime_bundle)
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        updated_cache[layer_idx] = (
            layer_trace["full_key"].detach().clone(),
            layer_trace["full_value"].detach().clone(),
        )
        output = post_deepstack

    return {
        "stage_output": output,
        "layer_traces": traces,
        "cache_by_layer": updated_cache,
    }


def trace_text_decode_logits_with_runtime_cache(
    layer_input: torch.Tensor,
    bundle: dict,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    stage_trace = trace_text_decode_stage_with_runtime_cache(
        layer_input,
        bundle,
        cache_by_layer=cache_by_layer,
    )
    norm_output = rms_norm(
        stage_trace["stage_output"],
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_trace["stage_output"],
        "norm_output": norm_output,
        "logits": logits,
        "cache_by_layer": stage_trace["cache_by_layer"],
        "layer_traces": stage_trace["layer_traces"],
    }


def trace_text_decode_stage_tp_with_runtime_cache(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    current_cache = cache_by_layer or {}
    updated_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = int(layer_bundle["layer_idx"])
        past_key, past_value = current_cache.get(
            layer_idx,
            (runtime_bundle.get("past_key"), runtime_bundle.get("past_value")),
        )
        runtime_bundle["past_key"] = past_key
        runtime_bundle["past_value"] = past_value

        layer_trace = trace_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        updated_cache[layer_idx] = (
            layer_trace["full_key"].detach().clone(),
            layer_trace["full_value"].detach().clone(),
        )
        output = post_deepstack

    return {
        "stage_output": output,
        "layer_traces": traces,
        "cache_by_layer": updated_cache,
    }


def trace_text_decode_logits_tp_with_runtime_cache(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    stage_trace = trace_text_decode_stage_tp_with_runtime_cache(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
        cache_by_layer=cache_by_layer,
    )
    norm_output = rms_norm(
        stage_trace["stage_output"],
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_trace["stage_output"],
        "norm_output": norm_output,
        "logits": logits,
        "cache_by_layer": stage_trace["cache_by_layer"],
        "layer_traces": stage_trace["layer_traces"],
    }


def forward_text_decode_logits_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    stage_output = forward_text_decode_stage_tp(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    return F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )


def trace_text_prefill_stage_logits(hidden_states: torch.Tensor, bundle: dict) -> dict:
    hidden_stage_output = forward_text_stage(hidden_states, bundle)
    norm_output = rms_norm(
        hidden_stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "stage_input": hidden_states,
        "hidden_stage_output": hidden_stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def trace_text_prefill_stage_logits_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> dict:
    hidden_stage_output = forward_text_stage_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )
    norm_output = rms_norm(
        hidden_stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "stage_input": hidden_states,
        "hidden_stage_output": hidden_stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_prefill_stage_logits(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_prefill_stage_logits(hidden_states, bundle)["logits"]


def forward_text_prefill_stage_logits_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    return trace_text_prefill_stage_logits_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )["logits"]


def trace_text_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer(output, runtime_bundle)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_decode_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_cached(output, runtime_bundle)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_decode_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


# 兼容旧命名，避免已有实验脚本全部失效。
build_layer_runtime_bundle = compose_layer_bundle
replay_attn = forward_attention
replay_attn_tp = forward_attention_tp
replay_mlp = forward_mlp
replay_mlp_tp = forward_mlp_tp
replay_layer = forward_decoder_layer
replay_layer_trace = trace_decoder_layer
replay_layer_tp = forward_decoder_layer_tp
replay_layer_tp_trace = trace_decoder_layer_tp
replay_layer_range = forward_layer_range
replay_layer_range_tp = forward_layer_range_tp
