"""Shared helpers for Qwen3-VL replay execution kernels."""

from __future__ import annotations

import torch

from ..vision import apply_deepstack, get_deepstack_embeds


def compose_layer_state(layer_state: dict, stage_state: dict) -> dict:
    # 多层执行时，每层参数独立存放，共享的运行时张量在这里补齐。
    runtime_state = dict(layer_state)
    runtime_state["attention_mask"] = stage_state["attention_mask"]
    runtime_state["cos"] = stage_state["cos"]
    runtime_state["sin"] = stage_state["sin"]
    return runtime_state


def compose_layer_bundle(layer_bundle: dict, runtime_bundle: dict) -> dict:
    return compose_layer_state(layer_bundle, runtime_bundle)

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


__all__ = [
    "compose_layer_state",
    "apply_deepstack",
    "get_deepstack_embeds",
]
