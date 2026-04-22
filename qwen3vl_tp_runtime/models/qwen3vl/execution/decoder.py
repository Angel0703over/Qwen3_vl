"""Decoder-layer replay kernels composed from attention and MLP pieces."""

from __future__ import annotations

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution.attention import (
    forward_attention,
    forward_attention_cached,
    forward_attention_cached_tp,
    forward_attention_tp,
    trace_attention,
    trace_attention_cached,
    trace_attention_cached_tp,
    trace_attention_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution.mlp import (
    forward_mlp,
    forward_mlp_tp,
    trace_mlp,
    trace_mlp_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.functional import rms_norm


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


__all__ = [
    "forward_decoder_layer",
    "forward_decoder_layer_cached",
    "forward_decoder_layer_cached_tp",
    "trace_decoder_layer",
    "trace_decoder_layer_cached",
    "trace_decoder_layer_cached_tp",
    "forward_decoder_layer_tp",
    "trace_decoder_layer_tp",
]
