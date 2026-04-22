"""MLP replay kernels for Qwen3-VL text decoder layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from qwen3vl_tp_runtime.hexgen_core.distributed import all_gather_cpu, broadcast_cpu
from qwen3vl_tp_runtime.models.qwen3vl.execution.common import (
    _cast_optional_tensor,
    _resolve_tp_math_dtype,
)


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
    orig_dtype, _ = _resolve_tp_math_dtype(hidden_states, math_mode)
    device = hidden_states.device
    if intermediate_size % world_size != 0:
        raise ValueError("当前 TP MLP 要求 intermediate_size 能被 world_size 整除。")

    act_fn = ACT2FN[bundle["hidden_act"]]
    shard = intermediate_size // world_size
    start = rank * shard
    end = (rank + 1) * shard

    x = hidden_states.to(dtype=orig_dtype)
    gate_weight = bundle["gate_weight"].to(device=device, dtype=orig_dtype)
    gate_bias = _cast_optional_tensor(bundle["gate_bias"], device=device, dtype=orig_dtype)
    up_weight = bundle["up_weight"].to(device=device, dtype=orig_dtype)
    up_bias = _cast_optional_tensor(bundle["up_bias"], device=device, dtype=orig_dtype)
    down_weight = bundle["down_weight"].to(device=device, dtype=orig_dtype)
    down_bias = _cast_optional_tensor(bundle["down_bias"], device=device, dtype=orig_dtype)

    full_gate_out = F.linear(x, gate_weight, gate_bias)
    full_up_out = F.linear(x, up_weight, up_bias)
    gate_out = full_gate_out[..., start:end]
    up_out = full_up_out[..., start:end]
    fused_out = act_fn(gate_out) * up_out
    gathered_fused = all_gather_cpu(
        fused_out,
        device,
        orig_dtype,
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


__all__ = [
    "forward_mlp",
    "trace_mlp",
    "trace_mlp_tp",
    "forward_mlp_tp",
]
