"""Captured-bundle TP replay helpers kept out of the direct TP backend module."""

from __future__ import annotations

import torch

from ..hexgen_core.modules.tensor_parallel import run_stage_state_tp
from ..hexgen_core.stage import get_stage_input
from ..models.qwen3vl.capture.common import load_bundle, move_bundle
from ..models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype
from .tp_debug import TpDebugConfig, build_stage_traces, tensor_diff_stats


def load_text_stage_bundle(
    bundle_path: str,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    """Load one captured text-stage bundle and move it into the requested runtime dtype."""

    bundle = load_bundle(bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def run_text_tensor_parallel_stage(
    *,
    stage_input: torch.Tensor,
    bundle: dict | None = None,
    stage_state: dict | None = None,
    reference_input_override: torch.Tensor | None = None,
    local_rank: int,
    tp_degree: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    leader_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    debug_config: TpDebugConfig | None = None,
) -> dict:
    if stage_state is None:
        if bundle is None:
            raise ValueError("run_text_tensor_parallel_stage 需要 stage_state。")
        stage_state = bundle
    return run_stage_state_tp(
        stage_input=stage_input,
        stage_state=stage_state,
        reference_input_override=reference_input_override,
        local_rank=local_rank,
        tp_degree=tp_degree,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        leader_rank=leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        debug_config=debug_config,
    )


class TextTensorParallelRunner:
    """Stateful rank runner for one captured text stage executed under pure TP."""

    def __init__(
        self,
        bundle_path: str,
        device: torch.device,
        compute_dtype_arg: str,
        comm_dtype_arg: str,
        tp_attn_math_mode: str = "orig",
        tp_mlp_math_mode: str = "orig",
        debug_config: TpDebugConfig | None = None,
    ) -> None:
        self.bundle_path = bundle_path
        self.device = device
        self.compute_dtype_arg = compute_dtype_arg
        self.comm_dtype_arg = comm_dtype_arg
        self.tp_attn_math_mode = tp_attn_math_mode
        self.tp_mlp_math_mode = tp_mlp_math_mode
        self.debug_config = debug_config or TpDebugConfig()

    def run_rank(self, rank: int, world_size: int) -> dict:
        bundle, compute_dtype = load_text_stage_bundle(
            self.bundle_path,
            self.device,
            self.compute_dtype_arg,
        )
        comm_dtype = resolve_comm_dtype(self.comm_dtype_arg, compute_dtype)
        stage_input = get_stage_input(bundle)
        stats = run_text_tensor_parallel_stage(
            stage_input=stage_input,
            bundle=bundle,
            reference_input_override=None,
            local_rank=rank,
            tp_degree=world_size,
            comm_dtype=comm_dtype,
            tp_attn_math_mode=self.tp_attn_math_mode,
            tp_mlp_math_mode=self.tp_mlp_math_mode,
            debug_config=self.debug_config,
        )
        first_layer = bundle["layers"][0]
        stats.pop("stage_output")
        stats.update(
            {
                "rank": rank,
                "world_size": world_size,
                "start_idx": bundle["start_idx"],
                "end_idx": bundle["end_idx"],
                "num_layers": len(bundle["layers"]),
                "device": str(self.device),
                "comm_dtype": str(comm_dtype),
                "tp_attn_math_mode": self.tp_attn_math_mode,
                "tp_mlp_math_mode": self.tp_mlp_math_mode,
                "num_heads": first_layer["num_attention_heads"],
                "num_kv_heads": first_layer["num_key_value_heads"],
                "local_q_heads": first_layer["num_attention_heads"] // world_size,
                "local_kv_heads": first_layer["num_key_value_heads"] // world_size,
                "head_dim": first_layer["head_dim"],
                "hidden_act": first_layer["hidden_act"],
                "attn_impl": first_layer.get("attn_implementation", "unknown"),
                "deepstack_layer_indices": bundle.get("deepstack_layer_indices", []),
                "bundle_dtype": bundle["save_dtype"],
                "original_input_dtype": bundle["original_input_dtype"],
                "original_input_device": bundle["original_input_device"],
            }
        )
        return stats


def run_text_tensor_parallel_rank(
    *,
    rank: int,
    world_size: int,
    bundle_path: str,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    debug_config: TpDebugConfig | None = None,
) -> dict:
    runner = TextTensorParallelRunner(
        bundle_path=bundle_path,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
        comm_dtype_arg=comm_dtype_arg,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        debug_config=debug_config,
    )
    return runner.run_rank(rank, world_size)


DEBUG_REPLAY_EXPORTS = [
    "tensor_diff_stats",
    "load_text_stage_bundle",
    "build_stage_traces",
    "run_text_tensor_parallel_stage",
    "TextTensorParallelRunner",
    "run_text_tensor_parallel_rank",
]

__all__ = [*DEBUG_REPLAY_EXPORTS]
