"""Debug replay runtime for running one captured text stage under pure TP."""

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.tp_debug import (
    TpDebugConfig,
    build_stage_traces,
)
from qwen3vl_tp_runtime.hexgen_core.stage import (
    get_stage_input,
    get_stage_output,
    run_stage,
    run_stage_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.capture.common import load_bundle, move_bundle
from qwen3vl_tp_runtime.models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


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
    bundle: dict,
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
    """Execute one captured text stage under TP and optionally collect direct/trace comparisons."""

    debug_config = debug_config or TpDebugConfig()
    reference_input = (
        reference_input_override if reference_input_override is not None else get_stage_input(bundle)
    )
    reference_output = get_stage_output(bundle)
    boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)

    direct_output = None
    direct_max = None
    direct_mean = None
    tp_direct_max = None
    tp_direct_mean = None
    if debug_config.needs_direct_output:
        direct_output = run_stage(stage_input, bundle)
        direct_max, direct_mean = tensor_diff_stats(direct_output, reference_output)

    stage_output = run_stage_tp(
        stage_input,
        bundle,
        rank=local_rank,
        world_size=tp_degree,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        tp_src_rank=leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
    )
    stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

    if direct_output is not None:
        tp_direct_max, tp_direct_mean = tensor_diff_stats(stage_output, direct_output)

    traces = None
    outlier_dump = None
    trace_summary = None
    if debug_config.needs_layer_trace:
        traces, outlier_dump, trace_summary = build_stage_traces(
            reference_input=reference_input,
            stage_input=stage_input,
            bundle=bundle,
            local_rank=local_rank,
            tp_degree=tp_degree,
            comm_dtype=comm_dtype,
            tp_group=tp_group,
            leader_rank=leader_rank,
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            dump_layer=debug_config.dump_layer,
            dump_topk=debug_config.dump_topk,
        )

    return {
        "input_shape": tuple(stage_input.shape),
        "output_shape": tuple(stage_output.shape),
        "boundary_max_diff": boundary_max,
        "boundary_mean_diff": boundary_mean,
        "direct_max_diff": direct_max,
        "direct_mean_diff": direct_mean,
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
        "tp_direct_max_diff": tp_direct_max,
        "tp_direct_mean_diff": tp_direct_mean,
        "traces": traces,
        "trace_summary": trace_summary,
        "outlier_dump": outlier_dump,
        "stage_output": stage_output,
    }


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

__all__ = []
