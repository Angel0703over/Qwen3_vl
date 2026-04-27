"""Debug-only TP compare/trace helpers kept separate from the main TP runtime path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen3vl_tp_runtime.hexgen_core.stage import trace_stage, trace_stage_tp


@dataclass(frozen=True)
class TpDebugConfig:
    compare_direct: bool = False
    trace_layers: bool = False
    dump_layer: int | None = None
    dump_topk: int = 5

    @property
    def debug_mode(self) -> bool:
        return self.compare_direct or self.trace_layers or self.dump_layer is not None

    @property
    def needs_direct_output(self) -> bool:
        return self.debug_mode

    @property
    def needs_layer_trace(self) -> bool:
        return self.trace_layers or self.dump_layer is not None

    def to_summary_fields(self) -> dict[str, bool | int | None]:
        return {
            "debug_mode": self.debug_mode,
            "compare_direct": self.compare_direct,
            "trace_layers": self.trace_layers,
            "dump_layer": self.dump_layer,
            "dump_topk": self.dump_topk,
        }


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def _flat_index_to_tuple(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    indices = []
    for dim in reversed(shape):
        indices.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(indices))


def _build_tensor_outliers(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    topk: int,
    last_dim_offset: int = 0,
) -> list[dict]:
    diff = (lhs - rhs).abs().reshape(-1).to(torch.float32)
    if diff.numel() == 0:
        return []

    k = min(topk, diff.numel())
    values, flat_indices = torch.topk(diff, k=k)
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    shape = tuple(lhs.shape)

    outliers = []
    for value, flat_idx in zip(values.tolist(), flat_indices.tolist()):
        index = _flat_index_to_tuple(flat_idx, shape)
        if last_dim_offset and index:
            index = (*index[:-1], index[-1] + last_dim_offset)
        outliers.append(
            {
                "index": index,
                "lhs": lhs_flat[flat_idx].item(),
                "rhs": rhs_flat[flat_idx].item(),
                "abs_diff": value,
            }
        )
    return outliers


def _align_tp_trace_pair(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if tp_tensor.shape == direct_tensor.shape:
        return tp_tensor, direct_tensor, 0

    if tp_tensor.dim() != direct_tensor.dim():
        raise ValueError(
            "TP trace tensor 维度和 direct trace tensor 不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)}"
        )
    if tp_tensor.shape[:-1] != direct_tensor.shape[:-1]:
        raise ValueError(
            "TP trace tensor 和 direct trace tensor 的前置维度不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)}"
        )
    if tp_degree <= 0:
        raise ValueError(f"tp_degree 必须是正整数，当前拿到 {tp_degree}。")
    if direct_tensor.shape[-1] % tp_degree != 0:
        raise ValueError(
            "direct trace tensor 的最后一维不能按 TP 度数整除，"
            f"direct_shape={tuple(direct_tensor.shape)} tp_degree={tp_degree}"
        )

    shard = direct_tensor.shape[-1] // tp_degree
    if tp_tensor.shape[-1] != shard:
        raise ValueError(
            "TP trace tensor 的最后一维和按 TP 切分后的 shard 大小不一致，"
            f"tp_shape={tuple(tp_tensor.shape)} direct_shape={tuple(direct_tensor.shape)} "
            f"tp_degree={tp_degree}"
        )

    start = local_rank * shard
    end = start + shard
    return tp_tensor, direct_tensor[..., start:end], start


def _tp_vs_direct_stats(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
) -> tuple[float, float]:
    aligned_tp, aligned_direct, _ = _align_tp_trace_pair(
        tp_tensor,
        direct_tensor,
        local_rank=local_rank,
        tp_degree=tp_degree,
    )
    return tensor_diff_stats(aligned_tp, aligned_direct)


def _tp_vs_direct_outliers(
    tp_tensor: torch.Tensor,
    direct_tensor: torch.Tensor,
    *,
    local_rank: int,
    tp_degree: int,
    topk: int,
) -> list[dict]:
    aligned_tp, aligned_direct, last_dim_offset = _align_tp_trace_pair(
        tp_tensor,
        direct_tensor,
        local_rank=local_rank,
        tp_degree=tp_degree,
    )
    return _build_tensor_outliers(
        aligned_tp,
        aligned_direct,
        topk,
        last_dim_offset=last_dim_offset,
    )


def _build_layer_outlier_dump(
    reference_trace: dict,
    direct_trace: dict,
    tp_trace: dict,
    local_rank: int,
    tp_degree: int,
    topk: int,
) -> dict:
    return {
        "layer_idx": reference_trace["layer_idx"],
        "deepstack_applied": reference_trace["deepstack_applied"],
        "direct_vs_ref": {
            "layer_input": _build_tensor_outliers(direct_trace["layer_input"], reference_trace["layer_input"], topk),
            "attn_input": _build_tensor_outliers(direct_trace["attn_input"], reference_trace["attn_input"], topk),
            "attn_context": _build_tensor_outliers(direct_trace["attn_context"], reference_trace["attn_context"], topk),
            "attn_output": _build_tensor_outliers(direct_trace["attn_output"], reference_trace["attn_output"], topk),
            "after_attn": _build_tensor_outliers(direct_trace["after_attn"], reference_trace["after_attn"], topk),
            "mlp_input": _build_tensor_outliers(direct_trace["mlp_input"], reference_trace["mlp_input"], topk),
            "gate_out": _build_tensor_outliers(direct_trace["gate_out"], reference_trace["gate_out"], topk),
            "up_out": _build_tensor_outliers(direct_trace["up_out"], reference_trace["up_out"], topk),
            "fused": _build_tensor_outliers(direct_trace["fused"], reference_trace["fused"], topk),
            "mlp_output": _build_tensor_outliers(direct_trace["mlp_output"], reference_trace["mlp_output"], topk),
            "layer_output": _build_tensor_outliers(direct_trace["layer_output"], reference_trace["layer_output"], topk),
        },
        "tp_vs_direct": {
            "layer_input": _tp_vs_direct_outliers(
                tp_trace["layer_input"],
                direct_trace["layer_input"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "attn_input": _tp_vs_direct_outliers(
                tp_trace["attn_input"],
                direct_trace["attn_input"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "attn_context": _tp_vs_direct_outliers(
                tp_trace["attn_context"],
                direct_trace["attn_context"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "attn_output": _tp_vs_direct_outliers(
                tp_trace["attn_output"],
                direct_trace["attn_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "after_attn": _tp_vs_direct_outliers(
                tp_trace["after_attn"],
                direct_trace["after_attn"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "mlp_input": _tp_vs_direct_outliers(
                tp_trace["mlp_input"],
                direct_trace["mlp_input"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "gate_out": _tp_vs_direct_outliers(
                tp_trace["gate_out"],
                direct_trace["gate_out"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "up_out": _tp_vs_direct_outliers(
                tp_trace["up_out"],
                direct_trace["up_out"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "fused": _tp_vs_direct_outliers(
                tp_trace["fused"],
                direct_trace["fused"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "mlp_output": _tp_vs_direct_outliers(
                tp_trace["mlp_output"],
                direct_trace["mlp_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
            "layer_output": _tp_vs_direct_outliers(
                tp_trace["layer_output"],
                direct_trace["layer_output"],
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=topk,
            ),
        },
    }


_TRACE_DIFF_COMPONENTS = (
    "layer_input",
    "attn_input",
    "attn_context",
    "attn_output",
    "after_attn",
    "mlp_input",
    "gate_out",
    "up_out",
    "fused",
    "mlp_output",
    "layer_output",
    "post_deepstack",
)


def _summarize_trace_family(traces: list[dict], family_key: str) -> dict:
    first_nonzero_layer_idx = None
    first_nonzero_component = None
    first_nonzero_max_diff = None
    max_layer_idx = None
    max_component = None
    max_diff = 0.0

    for trace in traces:
        layer_idx = int(trace["layer_idx"])
        layer_first_component = None
        layer_first_value = 0.0

        for component in _TRACE_DIFF_COMPONENTS:
            value = float(trace[family_key][component][0])
            if value > max_diff:
                max_diff = value
                max_layer_idx = layer_idx
                max_component = component
            if value > 0.0 and layer_first_component is None:
                layer_first_component = component
                layer_first_value = value

        if layer_first_component is not None and first_nonzero_layer_idx is None:
            first_nonzero_layer_idx = layer_idx
            first_nonzero_component = layer_first_component
            first_nonzero_max_diff = layer_first_value

    return {
        "first_nonzero_layer_idx": first_nonzero_layer_idx,
        "first_nonzero_component": first_nonzero_component,
        "first_nonzero_max_diff": first_nonzero_max_diff,
        "max_layer_idx": max_layer_idx,
        "max_component": max_component,
        "max_diff": max_diff,
    }


def _summarize_stage_traces(traces: list[dict]) -> dict:
    direct_vs_ref = _summarize_trace_family(traces, "direct_vs_ref")
    tp_vs_direct = _summarize_trace_family(traces, "tp_vs_direct")

    auto_dump_layer_idx = (
        tp_vs_direct["first_nonzero_layer_idx"]
        or direct_vs_ref["first_nonzero_layer_idx"]
        or tp_vs_direct["max_layer_idx"]
        or direct_vs_ref["max_layer_idx"]
    )
    return {
        "direct_vs_ref": direct_vs_ref,
        "tp_vs_direct": tp_vs_direct,
        "auto_dump_layer_idx": auto_dump_layer_idx,
    }


def build_stage_traces(
    *,
    reference_input: torch.Tensor,
    stage_input: torch.Tensor,
    bundle: dict,
    local_rank: int,
    tp_degree: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    leader_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    dump_layer: int | None = None,
    dump_topk: int = 5,
) -> tuple[list[dict], dict | None, dict]:
    """Build layer-wise direct-vs-reference and TP-vs-direct trace summaries for one text stage."""

    reference_traces = trace_stage(reference_input, bundle)
    direct_traces = trace_stage(stage_input, bundle)
    tp_traces = trace_stage_tp(
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

    traces = []
    trace_triplets = []
    outlier_dump = None
    for reference_trace, direct_trace, tp_trace in zip(reference_traces, direct_traces, tp_traces):
        layer_idx = reference_trace["layer_idx"]
        trace_triplets.append((reference_trace, direct_trace, tp_trace))
        traces.append(
            {
                "layer_idx": layer_idx,
                "deepstack_applied": reference_trace["deepstack_applied"],
                "direct_vs_ref": {
                    "layer_input": tensor_diff_stats(direct_trace["layer_input"], reference_trace["layer_input"]),
                    "attn_input": tensor_diff_stats(direct_trace["attn_input"], reference_trace["attn_input"]),
                    "attn_context": tensor_diff_stats(direct_trace["attn_context"], reference_trace["attn_context"]),
                    "attn_output": tensor_diff_stats(direct_trace["attn_output"], reference_trace["attn_output"]),
                    "after_attn": tensor_diff_stats(direct_trace["after_attn"], reference_trace["after_attn"]),
                    "mlp_input": tensor_diff_stats(direct_trace["mlp_input"], reference_trace["mlp_input"]),
                    "gate_out": tensor_diff_stats(direct_trace["gate_out"], reference_trace["gate_out"]),
                    "up_out": tensor_diff_stats(direct_trace["up_out"], reference_trace["up_out"]),
                    "fused": tensor_diff_stats(direct_trace["fused"], reference_trace["fused"]),
                    "mlp_output": tensor_diff_stats(direct_trace["mlp_output"], reference_trace["mlp_output"]),
                    "layer_output": tensor_diff_stats(direct_trace["layer_output"], reference_trace["layer_output"]),
                    "post_deepstack": tensor_diff_stats(
                        direct_trace["post_deepstack"],
                        reference_trace["post_deepstack"],
                    ),
                },
                "tp_vs_direct": {
                    "layer_input": _tp_vs_direct_stats(
                        tp_trace["layer_input"],
                        direct_trace["layer_input"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "attn_input": _tp_vs_direct_stats(
                        tp_trace["attn_input"],
                        direct_trace["attn_input"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "attn_context": _tp_vs_direct_stats(
                        tp_trace["attn_context"],
                        direct_trace["attn_context"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "attn_output": _tp_vs_direct_stats(
                        tp_trace["attn_output"],
                        direct_trace["attn_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "after_attn": _tp_vs_direct_stats(
                        tp_trace["after_attn"],
                        direct_trace["after_attn"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "mlp_input": _tp_vs_direct_stats(
                        tp_trace["mlp_input"],
                        direct_trace["mlp_input"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "gate_out": _tp_vs_direct_stats(
                        tp_trace["gate_out"],
                        direct_trace["gate_out"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "up_out": _tp_vs_direct_stats(
                        tp_trace["up_out"],
                        direct_trace["up_out"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "fused": _tp_vs_direct_stats(
                        tp_trace["fused"],
                        direct_trace["fused"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "mlp_output": _tp_vs_direct_stats(
                        tp_trace["mlp_output"],
                        direct_trace["mlp_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "layer_output": _tp_vs_direct_stats(
                        tp_trace["layer_output"],
                        direct_trace["layer_output"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                    "post_deepstack": _tp_vs_direct_stats(
                        tp_trace["post_deepstack"],
                        direct_trace["post_deepstack"],
                        local_rank=local_rank,
                        tp_degree=tp_degree,
                    ),
                },
            }
        )
        if dump_layer is not None and dump_layer >= 0 and layer_idx == dump_layer:
            outlier_dump = _build_layer_outlier_dump(
                reference_trace=reference_trace,
                direct_trace=direct_trace,
                tp_trace=tp_trace,
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=dump_topk,
            )
    trace_summary = _summarize_stage_traces(traces)
    resolved_dump_layer = dump_layer
    if resolved_dump_layer is not None and resolved_dump_layer < 0:
        resolved_dump_layer = trace_summary["auto_dump_layer_idx"]
    if outlier_dump is None and resolved_dump_layer is not None:
        for reference_trace, direct_trace, tp_trace in trace_triplets:
            if int(reference_trace["layer_idx"]) != int(resolved_dump_layer):
                continue
            outlier_dump = _build_layer_outlier_dump(
                reference_trace=reference_trace,
                direct_trace=direct_trace,
                tp_trace=tp_trace,
                local_rank=local_rank,
                tp_degree=tp_degree,
                topk=dump_topk,
            )
            break
    return traces, outlier_dump, trace_summary


__all__ = [
    "TpDebugConfig",
    "build_stage_traces",
]
