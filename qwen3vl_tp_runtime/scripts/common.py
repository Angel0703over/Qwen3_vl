"""Shared helpers for qwen3vl_tp_runtime command-line scripts."""

from __future__ import annotations

import gc

import torch

from qwen3vl_tp_runtime.models.qwen3vl import dtype_from_name, load_bundle, move_bundle


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def summarize_last_token_topk(logits: torch.Tensor, topk: int) -> list[dict[str, float | int]]:
    last_token_logits = logits[0, -1].detach().to(torch.float32)
    values, indices = torch.topk(last_token_logits, k=min(topk, last_token_logits.numel()))
    return [
        {
            "token_id": int(token_id),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def load_runtime_bundle(
    bundle_path: str,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[dict, torch.dtype]:
    bundle = load_bundle(bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "load_runtime_bundle",
    "release_unused_memory",
    "summarize_last_token_topk",
    "tensor_diff_stats",
]
