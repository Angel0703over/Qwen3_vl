"""Selective tensor loading helpers backed by the weight index."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from qwen3vl_tp_runtime.models.qwen3vl.weights.index import ModelWeightIndex, load_model_weight_index
from qwen3vl_tp_runtime.models.qwen3vl.weights.planner import TensorSliceSpec


def load_tensors_by_name(
    model_path: str,
    tensor_names: Iterable[str],
    *,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype | None = None,
    strict: bool = True,
    tensor_slices: Mapping[str, tuple[TensorSliceSpec, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    index = load_model_weight_index(model_path)
    return load_tensors_from_index(
        index,
        tensor_names,
        device=device,
        compute_dtype=compute_dtype,
        strict=strict,
        tensor_slices=tensor_slices,
    )


def load_tensors_from_index(
    index: ModelWeightIndex,
    tensor_names: Iterable[str],
    *,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype | None = None,
    strict: bool = True,
    tensor_slices: Mapping[str, tuple[TensorSliceSpec, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    requested_names = [str(name) for name in tensor_names]
    grouped_names: dict[str, list[str]] = defaultdict(list)
    missing_names: list[str] = []
    for tensor_name in requested_names:
        if tensor_name in index.weight_map:
            grouped_names[index.weight_map[tensor_name]].append(tensor_name)
        else:
            missing_names.append(tensor_name)

    if strict and missing_names:
        raise KeyError(f"以下 tensor 不在权重索引里: {missing_names}")

    target_device = torch.device(device)
    loaded: dict[str, torch.Tensor] = {}
    for shard_name, shard_tensor_names in grouped_names.items():
        shard_path = Path(index.model_path) / shard_name
        shard_payload = _load_shard_subset(
            shard_path,
            shard_tensor_names,
            format_name=index.format,
            tensor_slices=tensor_slices,
        )
        for tensor_name, tensor in shard_payload.items():
            out = tensor.detach().to(device=target_device)
            if compute_dtype is not None and out.is_floating_point():
                out = out.to(dtype=compute_dtype)
            loaded[tensor_name] = out
    return loaded


def _load_shard_subset(
    shard_path: Path,
    tensor_names: list[str],
    *,
    format_name: str,
    tensor_slices: Mapping[str, tuple[TensorSliceSpec, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    if format_name == "safetensors":
        safe_open = _resolve_safe_open()
        loaded: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for tensor_name in tensor_names:
                slice_specs = None if tensor_slices is None else tensor_slices.get(tensor_name)
                if slice_specs:
                    loaded[tensor_name] = handle.get_slice(tensor_name)[_build_tensor_index(slice_specs)]
                else:
                    loaded[tensor_name] = handle.get_tensor(tensor_name)
        return loaded

    state_dict = _normalize_torch_state_dict(torch.load(shard_path, map_location="cpu"))
    return {
        tensor_name: _apply_tensor_slices(
            state_dict[tensor_name],
            None if tensor_slices is None else tensor_slices.get(tensor_name),
        )
        for tensor_name in tensor_names
        if tensor_name in state_dict
    }


def _apply_tensor_slices(
    tensor: torch.Tensor,
    slice_specs: tuple[TensorSliceSpec, ...] | None,
) -> torch.Tensor:
    if not slice_specs:
        return tensor
    return tensor[_build_tensor_index(slice_specs)]


def _build_tensor_index(slice_specs: tuple[TensorSliceSpec, ...]) -> tuple[slice, ...]:
    max_dim = max(int(spec.dim) for spec in slice_specs)
    index = [slice(None)] * (max_dim + 1)
    for spec in slice_specs:
        index[int(spec.dim)] = slice(int(spec.start), int(spec.end))
    return tuple(index)


def _normalize_torch_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"torch 权重文件内容不是 dict，当前拿到 {type(payload)!r}")
    return payload


def _resolve_safe_open():
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "当前环境没有可用的 safetensors 支持，无法按参数名读取 safetensors 权重。"
        ) from exc
    return safe_open


__all__ = [
    "load_tensors_by_name",
    "load_tensors_from_index",
]
