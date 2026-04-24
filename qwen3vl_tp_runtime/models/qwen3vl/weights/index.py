"""Model weight index helpers for stage-only and shard-only loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch


_SAFE_TENSORS_INDEX = "model.safetensors.index.json"
_TORCH_BIN_INDEX = "pytorch_model.bin.index.json"
_SAFE_TENSORS_FILE = "model.safetensors"
_TORCH_BIN_FILE = "pytorch_model.bin"


@dataclass(slots=True)
class ModelWeightIndex:
    """Resolved mapping from parameter name to shard file."""

    model_path: str
    format: str
    index_file: str | None
    weight_map: dict[str, str]
    metadata: dict[str, Any]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.weight_map))

    @property
    def shard_files(self) -> tuple[str, ...]:
        return tuple(sorted({self.weight_map[name] for name in self.weight_map}))

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self.weight_map

    def resolve_tensor_file(self, tensor_name: str) -> str:
        try:
            shard_name = self.weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(f"权重索引里没有 tensor={tensor_name!r}") from exc
        return str(Path(self.model_path) / shard_name)

    def files_for_tensors(self, tensor_names: Iterable[str]) -> tuple[str, ...]:
        shard_names = {
            self.weight_map[tensor_name]
            for tensor_name in tensor_names
            if tensor_name in self.weight_map
        }
        return tuple(sorted(str(Path(self.model_path) / shard_name) for shard_name in shard_names))


def load_model_weight_index(model_path: str) -> ModelWeightIndex:
    """Load a HF-style weight index, or synthesize one for a single-file checkpoint."""

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_path 不存在: {model_path}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_path 需要是目录: {model_path}")

    for filename, format_name in (
        (_SAFE_TENSORS_INDEX, "safetensors"),
        (_TORCH_BIN_INDEX, "torch_bin"),
    ):
        index_path = model_dir / filename
        if index_path.exists():
            payload = json.loads(index_path.read_text())
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index_path} 里缺少有效的 weight_map。")
            return ModelWeightIndex(
                model_path=str(model_dir),
                format=format_name,
                index_file=str(index_path),
                weight_map={str(name): str(shard_name) for name, shard_name in weight_map.items()},
                metadata=dict(payload.get("metadata", {})),
            )

    for filename, format_name in (
        (_SAFE_TENSORS_FILE, "safetensors"),
        (_TORCH_BIN_FILE, "torch_bin"),
    ):
        checkpoint_path = model_dir / filename
        if checkpoint_path.exists():
            tensor_names = _list_single_file_tensor_names(checkpoint_path, format_name)
            return ModelWeightIndex(
                model_path=str(model_dir),
                format=format_name,
                index_file=None,
                weight_map={tensor_name: filename for tensor_name in tensor_names},
                metadata={},
            )

    raise FileNotFoundError(
        "没有在 model_path 下找到支持的 HF 权重索引或单文件权重。"
        f" 已检查: {_SAFE_TENSORS_INDEX}, {_TORCH_BIN_INDEX}, {_SAFE_TENSORS_FILE}, {_TORCH_BIN_FILE}"
    )


def _list_single_file_tensor_names(checkpoint_path: Path, format_name: str) -> list[str]:
    if format_name == "safetensors":
        safe_open = _resolve_safe_open()
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            return sorted(handle.keys())

    state_dict = _normalize_torch_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return sorted(str(name) for name in state_dict)


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
            "当前环境没有可用的 safetensors 支持，无法解析 safetensors 权重文件。"
        ) from exc
    return safe_open


__all__ = [
    "ModelWeightIndex",
    "load_model_weight_index",
]
