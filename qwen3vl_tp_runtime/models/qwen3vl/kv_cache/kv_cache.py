"""Stage-local KV cache helpers for runtime-only generate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class LayerKVCache:
    """Preallocated key/value buffers for one decoder layer."""

    max_seq_len: int
    key_buffer: torch.Tensor | None = None
    value_buffer: torch.Tensor | None = None
    current_length: int = 0
    append_count: int = 0

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "LayerKVCache expects key/value tensors with shape "
                f"[batch, heads, seq, head_dim], got key={tuple(key.shape)} value={tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: key={tuple(key.shape)} value={tuple(value.shape)}")
        append_len = int(key.shape[-2])
        if append_len <= 0:
            raise ValueError(f"append_len must be positive, got {append_len}")
        next_length = self.current_length + append_len
        if next_length > self.max_seq_len:
            raise RuntimeError(
                "LayerKVCache overflow: "
                f"current={self.current_length} append={append_len} next={next_length} max={self.max_seq_len}"
            )

        self._ensure_buffers(key, value)
        assert self.key_buffer is not None
        assert self.value_buffer is not None
        with torch.no_grad():
            self.key_buffer.narrow(-2, self.current_length, append_len).copy_(key)
            self.value_buffer.narrow(-2, self.current_length, append_len).copy_(value)
        self.current_length = next_length
        self.append_count += 1
        return self.view()

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_buffer is None or self.value_buffer is None:
            raise RuntimeError("LayerKVCache has not been initialized yet.")
        return (
            self.key_buffer.narrow(-2, 0, self.current_length),
            self.value_buffer.narrow(-2, 0, self.current_length),
        )

    def compact_prefix(self, keep_indices: Sequence[int], *, original_length: int) -> dict:
        """Compact the already-written prefill prefix in-place."""

        if self.key_buffer is None or self.value_buffer is None:
            raise RuntimeError("LayerKVCache has not been initialized yet.")
        original_length = int(original_length)
        if original_length <= 0:
            raise ValueError(f"original_length must be positive, got {original_length}")
        if self.current_length != original_length:
            raise ValueError(
                "LayerKVCache compact_prefix expects prefill-only cache: "
                f"current_length={self.current_length} original_length={original_length}"
            )
        keep_indices = [int(index) for index in keep_indices]
        if not keep_indices:
            raise ValueError("keep_indices must not be empty.")
        if keep_indices != sorted(keep_indices):
            raise ValueError("keep_indices must be sorted.")
        if keep_indices[0] < 0 or keep_indices[-1] >= original_length:
            raise ValueError(
                "keep_indices out of prefill range: "
                f"first={keep_indices[0]} last={keep_indices[-1]} original_length={original_length}"
            )
        if len(set(keep_indices)) != len(keep_indices):
            raise ValueError("keep_indices must not contain duplicates.")

        before_active_bytes = self.active_tensor_bytes()
        compact_length = len(keep_indices)
        if compact_length != original_length:
            keep_index = torch.tensor(
                keep_indices,
                device=self.key_buffer.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                compact_key = self.key_buffer.index_select(-2, keep_index)
                compact_value = self.value_buffer.index_select(-2, keep_index)
                self.key_buffer.narrow(-2, 0, compact_length).copy_(compact_key)
                self.value_buffer.narrow(-2, 0, compact_length).copy_(compact_value)
            self.current_length = compact_length
        after_active_bytes = self.active_tensor_bytes()
        return {
            "original_length": original_length,
            "compact_length": compact_length,
            "dropped_token_count": original_length - compact_length,
            "active_tensor_bytes_before": before_active_bytes,
            "active_tensor_bytes_after": after_active_bytes,
            "active_tensor_bytes_saved": max(0, before_active_bytes - after_active_bytes),
        }

    def active_tensor_bytes(self) -> int:
        if self.key_buffer is None or self.value_buffer is None:
            return 0
        key_token_stride = self.key_buffer.numel() // int(self.key_buffer.shape[-2])
        value_token_stride = self.value_buffer.numel() // int(self.value_buffer.shape[-2])
        return int(
            self.current_length
            * (
                key_token_stride * self.key_buffer.element_size()
                + value_token_stride * self.value_buffer.element_size()
            )
        )

    def _ensure_buffers(self, key: torch.Tensor, value: torch.Tensor) -> None:
        expected_shape = (*key.shape[:-2], self.max_seq_len, key.shape[-1])
        if self.key_buffer is None:
            self.key_buffer = key.new_empty(expected_shape)
            self.value_buffer = value.new_empty(expected_shape)
            return
        if tuple(self.key_buffer.shape) != tuple(expected_shape):
            raise ValueError(
                "LayerKVCache buffer shape mismatch: "
                f"buffer={tuple(self.key_buffer.shape)} expected={tuple(expected_shape)}"
            )
        if self.key_buffer.device != key.device or self.key_buffer.dtype != key.dtype:
            raise ValueError(
                "LayerKVCache key dtype/device mismatch: "
                f"buffer=({self.key_buffer.device}, {self.key_buffer.dtype}) "
                f"key=({key.device}, {key.dtype})"
            )
        if self.value_buffer.device != value.device or self.value_buffer.dtype != value.dtype:
            raise ValueError(
                "LayerKVCache value dtype/device mismatch: "
                f"buffer=({self.value_buffer.device}, {self.value_buffer.dtype}) "
                f"value=({value.device}, {value.dtype})"
            )


class StageKVCache:
    """Stage-local collection of per-layer KV caches."""

    def __init__(self, *, max_seq_len: int) -> None:
        if int(max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = int(max_seq_len)
        self._layers: dict[int, LayerKVCache] = {}

    def get_or_create(self, layer_idx: int) -> LayerKVCache:
        key = int(layer_idx)
        layer_cache = self._layers.get(key)
        if layer_cache is None:
            layer_cache = LayerKVCache(max_seq_len=self.max_seq_len)
            self._layers[key] = layer_cache
        return layer_cache

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_or_create(layer_idx).append(key, value)

    def as_cache_by_layer(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {layer_idx: layer_cache.view() for layer_idx, layer_cache in self._layers.items()}

    def compact_prefix(self, keep_indices: Sequence[int], *, original_length: int) -> dict:
        if not self._layers:
            raise RuntimeError("StageKVCache has no allocated layers to compact.")
        per_layer: dict[int, dict] = {}
        for layer_idx, layer_cache in self._layers.items():
            per_layer[layer_idx] = layer_cache.compact_prefix(
                keep_indices,
                original_length=original_length,
            )
        return {
            "layer_count": len(per_layer),
            "layers": per_layer,
            "original_length": int(original_length),
            "compact_length": len(keep_indices),
            "dropped_token_count": int(original_length) - len(keep_indices),
            "active_tensor_bytes_before": sum(
                int(layer["active_tensor_bytes_before"]) for layer in per_layer.values()
            ),
            "active_tensor_bytes_after": sum(
                int(layer["active_tensor_bytes_after"]) for layer in per_layer.values()
            ),
            "active_tensor_bytes_saved": sum(
                int(layer["active_tensor_bytes_saved"]) for layer in per_layer.values()
            ),
        }

    def summary(self) -> dict:
        allocated_layers = len(self._layers)
        append_count = sum(layer.append_count for layer in self._layers.values())
        tensor_bytes = 0
        active_tensor_bytes = 0
        lengths: dict[int, int] = {}
        for layer_idx, layer_cache in self._layers.items():
            lengths[layer_idx] = int(layer_cache.current_length)
            active_tensor_bytes += layer_cache.active_tensor_bytes()
            if layer_cache.key_buffer is not None:
                tensor_bytes += int(layer_cache.key_buffer.numel() * layer_cache.key_buffer.element_size())
            if layer_cache.value_buffer is not None:
                tensor_bytes += int(layer_cache.value_buffer.numel() * layer_cache.value_buffer.element_size())
        return {
            "max_seq_len": self.max_seq_len,
            "allocated_layers": allocated_layers,
            "append_count": append_count,
            "tensor_bytes": tensor_bytes,
            "active_tensor_bytes": active_tensor_bytes,
            "current_lengths": lengths,
        }


def build_stage_kv_cache(*, prefill_seq_len: int, max_new_tokens: int) -> StageKVCache:
    return StageKVCache(max_seq_len=int(prefill_seq_len) + int(max_new_tokens))


__all__ = [
    "LayerKVCache",
    "StageKVCache",
    "build_stage_kv_cache",
]
