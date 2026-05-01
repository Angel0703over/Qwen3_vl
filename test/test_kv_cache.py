from __future__ import annotations

import unittest

import torch

import qwen3vl_tp_runtime.models.qwen3vl.execution.stages as stages_module
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    LayerKVCache,
    StageKVCache,
    build_stage_kv_cache,
)


class KVCacheTest(unittest.TestCase):
    def test_layer_cache_appends_into_single_buffer(self) -> None:
        cache = LayerKVCache(max_seq_len=5)
        first_key = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
        first_value = first_key + 100

        key_view, value_view = cache.append(first_key, first_value)
        self.assertEqual(tuple(key_view.shape), (1, 2, 3, 2))
        self.assertEqual(tuple(value_view.shape), (1, 2, 3, 2))
        self.assertEqual(cache.current_length, 3)
        self.assertEqual(cache.append_count, 1)
        self.assertTrue(torch.equal(key_view, first_key))
        self.assertTrue(torch.equal(value_view, first_value))

        second_key = torch.full((1, 2, 1, 2), 7.0)
        second_value = torch.full((1, 2, 1, 2), 9.0)
        key_storage_ptr = cache.key_buffer.untyped_storage().data_ptr()
        value_storage_ptr = cache.value_buffer.untyped_storage().data_ptr()

        key_view, value_view = cache.append(second_key, second_value)
        self.assertEqual(tuple(key_view.shape), (1, 2, 4, 2))
        self.assertEqual(tuple(value_view.shape), (1, 2, 4, 2))
        self.assertEqual(cache.current_length, 4)
        self.assertEqual(cache.append_count, 2)
        self.assertEqual(cache.key_buffer.untyped_storage().data_ptr(), key_storage_ptr)
        self.assertEqual(cache.value_buffer.untyped_storage().data_ptr(), value_storage_ptr)
        self.assertTrue(torch.equal(key_view[:, :, :3, :], first_key))
        self.assertTrue(torch.equal(value_view[:, :, :3, :], first_value))
        self.assertTrue(torch.equal(key_view[:, :, 3:, :], second_key))
        self.assertTrue(torch.equal(value_view[:, :, 3:, :], second_value))

    def test_layer_cache_rejects_overflow(self) -> None:
        cache = LayerKVCache(max_seq_len=2)
        cache.append(torch.zeros((1, 1, 2, 4)), torch.zeros((1, 1, 2, 4)))

        with self.assertRaises(RuntimeError):
            cache.append(torch.zeros((1, 1, 1, 4)), torch.zeros((1, 1, 1, 4)))

    def test_stage_cache_tracks_layers_and_summary(self) -> None:
        cache = build_stage_kv_cache(prefill_seq_len=3, max_new_tokens=2)
        self.assertIsInstance(cache, StageKVCache)
        key = torch.ones((1, 1, 3, 2), dtype=torch.bfloat16)
        value = torch.ones((1, 1, 3, 2), dtype=torch.bfloat16)

        cache.append(4, key, value)
        cache.append(7, key[:, :, :1, :], value[:, :, :1, :])
        cache_map = cache.as_cache_by_layer()
        summary = cache.summary()

        self.assertEqual(set(cache_map), {4, 7})
        self.assertEqual(tuple(cache_map[4][0].shape), (1, 1, 3, 2))
        self.assertEqual(tuple(cache_map[7][0].shape), (1, 1, 1, 2))
        self.assertEqual(summary["max_seq_len"], 5)
        self.assertEqual(summary["allocated_layers"], 2)
        self.assertEqual(summary["append_count"], 2)
        self.assertEqual(summary["current_lengths"], {4: 3, 7: 1})
        self.assertEqual(summary["tensor_bytes"], 80)

    def test_stage_trace_with_stage_cache_skips_cache_clone(self) -> None:
        original_trace_decoder_layer_cached = stages_module.trace_decoder_layer_cached
        seen_layer_cache = []

        def fake_trace_decoder_layer_cached(output: torch.Tensor, layer_state: dict) -> dict:
            seen_layer_cache.append(layer_state.get("layer_kv_cache"))
            return {
                "layer_output": output + 1,
                "full_key": object(),
                "full_value": object(),
            }

        try:
            stages_module.trace_decoder_layer_cached = fake_trace_decoder_layer_cached
            stage_cache = StageKVCache(max_seq_len=4)
            result = stages_module.trace_text_decode_stage_with_runtime_cache(
                torch.zeros((1, 1, 2)),
                {
                    "layers": [{"layer_idx": 3}],
                    "attention_mask": None,
                    "cos": None,
                    "sin": None,
                },
                stage_kv_cache=stage_cache,
            )
        finally:
            stages_module.trace_decoder_layer_cached = original_trace_decoder_layer_cached

        self.assertEqual(result["cache_by_layer"], {})
        self.assertIs(result["stage_kv_cache"], stage_cache)
        self.assertEqual(len(seen_layer_cache), 1)
        self.assertIsInstance(seen_layer_cache[0], LayerKVCache)


if __name__ == "__main__":
    unittest.main()
