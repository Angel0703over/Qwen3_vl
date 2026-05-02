from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    VIDEO_KV_COMPACTION_SCHEMA,
    VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA,
    VIDEO_KV_COMPRESSION_PLAN_SCHEMA,
    StageKVCache,
    build_compact_decode_attention_mask_2d,
    compact_stage_kv_cache_for_video_plan,
    build_video_kv_compression_contract,
    build_video_kv_compression_plan,
    build_video_window_cache_index,
    materialize_video_kv_compression_plan,
    validate_video_kv_compression_decode_contract,
)
from qwen3vl_tp_runtime.models.qwen3vl.live.common import _build_multimodal_decode_position_ids


class VideoKVCompressionPlanTest(unittest.TestCase):
    def test_builds_noop_plan_from_video_window_cache(self) -> None:
        token_types = torch.zeros((1, 627), dtype=torch.long)
        for start, end in ((10, 154), (162, 306), (314, 458), (466, 610)):
            token_types[:, start:end] = 2

        index = build_video_window_cache_index(
            mm_token_type_ids=token_types,
            video_grid_thw=torch.tensor([[4, 24, 24]], dtype=torch.long),
            num_frames=8,
            owner_rank=0,
            stage_idx=0,
            layer_start=0,
            layer_end=35,
            tp_rank=0,
            tp_degree=2,
            cache_max_seq_len=631,
            sample_fps=1,
        )
        self.assertIsNotNone(index)
        assert index is not None

        plan = build_video_kv_compression_plan(
            video_window_cache=index.to_dict(),
            stage_kv_cache_summary={
                "max_seq_len": 631,
                "allocated_layers": 35,
                "append_count": 35,
                "tensor_bytes": 46_522_368,
                "current_lengths": {layer_idx: 627 for layer_idx in range(35)},
            },
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["schema"], VIDEO_KV_COMPRESSION_PLAN_SCHEMA)
        self.assertTrue(plan["planner_only"])
        self.assertFalse(plan["mutates_kv"])
        self.assertFalse(plan["compression_enabled"])
        self.assertFalse(plan["selector_enabled"])
        self.assertEqual(plan["method"], "none")
        self.assertEqual(plan["budget_source"], "none")
        self.assertEqual(plan["window_count"], 4)
        self.assertEqual(plan["total_original_tokens"], 576)
        self.assertEqual(plan["total_keep_tokens"], 576)
        self.assertEqual(plan["total_drop_tokens"], 0)
        self.assertEqual(plan["expected_keep_ratio"], 1.0)
        self.assertEqual(plan["estimated_local_kv_bytes_per_token"], 73_728)
        self.assertEqual(plan["estimated_original_kv_bytes"], 42_467_328)
        self.assertEqual(plan["estimated_keep_kv_bytes"], 42_467_328)
        self.assertGreater(plan["metadata_bytes"], 0)

        first_window = plan["windows"][0]
        self.assertEqual(first_window["token_range"], [10, 154])
        self.assertEqual(first_window["original_token_count"], 144)
        self.assertEqual(first_window["keep_token_count"], 144)
        self.assertEqual(first_window["drop_token_count"], 0)
        self.assertEqual(first_window["selector"], "none")
        self.assertEqual(first_window["selected_token_count"], 144)
        self.assertEqual(first_window["selected_token_sample"][:3], [10, 11, 12])
        self.assertEqual(first_window["selected_token_ranges"], [[10, 154]])
        self.assertEqual(first_window["candidate_token_count"], 144)
        self.assertEqual(first_window["candidate_token_sample"][:3], [10, 11, 12])
        self.assertEqual(first_window["candidate_token_ranges"], [[10, 154]])

    def test_uniform_plan_only_budget_does_not_mutate(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 10,
                    "token_end": 18,
                    "token_count": 8,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }

        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 10, "tensor_bytes": 160},
            method="uniform",
            keep_ratio=0.5,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan["mutates_kv"])
        self.assertFalse(plan["compression_enabled"])
        self.assertTrue(plan["selector_enabled"])
        self.assertEqual(plan["method"], "uniform")
        self.assertEqual(plan["budget_source"], "keep_ratio")
        self.assertEqual(plan["total_keep_tokens"], 4)
        self.assertEqual(plan["total_drop_tokens"], 4)
        first_window = plan["windows"][0]
        self.assertEqual(first_window["selector"], "uniform")
        self.assertEqual(first_window["selector_status"], "planned_uniform")
        self.assertEqual(first_window["selected_token_count"], 4)
        self.assertEqual(first_window["selected_token_sample"], [10, 12, 14, 16])
        self.assertEqual(first_window["selected_token_ranges"], [[10, 11], [12, 13], [14, 15], [16, 17]])
        self.assertEqual(first_window["candidate_token_sample"], first_window["selected_token_sample"])
        self.assertEqual(first_window["candidate_token_ranges"], first_window["selected_token_ranges"])

    def test_swa_selector_keeps_recent_window(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 10,
                    "token_end": 18,
                    "token_count": 8,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }

        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 10, "tensor_bytes": 160},
            method="swa",
            keep_tokens_per_window=3,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan["mutates_kv"])
        self.assertFalse(plan["compression_enabled"])
        self.assertTrue(plan["selector_enabled"])
        self.assertEqual(plan["method"], "swa")
        self.assertEqual(plan["budget_source"], "keep_tokens_per_window")
        self.assertEqual(plan["total_keep_tokens"], 3)
        self.assertEqual(plan["total_drop_tokens"], 5)
        first_window = plan["windows"][0]
        self.assertEqual(first_window["selector"], "swa")
        self.assertEqual(first_window["selector_status"], "planned_recent_window")
        self.assertEqual(first_window["selected_token_count"], 3)
        self.assertEqual(first_window["selected_token_sample"], [15, 16, 17])
        self.assertEqual(first_window["selected_token_ranges"], [[15, 18]])

    def test_infinipot_v_plan_requires_local_kv_scores_then_materializes(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 2,
                    "token_end": 8,
                    "token_count": 6,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }

        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 10, "tensor_bytes": 160},
            method="infinipot-v",
            keep_ratio=0.5,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan["selector_enabled"])
        self.assertFalse(plan.get("selector_materialized", False))
        first_window = plan["windows"][0]
        self.assertEqual(first_window["selector"], "infinipot-v")
        self.assertEqual(first_window["selector_status"], "requires_layer_kv_scores")
        self.assertEqual(first_window["selected_token_count"], 0)

        cache = StageKVCache(max_seq_len=10)
        key = torch.zeros((1, 1, 10, 2), dtype=torch.float32)
        value = torch.zeros((1, 1, 10, 2), dtype=torch.float32)
        value[..., 3, :] = torch.tensor([9.0, 0.0])
        value[..., 5, :] = torch.tensor([8.0, 0.0])
        value[..., 7, :] = torch.tensor([1.0, 0.0])
        cache.append(0, key, value)

        materialized = materialize_video_kv_compression_plan(
            compression_plan=plan,
            stage_kv_cache=cache,
            prefill_seq_len=10,
        )

        self.assertTrue(materialized["selector_materialized"])
        self.assertEqual(materialized["method"], "infinipot-v")
        first_window = materialized["windows"][0]
        self.assertEqual(first_window["selector_status"], "materialized_local_kv_scores")
        self.assertEqual(first_window["selector_score_source"], "local_stage_kv_cache")
        self.assertEqual(first_window["score_layer_count"], 1)
        self.assertEqual(first_window["tar_ratio"], 0.5)
        self.assertEqual(first_window["query_ratio"], 0.25)
        self.assertEqual(first_window["selected_token_sample"], [3, 5, 7])
        self.assertEqual(first_window["selected_token_ranges"], [[3, 4], [5, 6], [7, 8]])

        contract = build_video_kv_compression_contract(
            compression_plan=materialized,
            prefill_seq_len=10,
        )
        assert contract is not None
        self.assertEqual(contract["prefill"]["compact_length"], 7)

        compaction = compact_stage_kv_cache_for_video_plan(
            stage_kv_cache=cache,
            compression_plan=materialized,
            prefill_seq_len=10,
        )
        self.assertIsNotNone(compaction)
        assert compaction is not None
        self.assertEqual(compaction["method"], "infinipot-v")
        self.assertEqual(compaction["compact_prefill_length"], 7)
        key_view, value_view = cache.as_cache_by_layer()[0]
        expected_keep = torch.tensor([0, 1, 3, 5, 7, 8, 9], dtype=torch.long)
        self.assertTrue(torch.equal(key_view, key.index_select(-2, expected_keep)))
        self.assertTrue(torch.equal(value_view, value.index_select(-2, expected_keep)))

    def test_build_plan_can_materialize_infinipot_v_with_stage_kv_cache(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 2,
                    "token_end": 8,
                    "token_count": 6,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }
        cache = StageKVCache(max_seq_len=10)
        key = torch.zeros((1, 1, 10, 2), dtype=torch.float32)
        value = torch.zeros((1, 1, 10, 2), dtype=torch.float32)
        value[..., 3, :] = torch.tensor([9.0, 0.0])
        value[..., 5, :] = torch.tensor([8.0, 0.0])
        value[..., 7, :] = torch.tensor([1.0, 0.0])
        cache.append(0, key, value)

        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary=cache.summary(),
            stage_kv_cache=cache,
            method="infinipot-v",
            keep_ratio=0.5,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan["selector_materialized"])
        self.assertEqual(plan["total_keep_tokens"], 3)
        self.assertEqual(plan["windows"][0]["selected_token_sample"], [3, 5, 7])

    def test_compression_contract_separates_physical_and_logical_lengths(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 10,
                    "token_end": 18,
                    "token_count": 8,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }
        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 24, "tensor_bytes": 384},
            method="uniform",
            keep_ratio=0.5,
        )
        assert plan is not None

        contract = build_video_kv_compression_contract(
            compression_plan=plan,
            prefill_seq_len=20,
            decoded_token_count=2,
            query_len=1,
        )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["schema"], VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA)
        self.assertTrue(contract["contract_only"])
        self.assertFalse(contract["mutates_kv"])
        self.assertEqual(contract["prefill"]["original_length"], 20)
        self.assertEqual(contract["prefill"]["compact_length"], 16)
        self.assertEqual(contract["prefill"]["dropped_token_count"], 4)
        self.assertEqual(contract["decode"]["physical_past_length"], 18)
        self.assertEqual(contract["decode"]["logical_past_length"], 22)
        self.assertEqual(contract["decode"]["attention_mask_key_length"], 19)
        self.assertEqual(contract["decode"]["logical_key_length"], 23)
        self.assertEqual(contract["decode"]["decode_position_start"], 22)
        self.assertTrue(contract["decode"]["requires_position_override"])
        self.assertTrue(contract["rules"]["compact_mask_must_not_drive_position_ids"])

        prefill_mask = torch.arange(20, dtype=torch.long).view(1, 20)
        compact_decode_mask = build_compact_decode_attention_mask_2d(
            prefill_mask,
            compression_plan=plan,
            decoded_token_count=2,
            query_len=1,
        )
        self.assertEqual(tuple(compact_decode_mask.shape), (1, 19))
        self.assertEqual(
            compact_decode_mask[0, :16].tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19],
        )
        self.assertEqual(compact_decode_mask[0, 16:].tolist(), [1, 1, 1])

        validate_video_kv_compression_decode_contract(
            attention_mask_2d=compact_decode_mask,
            key_length=19,
            query_len=1,
            compression_contract=contract,
        )
        with self.assertRaisesRegex(ValueError, "key length does not match"):
            validate_video_kv_compression_decode_contract(
                attention_mask_2d=compact_decode_mask,
                key_length=23,
                query_len=1,
                compression_contract=contract,
            )

    def test_noop_contract_matches_existing_full_length(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 2,
                    "token_end": 6,
                    "token_count": 4,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }
        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 10, "tensor_bytes": 160},
            method="none",
        )
        assert plan is not None

        contract = build_video_kv_compression_contract(
            compression_plan=plan,
            prefill_seq_len=10,
            decoded_token_count=0,
            query_len=1,
        )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["prefill"]["compact_length"], 10)
        self.assertEqual(contract["decode"]["physical_past_length"], 10)
        self.assertEqual(contract["decode"]["logical_past_length"], 10)
        self.assertEqual(contract["decode"]["attention_mask_key_length"], 11)
        self.assertFalse(contract["decode"]["requires_position_override"])

        full_mask = build_compact_decode_attention_mask_2d(
            torch.ones((1, 10), dtype=torch.long),
            compression_plan=plan,
            decoded_token_count=0,
            query_len=1,
        )
        self.assertEqual(tuple(full_mask.shape), (1, 11))
        validate_video_kv_compression_decode_contract(
            attention_mask_2d=full_mask,
            key_length=11,
            query_len=1,
            compression_contract=contract,
        )

    def test_contract_rejects_plan_without_materialized_selection(self) -> None:
        plan = {
            "method": "infinipot-v",
            "total_original_tokens": 8,
            "total_keep_tokens": 4,
            "total_drop_tokens": 4,
            "windows": [
                {
                    "token_range": [10, 18],
                    "keep_token_count": 4,
                    "selected_token_count": 0,
                    "selected_token_ranges": [],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "no materialized selected ranges"):
            build_video_kv_compression_contract(
                compression_plan=plan,
                prefill_seq_len=20,
            )

    def test_compacts_stage_kv_cache_for_uniform_plan(self) -> None:
        video_window_cache = {
            "windows": [
                {
                    "window_id": {"batch_index": 0, "window_index": 0},
                    "token_start": 2,
                    "token_end": 6,
                    "token_count": 4,
                    "kv_location": {"layer_start": 0, "layer_end": 1},
                }
            ]
        }
        plan = build_video_kv_compression_plan(
            video_window_cache=video_window_cache,
            stage_kv_cache_summary={"max_seq_len": 10, "tensor_bytes": 160},
            method="uniform",
            keep_ratio=0.5,
        )
        assert plan is not None
        cache = StageKVCache(max_seq_len=10)
        key = torch.arange(20, dtype=torch.float32).view(1, 1, 10, 2)
        value = key + 100
        cache.append(0, key, value)

        compaction = compact_stage_kv_cache_for_video_plan(
            stage_kv_cache=cache,
            compression_plan=plan,
            prefill_seq_len=10,
        )

        self.assertIsNotNone(compaction)
        assert compaction is not None
        self.assertEqual(compaction["schema"], VIDEO_KV_COMPACTION_SCHEMA)
        self.assertTrue(compaction["applied"])
        self.assertTrue(compaction["mutates_kv"])
        self.assertEqual(compaction["original_prefill_length"], 10)
        self.assertEqual(compaction["compact_prefill_length"], 8)
        self.assertEqual(compaction["dropped_prefill_tokens"], 2)
        self.assertEqual(compaction["video_keep_token_count"], 2)
        self.assertEqual(compaction["keep_token_range_sample"], [[0, 3], [4, 5], [6, 10]])
        self.assertEqual(compaction["active_tensor_bytes_saved"], 32)
        self.assertEqual(cache.summary()["current_lengths"], {0: 8})
        key_view, value_view = cache.as_cache_by_layer()[0]
        expected_keep = torch.tensor([0, 1, 2, 4, 6, 7, 8, 9], dtype=torch.long)
        self.assertTrue(torch.equal(key_view, key.index_select(-2, expected_keep)))
        self.assertTrue(torch.equal(value_view, value.index_select(-2, expected_keep)))

    def test_logical_position_override_ignores_compact_mask_cumsum(self) -> None:
        decode_input_ids = torch.tensor([[42]], dtype=torch.long)
        compact_attention_mask = torch.ones((1, 8), dtype=torch.long)
        position_ids, text_position_ids, vision_position_ids = _build_multimodal_decode_position_ids(
            decode_input_ids=decode_input_ids,
            attention_mask_2d=compact_attention_mask,
            rope_deltas=torch.tensor([5], dtype=torch.long),
            logical_position_start=20,
        )

        self.assertEqual(text_position_ids.tolist(), [[20]])
        self.assertEqual(vision_position_ids[:, 0, 0].tolist(), [25, 25, 25])
        self.assertEqual(position_ids[:, 0, 0].tolist(), [20, 25, 25, 25])


if __name__ == "__main__":
    unittest.main()
