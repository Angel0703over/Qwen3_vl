from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    VIDEO_KV_COMPRESSION_PLAN_SCHEMA,
    build_video_kv_compression_plan,
    build_video_window_cache_index,
)


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


if __name__ == "__main__":
    unittest.main()
