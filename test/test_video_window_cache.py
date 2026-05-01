from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    VIDEO_WINDOW_CACHE_SCHEMA,
    build_video_window_cache_index,
)


class VideoWindowCacheTest(unittest.TestCase):
    def test_builds_windows_from_qwen3vl_video_token_groups(self) -> None:
        token_types = torch.zeros((1, 627), dtype=torch.long)
        for start, end in ((10, 154), (162, 306), (314, 458), (466, 610)):
            token_types[:, start:end] = 2

        index = build_video_window_cache_index(
            mm_token_type_ids=token_types,
            video_grid_thw=torch.tensor([[4, 24, 24]], dtype=torch.long),
            num_frames=8,
            owner_rank=1,
            stage_idx=0,
            layer_start=0,
            layer_end=17,
            tp_rank=1,
            tp_degree=2,
            cache_max_seq_len=631,
            sample_fps=1,
        )
        self.assertIsNotNone(index)
        assert index is not None

        summary = index.to_dict()
        self.assertEqual(summary["schema"], VIDEO_WINDOW_CACHE_SCHEMA)
        self.assertEqual(summary["window_count"], 4)
        self.assertEqual(summary["total_video_tokens"], 576)
        self.assertGreater(summary["metadata_bytes"], 0)
        self.assertFalse(summary["compression_enabled"])
        self.assertFalse(summary["eviction_enabled"])
        self.assertFalse(summary["remote_fetch_enabled"])

        first_window = summary["windows"][0]
        self.assertEqual(first_window["token_start"], 10)
        self.assertEqual(first_window["token_end"], 154)
        self.assertEqual(first_window["token_count"], 144)
        self.assertEqual(first_window["frame_start"], 0)
        self.assertEqual(first_window["frame_end"], 2)
        self.assertEqual(first_window["grid_thw"], [1, 24, 24])
        self.assertEqual(first_window["kv_location"]["owner_rank"], 1)
        self.assertEqual(first_window["kv_location"]["layer_start"], 0)
        self.assertEqual(first_window["kv_location"]["layer_end"], 17)
        self.assertEqual(first_window["kv_location"]["kv_offset_start"], 10)
        self.assertEqual(first_window["kv_location"]["kv_offset_end"], 154)

        last_window = summary["windows"][-1]
        self.assertEqual(last_window["frame_start"], 6)
        self.assertEqual(last_window["frame_end"], 8)
        self.assertEqual(last_window["time_start_s"], 6.0)
        self.assertEqual(last_window["time_end_s"], 8.0)

    def test_returns_none_without_video_tokens(self) -> None:
        index = build_video_window_cache_index(
            mm_token_type_ids=torch.zeros((1, 8), dtype=torch.long),
            video_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
            num_frames=1,
            owner_rank=0,
            stage_idx=0,
            layer_start=0,
            layer_end=0,
            tp_rank=0,
            tp_degree=1,
            cache_max_seq_len=8,
        )
        self.assertIsNone(index)


if __name__ == "__main__":
    unittest.main()
