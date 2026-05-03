from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

from qwen3vl_tp_runtime.scripts.check_baseline_logs import (
    BaselineCheckError,
    check_baseline_logs,
    check_smoke_matrix,
    extract_last_json_summary,
)
from qwen3vl_tp_runtime.scripts.smoke_matrix import FRAME_MM_GENERATE_IDS, FRAME_MM_GENERATE_TEXT


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class CheckBaselineLogsTest(unittest.TestCase):
    def test_extract_last_json_summary_ignores_startup_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(
                Path(tmpdir) / "rank0.log",
                'noise {"not_summary": true}\n{"rank": 0, "generated_token_ids": [1, 2], "generated_text": "好"}\n',
            )

            summary = extract_last_json_summary(path)

        self.assertEqual(summary["generated_token_ids"], [1, 2])
        self.assertEqual(summary["generated_text"], "好")

    def test_tp_checks_rank_local_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for rank in range(2):
                paths.append(
                    _write(
                        Path(tmpdir) / f"rank{rank}.log",
                        (
                            "{"
                            f'"rank": {rank}, "backend": "tp", '
                            '"generated_token_ids": [1, 2], "generated_text": "好", '
                            '"weight_load": {'
                            '"tp_weight_sharded": true, '
                            f'"tp_shard_rank": {rank}, '
                            '"tp_shard_world_size": 2, '
                            '"tp_shard_shape_ok": true, '
                            '"tp_stage_loaded_weight_tensor_bytes_equal": true, '
                            '"loaded_weight_tensor_bytes": 16'
                            "}}"
                        ),
                    )
                )

            items = check_baseline_logs("tp-text-generate", paths)

        self.assertEqual(len(items), 2)

    def test_common_check_rejects_generated_token_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [
                _write(
                    Path(tmpdir) / "rank0.log",
                    '{"rank": 0, "backend": "pp", "stage_idx": 0, "generated_token_ids": [1], '
                    '"generated_text": "好", "start_idx": 0, "end_idx": 0, '
                    '"weight_load": {"stage_weight_scope_ok": true, "stage_start_idx": 0, "stage_end_idx": 0}}',
                ),
                _write(
                    Path(tmpdir) / "rank1.log",
                    '{"rank": 1, "backend": "pp", "stage_idx": 1, "generated_token_ids": [2], '
                    '"generated_text": "好", "start_idx": 1, "end_idx": 1, '
                    '"weight_load": {"stage_weight_scope_ok": true, "stage_start_idx": 1, "stage_end_idx": 1}}',
                ),
            ]

            with self.assertRaises(BaselineCheckError):
                check_baseline_logs("pp-text-generate", paths)

    def test_smoke_matrix_checks_expected_frame_output_and_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for rank in range(2):
                paths.append(
                    _write(
                        Path(tmpdir) / f"tp-mm-generate-frame-regression-rank{rank}.log",
                        json.dumps(
                            {
                                "rank": rank,
                                "backend": "tp",
                                "stage_idx": 0,
                                "generated_token_ids": FRAME_MM_GENERATE_IDS,
                                "generated_text": FRAME_MM_GENERATE_TEXT,
                                "prefill": {
                                    "video_input": {
                                        "source": "frame_paths",
                                    },
                                },
                                "runtime_metrics": {
                                    "transport": {
                                        "totals_by_kind": {
                                            "startup_contract": {
                                                "event_count": 1,
                                                "elapsed_seconds": 0.1,
                                                "object_bytes": 1,
                                                "tensor_bytes": 2,
                                                "total_bytes": 3,
                                            },
                                            "scaffold": {
                                                "event_count": 0,
                                                "elapsed_seconds": 0.0,
                                                "object_bytes": 0,
                                                "tensor_bytes": 0,
                                                "total_bytes": 0,
                                            },
                                            "stage_handoff": {
                                                "event_count": 0,
                                                "elapsed_seconds": 0.0,
                                                "object_bytes": 0,
                                                "tensor_bytes": 0,
                                                "total_bytes": 0,
                                            },
                                            "tp_collective": {
                                                "event_count": 1,
                                                "elapsed_seconds": 0.2,
                                                "object_bytes": 0,
                                                "tensor_bytes": 4,
                                                "total_bytes": 4,
                                            },
                                        },
                                        "events": [],
                                    },
                                },
                                "weight_load": {
                                    "tp_weight_sharded": True,
                                    "tp_shard_rank": rank,
                                    "tp_shard_world_size": 2,
                                    "tp_shard_shape_ok": True,
                                    "tp_stage_loaded_weight_tensor_bytes_equal": True,
                                    "stage_weight_scope_ok": True,
                                    "loaded_weight_tensor_bytes": 16,
                                },
                            }
                        )
                        + "\nmultimodal_frontend_mode=consume-only\n",
                    )
                )

            items = check_baseline_logs("tp-mm-generate-frame-regression", paths)

        self.assertEqual(len(items), 2)

    def test_transport_metrics_can_be_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for rank in range(2):
                paths.append(
                    _write(
                        Path(tmpdir) / f"tp-mm-generate-frame-regression-rank{rank}.log",
                        json.dumps(
                            {
                                "rank": rank,
                                "backend": "tp",
                                "stage_idx": 0,
                                "generated_token_ids": FRAME_MM_GENERATE_IDS,
                                "generated_text": FRAME_MM_GENERATE_TEXT,
                                "prefill": {"video_input": {"source": "frame_paths"}},
                                "weight_load": {
                                    "tp_weight_sharded": True,
                                    "tp_shard_rank": rank,
                                    "tp_shard_world_size": 2,
                                    "tp_shard_shape_ok": True,
                                    "loaded_weight_tensor_bytes": 16,
                                },
                            }
                        )
                        + "\nmultimodal_frontend_mode=consume-only\n",
                    )
                )

            with self.assertRaises(BaselineCheckError):
                check_baseline_logs("tp-mm-generate-frame-regression", paths)

    def test_hybrid_video_source_allows_consume_only_rank_without_video_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            common = {
                "backend": "hybrid",
                "generated_token_ids": FRAME_MM_GENERATE_IDS,
                "generated_text": FRAME_MM_GENERATE_TEXT,
            }
            rank_summaries = [
                {
                    **common,
                    "rank": 0,
                    "stage_idx": 0,
                    "local_rank": 0,
                    "tp_degree": 2,
                    "prefill": {"video_input": {"source": "frame_paths"}},
                    "weight_load": {
                        "stage_weight_scope_ok": True,
                        "tp_weight_sharded": True,
                        "tp_shard_rank": 0,
                        "tp_shard_world_size": 2,
                        "tp_shard_shape_ok": True,
                        "loaded_top_level_weight_names": ["embed_tokens_weight"],
                    },
                },
                {
                    **common,
                    "rank": 1,
                    "stage_idx": 0,
                    "local_rank": 1,
                    "tp_degree": 2,
                    "weight_load": {
                        "stage_weight_scope_ok": True,
                        "tp_weight_sharded": True,
                        "tp_shard_rank": 1,
                        "tp_shard_world_size": 2,
                        "tp_shard_shape_ok": True,
                        "loaded_top_level_weight_names": ["embed_tokens_weight"],
                    },
                },
                {
                    **common,
                    "rank": 2,
                    "stage_idx": 1,
                    "local_rank": 0,
                    "tp_degree": 1,
                    "weight_load": {
                        "stage_weight_scope_ok": True,
                        "tp_weight_sharded": False,
                        "loaded_top_level_weight_names": ["final_norm_weight", "lm_head_weight"],
                        "multimodal_frontend_mode": "consume-only",
                    },
                },
            ]
            for rank, summary in enumerate(rank_summaries):
                paths.append(_write(Path(tmpdir) / f"hybrid-mm-generate-rank{rank}.log", json.dumps(summary)))

            items = check_baseline_logs("hybrid-mm-generate", paths)

        self.assertEqual(len(items), 3)

    def test_smoke_matrix_reports_missing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(BaselineCheckError):
                check_smoke_matrix(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
