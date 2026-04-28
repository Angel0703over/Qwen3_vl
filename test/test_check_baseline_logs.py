from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen3vl_tp_runtime.scripts.check_baseline_logs import (
    BaselineCheckError,
    check_baseline_logs,
    extract_last_json_summary,
)


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


if __name__ == "__main__":
    unittest.main()

