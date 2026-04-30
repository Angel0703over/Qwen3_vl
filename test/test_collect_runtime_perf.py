from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen3vl_tp_runtime.scripts.collect_runtime_perf import collect_records, records_to_markdown


class CollectRuntimePerfTest(unittest.TestCase):
    def test_collects_legacy_startup_and_time_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_dir = Path(tmp)
            log_path = baseline_dir / "tp-text-generate-rank0.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[startup][direct-builder:text_generate] host=jetson rank=0 local_rank=0 "
                        "done prepare runtime-only text session stages=[0:0-35] in 0.58s",
                        "[startup][direct-builder:text_generate] host=jetson rank=0 local_rank=0 "
                        "done materialize stage_idx=0 range=0:35 in 0.03s",
                        json.dumps(
                            {
                                "rank": 0,
                                "backend": "tp",
                                "pipeline_type": "text_generate",
                                "generated_token_ids": [1, 2],
                                "generated_text": "ok",
                                "weight_load": {
                                    "loaded_weight_tensor_bytes": 2048,
                                    "tp_weight_sharded": True,
                                },
                            }
                        ),
                        "real 1.23",
                    ]
                ),
                encoding="utf-8",
            )

            records = collect_records(baseline_dir)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["case_id"], "tp-text-generate")
        self.assertEqual(record["timing"]["runtime_total_seconds"], 1.23)
        self.assertEqual(record["timing"]["prepare_session_seconds"], 0.58)
        self.assertEqual(record["timing"]["materialize_stage_seconds"], 0.03)
        self.assertEqual(record["weight_load"]["loaded_weight_tensor_bytes"], 2048)
        self.assertIn("tp-text-generate", records_to_markdown(records))

    def test_prefers_runtime_metrics_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline_dir = Path(tmp)
            log_path = baseline_dir / "pp-mm-generate-rank1.log"
            log_path.write_text(
                json.dumps(
                    {
                        "rank": 1,
                        "backend": "pp",
                        "pipeline_type": "multimodal_generate",
                        "generated_token_ids": [3],
                        "generated_text": "视频",
                        "runtime_metrics": {
                            "timing": {"runtime_total_seconds": 9.5},
                            "startup": {
                                "events": [],
                                "totals_by_kind": {
                                    "prepare_session_seconds": 0.2,
                                    "startup_contract_seconds": 1.0,
                                    "startup_contract_transport_seconds": 0.4,
                                    "materialize_stage_seconds": 0.3,
                                    "post_load_barrier_seconds": 0.1,
                                    "scaffold_transport_seconds": 0.0,
                                },
                            },
                            "transport": {
                                "event_count": 3,
                                "events": [
                                    {
                                        "kind": "tp_collective",
                                        "operation": "all_reduce",
                                        "label": "tp_all_reduce",
                                        "elapsed_seconds": 0.2,
                                        "total_tensor_bytes": 4096,
                                        "payload_prepare_seconds": 0.01,
                                        "device_to_cpu_seconds": 0.01,
                                        "gloo_collective_seconds": 0.15,
                                        "cpu_to_device_seconds": 0.02,
                                        "phase": "prefill",
                                        "module": "attention",
                                        "reason": "row_parallel_reduce",
                                    },
                                    {
                                        "kind": "tp_collective",
                                        "operation": "all_reduce",
                                        "label": "tp_all_reduce",
                                        "elapsed_seconds": 0.1,
                                        "total_tensor_bytes": 4096,
                                        "payload_prepare_seconds": 0.005,
                                        "device_to_cpu_seconds": 0.005,
                                        "gloo_collective_seconds": 0.08,
                                        "cpu_to_device_seconds": 0.01,
                                        "phase": "decode",
                                        "module": "mlp",
                                        "reason": "row_parallel_reduce",
                                    },
                                ],
                                "totals_by_kind": {
                                    "startup_contract": {
                                        "event_count": 1,
                                        "elapsed_seconds": 0.4,
                                        "object_bytes": 128,
                                        "tensor_bytes": 2048,
                                        "total_bytes": 2176,
                                    },
                                    "stage_handoff": {
                                        "event_count": 1,
                                        "elapsed_seconds": 0.2,
                                        "object_bytes": 0,
                                        "tensor_bytes": 4096,
                                        "total_bytes": 4096,
                                    },
                                    "tp_collective": {
                                        "event_count": 1,
                                        "elapsed_seconds": 0.3,
                                        "object_bytes": 0,
                                        "tensor_bytes": 8192,
                                        "total_bytes": 8192,
                                    },
                                },
                            },
                            "memory": {
                                "cpu_max_rss_bytes": 4096,
                                "cuda_available": True,
                                "peak_allocated_bytes": 1024,
                                "peak_reserved_bytes": 2048,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            records = collect_records(baseline_dir)

        self.assertEqual(records[0]["timing"]["runtime_total_seconds"], 9.5)
        self.assertEqual(records[0]["timing"]["startup_contract_transport_seconds"], 0.4)
        self.assertEqual(records[0]["payload"]["startup_contract_bytes"], 2176)
        self.assertEqual(records[0]["payload"]["stage_handoff_bytes"], 4096)
        self.assertEqual(records[0]["payload"]["tp_collective_seconds"], 0.3)
        self.assertEqual(
            records[0]["payload"]["tp_collective_breakdown"],
            [
                {
                    "phase": "decode",
                    "module": "mlp",
                    "reason": "row_parallel_reduce",
                    "operation": "all_reduce",
                    "event_count": 1,
                    "elapsed_seconds": 0.1,
                    "tensor_bytes": 4096,
                    "payload_prepare_seconds": 0.005,
                    "device_to_cpu_seconds": 0.005,
                    "gloo_collective_seconds": 0.08,
                    "cpu_to_device_seconds": 0.01,
                },
                {
                    "phase": "prefill",
                    "module": "attention",
                    "reason": "row_parallel_reduce",
                    "operation": "all_reduce",
                    "event_count": 1,
                    "elapsed_seconds": 0.2,
                    "tensor_bytes": 4096,
                    "payload_prepare_seconds": 0.01,
                    "device_to_cpu_seconds": 0.01,
                    "gloo_collective_seconds": 0.15,
                    "cpu_to_device_seconds": 0.02,
                },
            ],
        )
        self.assertEqual(
            records[0]["payload"]["tp_collective_substage_seconds"],
            {
                "event_count": 2,
                "elapsed_seconds": 0.3,
                "payload_prepare_seconds": 0.015,
                "device_to_cpu_seconds": 0.015,
                "gloo_collective_seconds": 0.23,
                "cpu_to_device_seconds": 0.03,
            },
        )
        self.assertEqual(records[0]["memory"]["cuda_peak_allocated_bytes"], 1024)


if __name__ == "__main__":
    unittest.main()
