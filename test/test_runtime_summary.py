from __future__ import annotations

import time
import unittest
from unittest import mock

import torch

from qwen3vl_tp_runtime.hexgen_core.distributed import record_transport_profile_event, startup_timer
from qwen3vl_tp_runtime.hexgen_core.schema import PayloadSummary, StageSpec, TextHybridManifest, TextPipelineManifest
from qwen3vl_tp_runtime.debug.tp_debug import TpDebugConfig
from qwen3vl_tp_runtime.scripts.runtime_summary import (
    _attach_runtime_metrics,
    _summarize_hybrid_run,
    _summarize_pipeline_generate_run,
    reset_runtime_metrics,
)


def _build_generate_phase_stats(predicted_token_id: int) -> dict:
    return {
        "input_shape": [1, 1, 4],
        "output_shape": [1, 1, 4],
        "received_payload_keys": [],
        "sent_payload_keys": [],
        "sent_tensor_shapes": {},
        "boundary_max_diff": None,
        "boundary_mean_diff": None,
        "embedding_max_diff": None,
        "embedding_mean_diff": None,
        "hidden_stage_max_diff": None,
        "hidden_stage_mean_diff": None,
        "norm_max_diff": None,
        "norm_mean_diff": None,
        "stage_max_diff": None,
        "stage_mean_diff": None,
        "predicted_token_id": predicted_token_id,
        "reference_token_id": None,
    }


class RuntimeSummaryTest(unittest.TestCase):
    def test_pipeline_generate_summary_skips_missing_reference_tensors(self) -> None:
        manifest = TextPipelineManifest(
            pipeline_type="text_generate",
            num_stages=1,
            stage_ranges=[(0, 0)],
            bundle_dir=None,
            stages=[
                StageSpec(
                    stage_idx=0,
                    start_idx=0,
                    end_idx=0,
                    num_layers=1,
                    save_dtype="float32",
                    bundle_path="/tmp/runtime-only-pipeline-stage.pt",
                )
            ],
            boundaries=[],
            num_frames=0,
            save_dtype="float32",
            runtime_config={"model_path": "/tmp/fake-model"},
        )
        stats = {
            "rank": 0,
            "stage_idx": 0,
            "num_stages": 1,
            "start_idx": 0,
            "end_idx": 0,
            "num_layers": 1,
            "weight_load": {
                "tp_weight_sharded": False,
                "tp_shard_rank": None,
                "tp_shard_world_size": None,
                "stage_start_idx": 0,
                "stage_end_idx": 0,
                "loaded_layer_indices": [0],
                "loaded_layer_count": 1,
                "loaded_top_level_weight_names": ["embed_tokens_weight", "final_norm_weight", "lm_head_weight"],
                "unexpected_layer_indices": [],
                "stage_weight_scope_ok": True,
                "loaded_weight_tensor_count": 3,
                "loaded_weight_tensor_bytes": 128,
            },
            "device": "cpu",
            "comm_dtype": "torch.float32",
            "prefill_seq_len": 4,
            "max_new_tokens": 3,
            "prefill": _build_generate_phase_stats(3),
            "steps": [_build_generate_phase_stats(2), _build_generate_phase_stats(1)],
            "generated_token_ids": [3, 2, 1],
            "prefill_output_tensor": torch.tensor([[[0.1, 0.2, 0.9, 0.3]]], dtype=torch.float32),
            "step_output_tensors": [
                torch.tensor([[[0.3, 0.8, 0.2, 0.1]]], dtype=torch.float32),
                torch.tensor([[[0.6, 0.1, 0.2, 0.4]]], dtype=torch.float32),
            ],
        }

        summary = _summarize_pipeline_generate_run(stats, manifest, topk=2)

        self.assertIn("prefill_topk", summary)
        self.assertIn("step_topks", summary)
        self.assertNotIn("reference_prefill_topk", summary)
        self.assertNotIn("reference_generated_token_ids", summary)
        self.assertNotIn("token_match", summary)
        self.assertEqual(len(summary["step_topks"]), 2)
        self.assertNotIn("reference_topk", summary["step_topks"][0])
        self.assertTrue(summary["weight_load"]["stage_weight_scope_ok"])
        self.assertEqual(summary["weight_load"]["loaded_layer_indices"], [0])

    def test_hybrid_generate_summary_skips_missing_reference_tensors(self) -> None:
        manifest = TextHybridManifest(
            runtime="text_generate_hybrid",
            tp_degrees=[1],
            stage_rank_groups=[[2]],
            pp_rank_groups=[[2]],
            world_size=1,
            num_stages=1,
            send_list=[[]],
            recv_list=[[]],
            send_empty_list=[[]],
            recv_empty_list=[[]],
            stage_ranges=[(0, 0)],
            bundle_dir=None,
            stages=[
                StageSpec(
                    stage_idx=0,
                    start_idx=0,
                    end_idx=0,
                    num_layers=1,
                    save_dtype="float32",
                    bundle_path="/tmp/runtime-only-hybrid-stage.pt",
                )
            ],
            boundaries=[],
            num_frames=0,
            save_dtype="float32",
            pipeline_type="text_generate",
            runtime_config={"model_path": "/tmp/fake-model"},
        )
        stats = {
            "rank": 2,
            "stage_idx": 0,
            "stage_ranks": [2],
            "local_rank": 0,
            "tp_degree": 1,
            "leader_rank": 2,
            "current_pp_group": [2],
            "num_stages": 1,
            "weight_load": {
                "tp_weight_sharded": False,
                "tp_shard_rank": None,
                "tp_shard_world_size": None,
                "loaded_weight_tensor_count": 3,
                "loaded_weight_tensor_bytes": 128,
            },
            "prefill_seq_len": 4,
            "max_new_tokens": 3,
            "prefill": _build_generate_phase_stats(3),
            "steps": [_build_generate_phase_stats(2), _build_generate_phase_stats(1)],
            "generated_token_ids": [3, 2, 1],
            "prefill_output_tensor": torch.tensor([[[0.1, 0.2, 0.9, 0.3]]], dtype=torch.float32),
            "step_output_tensors": [
                torch.tensor([[[0.3, 0.8, 0.2, 0.1]]], dtype=torch.float32),
                torch.tensor([[[0.6, 0.1, 0.2, 0.4]]], dtype=torch.float32),
            ],
        }

        summary = _summarize_hybrid_run(
            stats,
            manifest,
            backend="hybrid",
            topk=2,
            debug_config=TpDebugConfig(),
        )

        self.assertIn("prefill_topk", summary)
        self.assertIn("step_topks", summary)
        self.assertNotIn("reference_prefill_topk", summary)
        self.assertNotIn("reference_generated_token_ids", summary)
        self.assertNotIn("token_match", summary)
        self.assertEqual(len(summary["step_topks"]), 2)
        self.assertNotIn("reference_topk", summary["step_topks"][0])
        self.assertEqual(summary["weight_load"]["loaded_weight_tensor_bytes"], 128)

    def test_attach_runtime_metrics_records_startup_events(self) -> None:
        with mock.patch("qwen3vl_tp_runtime.scripts.runtime_summary.torch.cuda.is_available", return_value=False):
            reset_runtime_metrics()
            started_at = time.perf_counter()
            with startup_timer("direct-builder:text_generate", "prepare runtime-only text session stages=[0:0-0]"):
                pass
            record_transport_profile_event(
                channel="tensor-send",
                operation="send",
                label="multimodal_startup_contract_tensors stage_idx=1",
                peer=1,
                summary=PayloadSummary(
                    is_empty=False,
                    num_tensors=1,
                    payload_keys=["hidden_states"],
                    tensor_shapes={"hidden_states": (1, 2, 4)},
                    tensor_dtypes={"hidden_states": "torch.float32"},
                    tensor_numels={"hidden_states": 8},
                    tensor_bytes={"hidden_states": 32},
                    total_tensor_bytes=32,
                ),
                elapsed_seconds=0.25,
            )
            record_transport_profile_event(
                channel="all_reduce_cpu",
                operation="all_reduce",
                label="tp_all_reduce",
                summary=None,
                elapsed_seconds=0.5,
                extra={
                    "is_empty": False,
                    "num_tensors": 1,
                    "payload_keys": ["tensor"],
                    "tensor_shapes": {"tensor": [1, 2, 4]},
                    "tensor_dtypes": {"tensor": "torch.float32"},
                    "tensor_numels": {"tensor": 8},
                    "tensor_bytes": {"tensor": 32},
                    "total_tensor_bytes": 32,
                    "phase": "prefill",
                    "layer_idx": 3,
                    "module": "attention",
                    "reason": "row_parallel_reduce",
                },
            )
            summary = _attach_runtime_metrics({"backend": "tp"}, started_at=started_at)

        metrics = summary["runtime_metrics"]
        self.assertIn("runtime_total_seconds", metrics["timing"])
        self.assertEqual(metrics["startup"]["event_count"], 1)
        self.assertIn("prepare_session_seconds", metrics["startup"]["totals_by_kind"])
        self.assertEqual(metrics["transport"]["event_count"], 2)
        self.assertEqual(metrics["transport"]["totals_by_kind"]["startup_contract"]["tensor_bytes"], 32)
        self.assertEqual(metrics["transport"]["totals_by_kind"]["tp_collective"]["tensor_bytes"], 32)
        self.assertEqual(metrics["transport"]["pin_memory"]["requested_event_count"], 0)
        self.assertEqual(metrics["transport"]["pin_memory"]["used_event_count"], 0)
        tp_event = metrics["transport"]["events"][1]
        self.assertEqual(tp_event["phase"], "prefill")
        self.assertEqual(tp_event["layer_idx"], 3)
        self.assertEqual(tp_event["module"], "attention")
        self.assertEqual(tp_event["reason"], "row_parallel_reduce")
        self.assertIn("cpu_max_rss_bytes", metrics["memory"])


if __name__ == "__main__":
    unittest.main()
