from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from qwen3vl_tp_runtime.hexgen_core.modules import tensor_parallel as tensor_parallel_module
from qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel import (
    TensorParallelManifest,
    TensorParallelRunner,
    load_stage_state_for_tp_rank,
)
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmFrontendSeed,
    compact_mm_runtime_shared,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    pack_mm_startup_transport,
    select_mm_startup_contract,
)


def _build_tp_manifest(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
    modality: str = "text",
    include_runtime_reference: bool = False,
) -> TensorParallelManifest:
    stages = [
        StageSpec(
            stage_idx=stage_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            num_layers=end_idx - start_idx + 1,
            save_dtype="float32",
            bundle_path=None,
        )
        for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges)
    ]
    return TensorParallelManifest(
        runtime="text_generate_tp",
        tp_degrees=tp_degrees,
        stage_ranges=stage_ranges,
        bundle_dir=None,
        stages=stages,
        boundaries=[],
        num_frames=0,
        save_dtype="float32",
        pipeline_type=f"{modality}_generate",
        runtime_config={
            "modality": modality,
            "mode": "generate",
            "model_path": "/tmp/fake-model",
            "save_dtype": "float32",
            "include_runtime_reference": include_runtime_reference,
        },
    )


def _build_mm_startup_contract(*, num_stages: int) -> dict[str, object]:
    frontend_seed = MmFrontendSeed(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
        position_ids=torch.zeros(4, 1, 3, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 3, 4),
        attention_mask=torch.zeros(1, 1, 3, 3),
        cos=torch.zeros(1, 3, 4),
        sin=torch.zeros(1, 3, 4),
        visual_pos_masks=torch.ones(1, 3, dtype=torch.bool),
        deepstack_by_layer={},
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 1, 0]], dtype=torch.int),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    return {
        "shared": compact_mm_runtime_shared(frontend_seed),
        "stage_handoffs": {
            stage_idx: {
                "stage_input": torch.zeros(1, 3, 4) + float(stage_idx),
                "stage_output": torch.zeros(1, 3, 4) + float(stage_idx + 1),
            }
            for stage_idx in range(num_stages)
        },
        "stage_visuals": {
            stage_idx: {
                "visual_pos_masks": torch.ones(1, 3, dtype=torch.bool),
                "deepstack_by_layer": {},
            }
            for stage_idx in range(num_stages)
        },
        "num_frames": 2,
        "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
    }


class TensorParallelDirectRunnerTest(unittest.TestCase):
    def test_direct_tp_runner_uses_tensor_parallel_loader(self) -> None:
        manifest = _build_tp_manifest(stage_ranges=[(0, 35)], tp_degrees=[2])
        runner = TensorParallelRunner(
            manifest=manifest,
            device=torch.device("cpu"),
            compute_dtype_arg="float32",
            comm_dtype_arg="float32",
            return_tensors=True,
        )

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.load_stage_state_for_tp_rank",
            return_value=({"start_idx": 0, "end_idx": 35, "layers": []}, torch.float32),
        ) as load_mock, patch.object(
            TensorParallelRunner,
            "_run_generate_rank",
            return_value={"rank": 0, "backend": "tp"},
        ) as run_generate_mock:
            stats = runner.run_rank(rank=0, world_size=2)

        self.assertEqual(stats["backend"], "tp")
        load_mock.assert_called_once()
        self.assertIs(load_mock.call_args.args[0], manifest)
        run_generate_mock.assert_called_once()

    def test_direct_tp_runner_rejects_multi_stage_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "单 stage"):
            _build_tp_manifest(stage_ranges=[(0, 17), (18, 35)], tp_degrees=[2])

    def test_multimodal_tp_input_owner_broadcasts_startup_contract_once(self) -> None:
        manifest = _build_tp_manifest(
            stage_ranges=[(0, 35)],
            tp_degrees=[2],
            modality="multimodal",
        )
        startup_contract = _build_mm_startup_contract(num_stages=1)
        startup_meta, startup_tensors = pack_mm_startup_transport(
            select_mm_startup_contract(startup_contract, local_stage_indices=[0]),
            include_stage_output=False,
            include_derived_shared=False,
        )
        builder_instance = MagicMock()
        builder_instance.__enter__.return_value = builder_instance
        builder_instance.export_mm_startup_transport.return_value = (startup_meta, startup_tensors)
        scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }
        local_state = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.DirectStageStateBuilder",
            return_value=builder_instance,
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_object_cpu",
            return_value=startup_meta,
        ) as bcast_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_tensor_payload_cpu",
            return_value=startup_tensors,
        ) as bcast_tensor_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.build_direct_stage_state",
            return_value=scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.materialize_text_stage_state",
            return_value=local_state,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_capture_common.move_bundle",
            return_value=local_state,
        ):
            stage_state, compute_dtype = load_stage_state_for_tp_rank(
                manifest,
                rank=0,
                world_size=2,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        builder_cls.assert_called_once()
        builder_kwargs = builder_cls.call_args.kwargs
        self.assertEqual(builder_kwargs["stage_specs"], manifest.stages)
        self.assertFalse(builder_kwargs["include_text_weights"])
        self.assertTrue(builder_kwargs["mm_activate_frontend"])
        builder_instance.export_mm_startup_transport.assert_called_once_with(local_stage_indices=[0])
        bcast_meta_mock.assert_called_once()
        self.assertEqual(
            bcast_meta_mock.call_args.kwargs["label"],
            "tp_multimodal_startup_contract_meta stage_idx=0",
        )
        bcast_tensor_mock.assert_called_once()
        self.assertEqual(
            bcast_tensor_mock.call_args.kwargs["label"],
            "tp_multimodal_startup_contract_tensors stage_idx=0",
        )
        build_mock.assert_called_once()
        self.assertFalse(build_mock.call_args.kwargs["mm_activate_frontend"])
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(build_runtime_config["_mm_startup_contract_ready"])
        self.assertEqual(sorted(build_runtime_config["_mm_startup_stage_handoffs"]), [0])
        self.assertNotIn("_mm_startup_root_input", build_runtime_config)
        self.assertNotIn("_mm_startup_boundaries", build_runtime_config)
        materialize_mock.assert_called_once()
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 0)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        self.assertIs(stage_state, local_state)
        self.assertEqual(compute_dtype, torch.float32)

    def test_multimodal_tp_follower_consumes_input_owner_contract(self) -> None:
        manifest = _build_tp_manifest(
            stage_ranges=[(0, 35)],
            tp_degrees=[2],
            modality="multimodal",
        )
        startup_contract = _build_mm_startup_contract(num_stages=1)
        startup_meta, startup_tensors = pack_mm_startup_transport(
            select_mm_startup_contract(startup_contract, local_stage_indices=[0]),
            include_stage_output=False,
            include_derived_shared=False,
        )
        scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }
        local_state = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.DirectStageStateBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_object_cpu",
            return_value=startup_meta,
        ) as bcast_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_tensor_payload_cpu",
            return_value=startup_tensors,
        ) as bcast_tensor_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.build_direct_stage_state",
            return_value=scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_runtime_builder.materialize_text_stage_state",
            return_value=local_state,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_capture_common.move_bundle",
            return_value=local_state,
        ):
            stage_state, compute_dtype = load_stage_state_for_tp_rank(
                manifest,
                rank=1,
                world_size=2,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        builder_cls.assert_not_called()
        bcast_meta_mock.assert_called_once()
        self.assertEqual(bcast_meta_mock.call_args.kwargs["src"], 0)
        self.assertEqual(
            bcast_meta_mock.call_args.kwargs["label"],
            "tp_multimodal_startup_contract_meta stage_idx=0",
        )
        bcast_tensor_mock.assert_called_once()
        self.assertEqual(bcast_tensor_mock.call_args.kwargs["src"], 0)
        self.assertEqual(
            bcast_tensor_mock.call_args.kwargs["label"],
            "tp_multimodal_startup_contract_tensors stage_idx=0",
        )
        build_mock.assert_called_once()
        self.assertFalse(build_mock.call_args.kwargs["mm_activate_frontend"])
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(build_runtime_config["_mm_startup_contract_ready"])
        self.assertEqual(sorted(build_runtime_config["_mm_startup_stage_handoffs"]), [0])
        self.assertNotIn("_mm_startup_root_input", build_runtime_config)
        self.assertNotIn("_mm_startup_boundaries", build_runtime_config)
        materialize_mock.assert_called_once()
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 1)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        self.assertIs(stage_state, local_state)
        self.assertEqual(compute_dtype, torch.float32)

    def test_generate_phase_uses_local_prefill_embeddings_without_stage_input_broadcast(self) -> None:
        embed_tokens_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        )
        runtime_state = {
            "prefill_seq_len": 3,
            "batch_size": 1,
            "hidden_size": 2,
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "embed_tokens_weight": embed_tokens_weight,
            "layers": [],
        }
        expected_stage_input = torch.nn.functional.embedding(
            runtime_state["input_ids"],
            embed_tokens_weight,
        )
        trace_result = {
            "logits": torch.tensor([[[0.0, 0.1, 0.2], [0.0, 0.3, 0.1], [0.9, 0.1, 0.0]]]),
            "stage_output": expected_stage_input,
            "norm_output": expected_stage_input,
            "cache_by_layer": {},
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_cpu",
            side_effect=AssertionError("stage_input broadcast should be skipped"),
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_execution.trace_text_decode_logits_tp_with_runtime_cache",
            return_value=trace_result,
        ) as trace_mock:
            stats, cache = tensor_parallel_module._run_generate_phase_tp(
                rank=1,
                world_size=2,
                runtime_state=runtime_state,
                phase_kind="prefill",
                current_token_id=None,
                cache_by_layer=None,
                comm_dtype=torch.float32,
                tp_attn_math_mode="orig",
                tp_mlp_math_mode="orig",
                return_tensor=False,
            )

        sent_stage_input = trace_mock.call_args.args[0]
        self.assertTrue(torch.equal(sent_stage_input, expected_stage_input))
        self.assertEqual(stats["runtime_input_source"], "local_embeddings")
        self.assertTrue(stats["runtime_input_broadcast_skipped"])
        self.assertIsNone(stats["predicted_token_id"])
        self.assertEqual(cache, {})

    def test_generate_phase_uses_local_decode_embedding_without_stage_input_broadcast(self) -> None:
        embed_tokens_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        )
        runtime_state = {
            "prefill_seq_len": 3,
            "batch_size": 1,
            "hidden_size": 2,
            "decode_input_ids": torch.zeros((1, 1), dtype=torch.long),
            "embed_tokens_weight": embed_tokens_weight,
            "layers": [],
        }
        expected_stage_input = torch.nn.functional.embedding(
            torch.tensor([[2]], dtype=torch.long),
            embed_tokens_weight,
        )
        trace_result = {
            "logits": torch.tensor([[[0.0, 0.1, 0.8]]]),
            "stage_output": expected_stage_input,
            "norm_output": expected_stage_input,
            "cache_by_layer": {0: (None, None)},
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.distributed_backend.broadcast_cpu",
            side_effect=AssertionError("stage_input broadcast should be skipped"),
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.qwen_execution.trace_text_decode_logits_tp_with_runtime_cache",
            return_value=trace_result,
        ) as trace_mock:
            stats, cache = tensor_parallel_module._run_generate_phase_tp(
                rank=1,
                world_size=2,
                runtime_state=runtime_state,
                phase_kind="decode",
                current_token_id=2,
                cache_by_layer={},
                comm_dtype=torch.float32,
                tp_attn_math_mode="orig",
                tp_mlp_math_mode="orig",
                return_tensor=False,
            )

        sent_stage_input = trace_mock.call_args.args[0]
        self.assertTrue(torch.equal(sent_stage_input, expected_stage_input))
        self.assertTrue(torch.equal(runtime_state["decode_input_ids_runtime"], torch.tensor([[2]])))
        self.assertEqual(stats["runtime_input_source"], "local_embeddings")
        self.assertTrue(stats["runtime_input_broadcast_skipped"])
        self.assertIsNone(stats["predicted_token_id"])
        self.assertEqual(cache, {0: (None, None)})


if __name__ == "__main__":
    unittest.main()
