from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import load_stage_bundle_for_hybrid_rank
from qwen3vl_tp_runtime.hexgen_core.schema import HybridRankContext, StageSpec, TextHybridManifest
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmFrontendSeed,
    compact_mm_runtime_shared,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    pack_mm_startup_transport,
    pack_text_scaffold_transport,
    restore_mm_startup_transport,
    select_mm_startup_contract,
)


def _build_manifest(*, stage_ranges: list[tuple[int, int]], tp_degrees: list[int], modality: str) -> TextHybridManifest:
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
    world_size = sum(tp_degrees)
    return TextHybridManifest(
        runtime="text_generate_hybrid",
        tp_degrees=tp_degrees,
        stage_rank_groups=[],
        pp_rank_groups=[],
        world_size=world_size,
        num_stages=len(stages),
        send_list=[],
        recv_list=[],
        send_empty_list=[],
        recv_empty_list=[],
        stage_ranges=stage_ranges,
        bundle_dir=None,
        stages=stages,
        boundaries=[],
        num_frames=0,
        save_dtype="float32",
        pipeline_type="text_generate",
        runtime_config={
            "modality": modality,
            "mode": "generate",
            "model_path": "/tmp/fake-model",
            "save_dtype": "float32",
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


class HybridDirectLoaderTest(unittest.TestCase):
    def test_direct_detection_falls_back_without_manifest_property(self) -> None:
        from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import _all_hybrid_stages_are_direct

        legacy_manifest = SimpleNamespace(
            stages=[
                SimpleNamespace(bundle_path=None),
                SimpleNamespace(bundle_path=None),
            ]
        )
        captured_manifest = SimpleNamespace(
            stages=[
                SimpleNamespace(bundle_path="/tmp/stage_00.pt"),
                SimpleNamespace(bundle_path=None),
            ]
        )

        self.assertTrue(_all_hybrid_stages_are_direct(legacy_manifest))
        self.assertFalse(_all_hybrid_stages_are_direct(captured_manifest))

    def test_multimodal_single_rank_direct_stage_skips_text_prompt_metadata(self) -> None:
        manifest = _build_manifest(stage_ranges=[(18, 35)], tp_degrees=[1], modality="multimodal")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[2],
            tp_degree=1,
            local_rank=0,
            leader_rank=2,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group=None,
            pp_group_idx=0,
            current_pp_group=[2],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }
        startup_contract = _build_mm_startup_contract(num_stages=1)
        local_startup_contract = select_mm_startup_contract(startup_contract, local_stage_indices=[0])
        local_startup_meta, local_startup_tensors = pack_mm_startup_transport(local_startup_contract)

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageBundleBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.recv_object_cpu",
            return_value=local_startup_meta,
        ) as recv_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.recv_tensor_payload_cpu",
            return_value=local_startup_tensors,
        ) as recv_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=2,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        builder_cls.assert_not_called()
        build_mock.assert_called_once()
        self.assertFalse(build_mock.call_args.kwargs["mm_activate_frontend"])
        recv_meta_mock.assert_called_once()
        self.assertEqual(
            recv_meta_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_meta stage_idx=0",
        )
        recv_mock.assert_called_once()
        self.assertEqual(
            recv_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_tensors stage_idx=0",
        )
        self.assertIn("_mm_startup_shared", manifest.runtime_config)
        self.assertIn("_mm_startup_stage_handoffs", manifest.runtime_config)
        self.assertEqual(sorted(manifest.runtime_config["_mm_startup_stage_handoffs"]), [0])
        self.assertIn("_mm_startup_root_input", manifest.runtime_config)
        self.assertTrue(torch.equal(manifest.runtime_config["_mm_startup_root_input"], torch.zeros(1, 3, 4)))
        self.assertIn("_mm_startup_stage_visuals", manifest.runtime_config)
        self.assertEqual(sorted(manifest.runtime_config["_mm_startup_stage_visuals"]), [0])
        self.assertNotIn("_mm_startup_boundaries", manifest.runtime_config)
        self.assertNotIn("_mm_startup_visual_pos_masks", manifest.runtime_config)
        self.assertNotIn("_mm_startup_deepstack_by_layer", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_plan", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_state", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_seed", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_meta", manifest.runtime_config)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 18)

    def test_multimodal_stage_zero_single_rank_marks_frontend_active(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17)], tp_degrees=[1], modality="multimodal")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[0],
            tp_degree=1,
            local_rank=0,
            leader_rank=0,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group=None,
            pp_group_idx=0,
            current_pp_group=[0],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [],
        }
        startup_contract = _build_mm_startup_contract(num_stages=1)
        builder_instance = MagicMock()
        builder_instance.__enter__.return_value = builder_instance
        builder_instance.export_mm_startup_transport.return_value = pack_mm_startup_transport(
            select_mm_startup_contract(
                startup_contract,
                local_stage_indices=[0],
            ),
        )

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageBundleBuilder",
            return_value=builder_instance,
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.send_object_cpu",
        ) as send_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.send_tensor_payload_cpu",
        ) as send_tensor_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=0,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        builder_cls.assert_called_once()
        builder_kwargs = builder_cls.call_args.kwargs
        self.assertEqual(builder_kwargs["stage_specs"], manifest.stages)
        self.assertEqual(builder_kwargs["runtime_config"]["modality"], "multimodal")
        self.assertEqual(builder_kwargs["runtime_config"]["mode"], "generate")
        self.assertEqual(builder_kwargs["runtime_config"]["model_path"], "/tmp/fake-model")
        self.assertEqual(builder_kwargs["runtime_config"]["save_dtype"], "float32")
        self.assertFalse(builder_kwargs["include_text_weights"])
        self.assertTrue(builder_kwargs["mm_activate_frontend"])
        builder_instance.export_mm_startup_transport.assert_called_once_with(local_stage_indices=[0])
        self.assertTrue(build_mock.call_args.kwargs["mm_activate_frontend"])
        send_meta_mock.assert_not_called()
        send_tensor_mock.assert_not_called()
        self.assertIn("_mm_startup_shared", manifest.runtime_config)
        self.assertIn("_mm_startup_stage_handoffs", manifest.runtime_config)
        self.assertEqual(sorted(manifest.runtime_config["_mm_startup_stage_handoffs"]), [0])
        self.assertIn("_mm_startup_root_input", manifest.runtime_config)
        self.assertTrue(torch.equal(manifest.runtime_config["_mm_startup_root_input"], torch.zeros(1, 3, 4)))
        self.assertIn("_mm_startup_stage_visuals", manifest.runtime_config)
        self.assertEqual(sorted(manifest.runtime_config["_mm_startup_stage_visuals"]), [0])
        self.assertNotIn("_mm_startup_boundaries", manifest.runtime_config)
        self.assertNotIn("_mm_startup_visual_pos_masks", manifest.runtime_config)
        self.assertNotIn("_mm_startup_deepstack_by_layer", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_plan", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_state", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_seed", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_meta", manifest.runtime_config)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 0)

    def test_multimodal_tp_follower_uses_stage_scaffold_transport_and_local_materialize(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17)], tp_degrees=[2], modality="multimodal")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[0, 1],
            tp_degree=2,
            local_rank=1,
            leader_rank=0,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group="fake-group",
            pp_group_idx=0,
            current_pp_group=[1],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        leader_scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [],
        }
        bundle_meta, bundle_tensors = pack_text_scaffold_transport(leader_scaffold)
        local_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [{"layer_idx": 0}],
            "tp_weight_sharded": True,
            "tp_shard_rank": 1,
            "tp_shard_world_size": 2,
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageBundleBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            return_value=bundle_meta,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value=bundle_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.recv_object_cpu",
        ) as recv_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.send_object_cpu",
        ) as send_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=1,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        builder_cls.assert_not_called()
        recv_mock.assert_not_called()
        send_mock.assert_not_called()
        build_mock.assert_not_called()
        self.assertEqual(bcast_mock.call_count, 1)
        self.assertEqual(bcast_mock.call_args.kwargs["label"], "stage_scaffold_meta stage_idx=0")
        tensor_bcast_mock.assert_called_once()
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "stage_scaffold_tensors stage_idx=0")
        self.assertNotIn("_mm_startup_shared", manifest.runtime_config)
        self.assertNotIn("_mm_startup_stage_handoffs", manifest.runtime_config)
        self.assertNotIn("_mm_startup_stage_visuals", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_plan", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_state", manifest.runtime_config)
        self.assertNotIn("_mm_frontend_seed", manifest.runtime_config)
        materialize_mock.assert_called_once()
        restored_scaffold = materialize_mock.call_args.kwargs["stage_bundle_scaffold"]
        self.assertEqual(restored_scaffold["save_dtype"], leader_scaffold["save_dtype"])
        self.assertEqual(materialize_mock.call_args.kwargs["runtime_config"], manifest.runtime_config)
        self.assertEqual(materialize_mock.call_args.kwargs["compute_dtype"], torch.float32)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 1)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["end_idx"], 17)

    def test_single_rank_direct_stage_skips_stage_bundle_broadcast(self) -> None:
        manifest = _build_manifest(stage_ranges=[(18, 35)], tp_degrees=[1], modality="text")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[2],
            tp_degree=1,
            local_rank=0,
            leader_rank=2,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group=None,
            pp_group_idx=0,
            current_pp_group=[2],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=2,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        build_mock.assert_called_once()
        bcast_mock.assert_not_called()
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 18)
        self.assertEqual(bundle["end_idx"], 35)

    def test_text_tp_follower_uses_scaffold_broadcast_and_local_materialize(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17)], tp_degrees=[2], modality="text")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[0, 1],
            tp_degree=2,
            local_rank=1,
            leader_rank=0,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group="fake-group",
            pp_group_idx=0,
            current_pp_group=[1],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        scaffold = {
            "save_dtype": "bfloat16",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [],
            "prefill": {
                "stage_input": torch.ones(1, 3, 4, dtype=torch.float32),
                "attention_mask_2d": torch.ones(1, 3, dtype=torch.int64),
            },
            "decode_steps": [
                {
                    "stage_input": torch.ones(1, 1, 4, dtype=torch.float32) * 2,
                    "attention_mask_2d": torch.ones(1, 4, dtype=torch.int64),
                }
            ],
        }
        scaffold_meta, scaffold_tensors = pack_text_scaffold_transport(scaffold)
        local_bundle = {
            "save_dtype": "bfloat16",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [{"layer_idx": 0}],
            "tp_weight_sharded": True,
            "tp_shard_rank": 1,
            "tp_shard_world_size": 2,
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            return_value=scaffold_meta,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value=scaffold_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=1,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="auto",
            )

        build_mock.assert_not_called()
        bcast_mock.assert_called_once()
        self.assertEqual(bcast_mock.call_args.kwargs["label"], "text_scaffold_meta stage_idx=0")
        tensor_bcast_mock.assert_called_once()
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "text_scaffold_tensors stage_idx=0")
        materialize_mock.assert_called_once()
        restored_scaffold = materialize_mock.call_args.kwargs["stage_bundle_scaffold"]
        self.assertEqual(restored_scaffold["save_dtype"], scaffold["save_dtype"])
        self.assertTrue(torch.equal(restored_scaffold["prefill"]["stage_input"], scaffold["prefill"]["stage_input"]))
        self.assertTrue(
            torch.equal(
                restored_scaffold["decode_steps"][0]["attention_mask_2d"],
                scaffold["decode_steps"][0]["attention_mask_2d"],
            )
        )
        self.assertEqual(materialize_mock.call_args.kwargs["runtime_config"], manifest.runtime_config)
        self.assertEqual(materialize_mock.call_args.kwargs["compute_dtype"], torch.bfloat16)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 1)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.bfloat16)
        self.assertEqual(bundle["end_idx"], 17)

    def test_text_tp_leader_builds_weightless_scaffold_then_materializes_local_shard(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 35)], tp_degrees=[2], modality="text")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[0, 1],
            tp_degree=2,
            local_rank=0,
            leader_rank=0,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group="fake-group",
            pp_group_idx=0,
            current_pp_group=[0],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }
        local_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [{"layer_idx": 0}],
            "tp_weight_sharded": True,
            "tp_shard_rank": 0,
            "tp_shard_world_size": 2,
        }

        def _echo_object(payload, **_kwargs):
            return payload

        def _echo_tensors(payload, **_kwargs):
            return payload

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            side_effect=_echo_object,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            side_effect=_echo_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=0,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        build_mock.assert_called_once()
        self.assertFalse(build_mock.call_args.kwargs["include_text_weights"])
        self.assertIsNone(build_mock.call_args.kwargs["mm_activate_frontend"])
        bcast_mock.assert_called_once()
        self.assertEqual(bcast_mock.call_args.kwargs["label"], "text_scaffold_meta stage_idx=0")
        tensor_bcast_mock.assert_called_once()
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "text_scaffold_tensors stage_idx=0")
        materialize_mock.assert_called_once()
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 0)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertTrue(bundle["tp_weight_sharded"])
        self.assertEqual(bundle["tp_shard_rank"], 0)
        self.assertEqual(bundle["tp_shard_world_size"], 2)

    def test_direct_tp_stage_rejects_materialized_full_weight_bundle(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 35)], tp_degrees=[2], modality="text")
        rank_stage = HybridRankContext(
            stage_idx=0,
            stage_ranks=[0, 1],
            tp_degree=2,
            local_rank=1,
            leader_rank=0,
            prev_leader_rank=None,
            next_leader_rank=None,
            stage_group="fake-group",
            pp_group_idx=0,
            current_pp_group=[1],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )
        scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [],
        }
        scaffold_meta, scaffold_tensors = pack_text_scaffold_transport(scaffold)
        full_weight_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 35,
            "layers": [{"layer_idx": 0}],
            "tp_weight_sharded": False,
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            return_value=scaffold_meta,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value=scaffold_tensors,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage",
            return_value=full_weight_bundle,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            with self.assertRaisesRegex(RuntimeError, "rank-local shard bundle"):
                load_stage_bundle_for_hybrid_rank(
                    manifest,
                    rank=1,
                    rank_stage=rank_stage,
                    device=torch.device("cpu"),
                    compute_dtype_arg="float32",
                )

        barrier_mock.assert_not_called()

    def test_single_rank_direct_stage_seeds_runtime_only_prompt_metadata_before_build(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], tp_degrees=[2, 1], modality="text")
        manifest.runtime_config["include_runtime_reference"] = False
        rank_stage = HybridRankContext(
            stage_idx=1,
            stage_ranks=[2],
            tp_degree=1,
            local_rank=0,
            leader_rank=2,
            prev_leader_rank=0,
            next_leader_rank=None,
            stage_group=None,
            pp_group_idx=1,
            current_pp_group=[1, 2],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        prompt_metadata = {
            "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 1,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value=prompt_metadata,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=2,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        bcast_mock.assert_called_once()
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(torch.equal(build_runtime_config["_runtime_only_input_ids"], prompt_metadata["input_ids"]))
        self.assertTrue(
            torch.equal(build_runtime_config["_runtime_only_attention_mask"], prompt_metadata["attention_mask"])
        )
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 18)

    def test_runtime_only_prompt_metadata_compact_broadcast_restores_input_ids(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], tp_degrees=[2, 1], modality="text")
        manifest.runtime_config["include_runtime_reference"] = False
        rank_stage = HybridRankContext(
            stage_idx=1,
            stage_ranks=[2],
            tp_degree=1,
            local_rank=0,
            leader_rank=2,
            prev_leader_rank=0,
            next_leader_rank=None,
            stage_group=None,
            pp_group_idx=1,
            current_pp_group=[1, 2],
            send_list=[],
            recv_list=[],
            send_empty_list=[],
            recv_empty_list=[],
        )

        prompt_metadata = {
            "input_ids_list": [7, 8, 9],
        }
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 1,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value={"input_ids": torch.tensor([[7, 8, 9]], dtype=torch.int64)},
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_hybrid_rank(
                manifest,
                rank=2,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        bcast_mock.assert_called_once()
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(
            torch.equal(
                build_runtime_config["_runtime_only_input_ids"],
                torch.tensor([[7, 8, 9]], dtype=torch.int64),
            )
        )
        self.assertNotIn("_runtime_only_attention_mask", build_runtime_config)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 18)


if __name__ == "__main__":
    unittest.main()
