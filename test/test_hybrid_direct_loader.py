from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from qwen3vl_tp_runtime.hexgen_core.modules import hybrid_parallel as hybrid_parallel_module
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import load_stage_state_for_hybrid_rank
from qwen3vl_tp_runtime.hexgen_core.schema import (
    HYBRID_RUNTIME_INPUT_PROTOCOL,
    HybridRankContext,
    HybridRuntimeInputSchema,
    StageSpec,
    TextHybridManifest,
)
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


def _fake_tp_projection_layer(layer_idx: int = 0) -> dict[str, object]:
    return {
        "layer_idx": layer_idx,
        "head_dim": 2,
        "tp_weight_sharded": True,
        "tp_shard_rank": 1,
        "tp_shard_world_size": 2,
        "tp_local_num_attention_heads": 2,
        "tp_local_num_key_value_heads": 1,
        "tp_local_intermediate_size": 8,
        "input_ln_weight": torch.ones(8),
        "q_weight": torch.zeros(4, 8),
        "k_weight": torch.zeros(2, 8),
        "v_weight": torch.zeros(2, 8),
        "o_weight": torch.zeros(8, 4),
        "gate_weight": torch.zeros(8, 8),
        "up_weight": torch.zeros(8, 8),
        "down_weight": torch.zeros(8, 8),
    }


def _fake_tp_sharded_bundle(
    *,
    stage_idx: int,
    start_idx: int,
    end_idx: int,
    tp_shard_rank: int,
    save_dtype: str = "float32",
) -> dict[str, object]:
    layer = _fake_tp_projection_layer(start_idx)
    layer["tp_shard_rank"] = tp_shard_rank
    return {
        "save_dtype": save_dtype,
        "stage_idx": stage_idx,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "layers": [layer],
        "tp_weight_sharded": True,
        "tp_shard_rank": tp_shard_rank,
        "tp_shard_world_size": 2,
    }


class HybridDirectLoaderTest(unittest.TestCase):
    def test_runtime_only_decode_phase_drops_prefill_handoff_reference(self) -> None:
        stage_bundle = {
            "modality": "multimodal",
            "prefill_seq_len": 3,
            "prefill_attention_mask_2d": torch.ones(1, 3, dtype=torch.long),
            "batch_size": 1,
            "hidden_size": 4,
            "token_id_dtype": "int64",
            "stage_input": torch.ones(1, 3, 4),
            "layer_input": torch.ones(1, 3, 4),
            "stage_output": torch.ones(1, 3, 4),
            "layer_output": torch.ones(1, 3, 4),
            "layers": [{"q_weight": torch.zeros(1, 1)}],
            "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_mm_decode_state_from_weights",
            return_value=SimpleNamespace(
                attention_mask=torch.zeros(1, 1, 1, 4),
                position_ids=torch.zeros(4, 1, 1, dtype=torch.long),
                cos=torch.zeros(1, 1, 4),
                sin=torch.zeros(1, 1, 4),
            ),
        ):
            decode_bundle = hybrid_parallel_module._build_runtime_only_text_generate_phase_state(
                stage_bundle,
                phase_kind="decode",
                attention_mask_2d=torch.ones(1, 4, dtype=torch.long),
                config_spec=SimpleNamespace(hidden_size=4),
                rotary_emb=None,
            )

        self.assertNotIn("stage_input", decode_bundle)
        self.assertNotIn("layer_input", decode_bundle)
        self.assertNotIn("stage_output", decode_bundle)
        self.assertNotIn("layer_output", decode_bundle)
        self.assertEqual(tuple(decode_bundle["decode_input_ids"].shape), (1, 1))

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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageStateBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
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
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        self.assertNotIn("_mm_startup_root_input", manifest.runtime_config)
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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageStateBuilder",
            return_value=builder_instance,
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.send_object_cpu",
        ) as send_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.send_tensor_payload_cpu",
        ) as send_tensor_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        self.assertNotIn("_mm_startup_root_input", manifest.runtime_config)
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
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=17,
            tp_shard_rank=1,
        )

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.DirectStageStateBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
        self.assertEqual(restored_scaffold["save_dtype"], leader_scaffold["save_dtype"])
        self.assertEqual(materialize_mock.call_args.kwargs["runtime_config"], manifest.runtime_config)
        self.assertEqual(materialize_mock.call_args.kwargs["compute_dtype"], torch.float32)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_rank"], 1)
        self.assertEqual(materialize_mock.call_args.kwargs["tp_shard_world_size"], 2)
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["end_idx"], 17)

    def test_multimodal_tp_leader_strips_prefill_scaffold_tensors_before_broadcast(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17)], tp_degrees=[2], modality="multimodal")
        manifest.runtime_config["_mm_startup_contract_ready"] = True
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

        leader_scaffold = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "hidden_size": 4,
            "batch_size": 1,
            "runtime_only_generate": True,
            "modality": "multimodal",
            "num_frames": 2,
            "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            "layers": [],
            "stage_input": torch.ones(1, 3, 4, dtype=torch.float32),
            "layer_input": torch.ones(1, 3, 4, dtype=torch.float32),
            "prefill_attention_mask_2d": torch.ones(1, 3, dtype=torch.long),
            "prefill_attention_mask": torch.zeros(1, 1, 3, 3, dtype=torch.float32),
            "prefill_position_ids": torch.zeros(4, 1, 3, dtype=torch.long),
            "prefill_cos": torch.zeros(1, 3, 4, dtype=torch.float32),
            "prefill_sin": torch.zeros(1, 3, 4, dtype=torch.float32),
            "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        }
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=17,
            tp_shard_rank=0,
        )

        def _echo_object(payload, **_kwargs):
            return payload

        def _echo_tensors(payload, **_kwargs):
            return payload

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=leader_scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            side_effect=_echo_object,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            side_effect=_echo_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ):
            load_stage_state_for_hybrid_rank(
                manifest,
                rank=0,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        build_mock.assert_called_once()
        sent_meta = bcast_mock.call_args.args[0]
        sent_scaffold_meta = sent_meta["scaffold"]
        self.assertIn("mm_prefill_runtime_tensors_local_rebuild", sent_scaffold_meta)
        self.assertIn("mm_frontend_metadata_local_rebuild", sent_scaffold_meta)
        for field_name in (
            "prefill_attention_mask_2d",
            "prefill_attention_mask",
            "prefill_position_ids",
            "prefill_cos",
            "prefill_sin",
            "num_frames",
            "frame_paths",
        ):
            self.assertNotIn(field_name, sent_scaffold_meta)
        sent_tensors = tensor_bcast_mock.call_args.args[0]
        for tensor_key in (
            "scaffold.prefill_attention_mask_2d",
            "scaffold.prefill_attention_mask",
            "scaffold.prefill_position_ids",
            "scaffold.prefill_cos",
            "scaffold.prefill_sin",
        ):
            self.assertNotIn(tensor_key, sent_tensors)
        materialize_mock.assert_called_once()
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
        self.assertNotIn("prefill_attention_mask_2d", restored_scaffold)
        self.assertNotIn("num_frames", restored_scaffold)
        self.assertNotIn("frame_paths", restored_scaffold)
        self.assertIn("stage_input", restored_scaffold)

    def test_multimodal_runtime_only_tp_leader_broadcasts_runtime_inputs_not_scaffold(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17)], tp_degrees=[2], modality="multimodal")
        manifest.runtime_config["include_runtime_reference"] = False
        manifest.runtime_config["_mm_startup_contract_ready"] = True
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

        stage_input = torch.ones(1, 3, 4, dtype=torch.float32)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask_2d = torch.ones(1, 3, dtype=torch.long)
        rope_deltas = torch.zeros(1, 1, dtype=torch.long)
        manifest.runtime_config["_mm_startup_shared"] = {
            "input_ids": input_ids,
            "attention_mask_2d": attention_mask_2d,
            "rope_deltas": rope_deltas,
        }
        manifest.runtime_config["_mm_startup_stage_handoffs"] = {
            0: {
                "stage_input": stage_input,
                "stage_output": torch.zeros_like(stage_input),
            },
        }
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=17,
            tp_shard_rank=0,
        )

        def _echo_object(payload, **_kwargs):
            return payload

        def _echo_tensors(payload, **_kwargs):
            return payload

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            side_effect=_echo_object,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            side_effect=_echo_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ):
            load_stage_state_for_hybrid_rank(
                manifest,
                rank=0,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        build_mock.assert_not_called()
        self.assertEqual(bcast_mock.call_args.kwargs["label"], "runtime_inputs_meta stage_idx=0")
        sent_meta = bcast_mock.call_args.args[0]
        self.assertIn("runtime_inputs", sent_meta)
        self.assertNotIn("scaffold", sent_meta)
        sent_runtime_inputs_meta = sent_meta["runtime_inputs"]
        self.assertEqual(sent_runtime_inputs_meta["protocol"], "hybrid_runtime_inputs_v1")
        for field_name in (
            "layers",
            "module_name",
            "stage_type",
            "stage_idx",
            "start_idx",
            "end_idx",
            "save_dtype",
            "hidden_size",
            "batch_size",
            "num_frames",
            "frame_paths",
            "prefill_attention_mask_2d",
            "prefill_attention_mask",
            "prefill_position_ids",
            "prefill_cos",
            "prefill_sin",
            "stage_input",
            "rope_deltas",
        ):
            self.assertNotIn(field_name, sent_runtime_inputs_meta)
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "runtime_inputs_tensors stage_idx=0")
        sent_tensors = tensor_bcast_mock.call_args.args[0]
        self.assertEqual(
            set(sent_tensors),
            {
                "runtime_inputs.shared.input_ids",
                "runtime_inputs.shared.rope_deltas",
                "runtime_inputs.stage_handoff.stage_input",
            },
        )
        materialize_mock.assert_called_once()
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
        self.assertTrue(restored_scaffold["runtime_inputs_from_broadcast"])
        self.assertEqual(restored_scaffold["stage_idx"], 0)
        self.assertEqual(restored_scaffold["start_idx"], 0)
        self.assertEqual(restored_scaffold["end_idx"], 17)
        self.assertEqual(restored_scaffold["save_dtype"], "float32")
        self.assertTrue(torch.equal(restored_scaffold["stage_input"], stage_input))
        self.assertTrue(torch.equal(restored_scaffold["rope_deltas"], rope_deltas))
        materialize_runtime_config = materialize_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(torch.equal(materialize_runtime_config["_mm_startup_shared"]["input_ids"], input_ids))
        self.assertTrue(
            torch.equal(
                materialize_runtime_config["_mm_startup_stage_handoffs"][0]["stage_input"],
                stage_input,
            )
        )
        self.assertNotIn("layers", sent_runtime_inputs_meta)
        self.assertEqual(restored_scaffold["layers"], [])

    def test_multimodal_runtime_input_builder_omits_rebuildable_shared_tensors(self) -> None:
        runtime_config = {
            "_mm_startup_shared": {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask_2d": torch.ones(1, 3, dtype=torch.long),
                "position_ids": torch.zeros(4, 1, 3, dtype=torch.long),
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 0, 0]], dtype=torch.int),
                "attention_mask": torch.zeros(1, 1, 3, 3),
            },
            "_mm_startup_stage_handoffs": {
                0: {
                    "stage_input": torch.zeros(1, 3, 4),
                },
            },
        }

        payload = hybrid_parallel_module._build_runtime_input_broadcast_payload(
            runtime_config,
            stage_idx=0,
            runtime_modality="multimodal",
        )
        self.assertNotIn("attention_mask_2d", payload["shared"])
        self.assertNotIn("position_ids", payload["shared"])
        self.assertNotIn("attention_mask", payload["shared"])

    def test_multimodal_runtime_input_schema_rejects_derived_shared_tensors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "禁止广播"):
            HybridRuntimeInputSchema.validate(
                {
                    "protocol": HYBRID_RUNTIME_INPUT_PROTOCOL,
                    "modality": "multimodal",
                    "mode": "generate",
                    "runtime_only_generate": True,
                    "shared": {
                        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                        "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                        "attention_mask": torch.zeros(1, 1, 3, 3),
                    },
                    "stage_handoff": {
                        "stage_input": torch.zeros(1, 3, 4),
                    },
                },
                context="test",
            )

    def test_text_runtime_only_tp_leader_broadcasts_prompt_runtime_inputs(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 35)], tp_degrees=[2], modality="text")
        manifest.runtime_config["include_runtime_reference"] = False
        manifest.runtime_config["_runtime_only_prompt_metadata_ready"] = True
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        manifest.runtime_config["_runtime_only_input_ids"] = input_ids
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
            "hidden_size": 4,
            "batch_size": 1,
            "runtime_only_generate": True,
            "modality": "text",
            "layers": [],
        }
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=35,
            tp_shard_rank=0,
        )

        def _echo_object(payload, **_kwargs):
            return payload

        def _echo_tensors(payload, **_kwargs):
            return payload

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            side_effect=_echo_object,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            side_effect=_echo_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ):
            load_stage_state_for_hybrid_rank(
                manifest,
                rank=0,
                rank_stage=rank_stage,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        build_mock.assert_not_called()
        self.assertEqual(bcast_mock.call_args.kwargs["label"], "runtime_inputs_meta stage_idx=0")
        sent_meta = bcast_mock.call_args.args[0]
        self.assertIn("runtime_inputs", sent_meta)
        self.assertNotIn("scaffold", sent_meta)
        sent_runtime_inputs_meta = sent_meta["runtime_inputs"]
        self.assertEqual(sent_runtime_inputs_meta["protocol"], "hybrid_runtime_inputs_v1")
        for field_name in (
            "layers",
            "module_name",
            "stage_type",
            "stage_idx",
            "start_idx",
            "end_idx",
            "save_dtype",
            "hidden_size",
            "batch_size",
        ):
            self.assertNotIn(field_name, sent_runtime_inputs_meta)
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "runtime_inputs_tensors stage_idx=0")
        self.assertEqual(set(tensor_bcast_mock.call_args.args[0]), {"runtime_inputs.input_ids"})
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
        self.assertTrue(restored_scaffold["runtime_inputs_from_broadcast"])
        self.assertTrue(restored_scaffold["runtime_only_prompt_local_rebuild"])
        self.assertEqual(restored_scaffold["start_idx"], 0)
        self.assertEqual(restored_scaffold["end_idx"], 35)
        self.assertEqual(restored_scaffold["layers"], [])
        materialize_runtime_config = materialize_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(torch.equal(materialize_runtime_config["_runtime_only_input_ids"], input_ids))

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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=17,
            tp_shard_rank=1,
            save_dtype="bfloat16",
        )

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            return_value=scaffold_meta,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            return_value=scaffold_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
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
            "hidden_size": 4,
            "batch_size": 1,
            "layers": [],
        }
        local_bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=35,
            tp_shard_rank=0,
        )

        def _echo_object(payload, **_kwargs):
            return payload

        def _echo_tensors(payload, **_kwargs):
            return payload

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=scaffold,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            side_effect=_echo_object,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_tensor_payload_cpu",
            side_effect=_echo_tensors,
        ) as tensor_bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=local_bundle,
        ) as materialize_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
        sent_meta = bcast_mock.call_args.args[0]
        sent_scaffold_meta = sent_meta["scaffold"]
        for field_name in ("stage_idx", "start_idx", "end_idx", "save_dtype", "hidden_size", "batch_size"):
            self.assertNotIn(field_name, sent_scaffold_meta)
        self.assertIn("rank_local_fields_local_rebuild", sent_scaffold_meta)
        tensor_bcast_mock.assert_called_once()
        self.assertEqual(tensor_bcast_mock.call_args.kwargs["label"], "text_scaffold_tensors stage_idx=0")
        materialize_mock.assert_called_once()
        restored_scaffold = materialize_mock.call_args.kwargs["stage_state_scaffold"]
        self.assertEqual(restored_scaffold["stage_idx"], 0)
        self.assertEqual(restored_scaffold["start_idx"], 0)
        self.assertEqual(restored_scaffold["end_idx"], 35)
        self.assertEqual(restored_scaffold["save_dtype"], "float32")
        self.assertNotIn("hidden_size", restored_scaffold)
        self.assertNotIn("batch_size", restored_scaffold)
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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_text_stage_state",
            return_value=full_weight_bundle,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            with self.assertRaisesRegex(RuntimeError, "rank-local shard StageState"):
                load_stage_state_for_hybrid_rank(
                    manifest,
                    rank=1,
                    rank_stage=rank_stage,
                    device=torch.device("cpu"),
                    compute_dtype_arg="float32",
                )

        barrier_mock.assert_not_called()

    def test_direct_tp_stage_records_equal_weight_bytes(self) -> None:
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
        bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=35,
            tp_shard_rank=0,
        )

        def _equal_gather(output, local_bytes, **_kwargs):
            output[:] = [int(local_bytes), int(local_bytes)]

        with patch.object(hybrid_parallel_module.dist, "is_initialized", return_value=True), patch.object(
            hybrid_parallel_module.dist,
            "all_gather_object",
            side_effect=_equal_gather,
        ) as gather_mock:
            hybrid_parallel_module._record_tp_stage_weight_load_consistency(bundle, rank_stage)

        gather_mock.assert_called_once()
        self.assertEqual(
            bundle["_tp_stage_loaded_weight_tensor_bytes"],
            [bundle["_tp_stage_loaded_weight_tensor_bytes"][0]] * 2,
        )
        self.assertTrue(bundle["_tp_stage_loaded_weight_tensor_bytes_equal"])
        self.assertTrue(bundle["_tp_stage_loaded_weight_tensor_bytes_checked"])

    def test_direct_tp_stage_rejects_unequal_weight_bytes(self) -> None:
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
        bundle = _fake_tp_sharded_bundle(
            stage_idx=0,
            start_idx=0,
            end_idx=35,
            tp_shard_rank=0,
        )

        def _unequal_gather(output, local_bytes, **_kwargs):
            output[:] = [int(local_bytes), int(local_bytes) + 4]

        with patch.object(hybrid_parallel_module.dist, "is_initialized", return_value=True), patch.object(
            hybrid_parallel_module.dist,
            "all_gather_object",
            side_effect=_unequal_gather,
        ):
            with self.assertRaisesRegex(RuntimeError, "权重字节数不一致"):
                hybrid_parallel_module._record_tp_stage_weight_load_consistency(bundle, rank_stage)

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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_state",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_state_for_hybrid_rank(
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
