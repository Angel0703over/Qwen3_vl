from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import load_stage_bundle_for_hybrid_rank
from qwen3vl_tp_runtime.hexgen_core.schema import HybridRankContext, StageSpec, TextHybridManifest


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
        bundle_dir="<direct>",
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


class HybridDirectLoaderTest(unittest.TestCase):
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
        }
        local_bundle = {
            "save_dtype": "bfloat16",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [{"layer_idx": 0}],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.build_direct_stage_bundle",
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
            return_value=scaffold,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.materialize_direct_text_stage_bundle_from_scaffold",
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
        materialize_mock.assert_called_once_with(
            stage_bundle_scaffold=scaffold,
            runtime_config=manifest.runtime_config,
            compute_dtype=torch.bfloat16,
            tp_shard_rank=1,
            tp_shard_world_size=2,
        )
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.bfloat16)
        self.assertEqual(bundle["end_idx"], 17)

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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_runtime_only_text_generate_prompt_metadata",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
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
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.prepare_runtime_only_text_generate_prompt_metadata",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel.broadcast_object_cpu",
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
