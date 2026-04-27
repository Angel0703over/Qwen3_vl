from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import load_stage_bundle_for_rank
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec, TextPipelineManifest
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmFrontendSeed,
    compact_mm_runtime_shared,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    pack_mm_startup_transport,
    restore_mm_startup_transport,
    select_mm_startup_contract,
)


def _build_manifest(*, stage_ranges: list[tuple[int, int]], modality: str) -> TextPipelineManifest:
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
    return TextPipelineManifest(
        pipeline_type="text_generate",
        num_stages=len(stages),
        stage_ranges=stage_ranges,
        bundle_dir=None,
        stages=stages,
        boundaries=[],
        num_frames=0,
        save_dtype="float32",
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


class PipelineDirectLoaderTest(unittest.TestCase):
    def test_direct_detection_falls_back_without_manifest_property(self) -> None:
        from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import _all_stages_are_direct

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

        self.assertTrue(_all_stages_are_direct(legacy_manifest))
        self.assertFalse(_all_stages_are_direct(captured_manifest))

    def test_multimodal_direct_stage_build_skips_text_prompt_metadata(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], modality="multimodal")
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 1,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }
        startup_contract = _build_mm_startup_contract(num_stages=2)
        local_startup_contract = select_mm_startup_contract(startup_contract, local_stage_indices=[1])
        local_startup_meta, local_startup_tensors = pack_mm_startup_transport(local_startup_contract)

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.DirectStageBundleBuilder",
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.recv_object_cpu",
            return_value=local_startup_meta,
        ) as recv_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.recv_tensor_payload_cpu",
            return_value=local_startup_tensors,
        ) as recv_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_available",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_initialized",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_rank(
                manifest,
                rank=1,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_not_called()
        builder_cls.assert_not_called()
        recv_meta_mock.assert_called_once()
        self.assertEqual(
            recv_meta_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_meta stage_idx=1",
        )
        recv_mock.assert_called_once()
        self.assertEqual(
            recv_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_tensors stage_idx=1",
        )
        build_mock.assert_called_once()
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertFalse(build_mock.call_args.kwargs["mm_activate_frontend"])
        self.assertTrue(build_runtime_config["_mm_startup_contract_ready"])
        self.assertIn("_mm_startup_shared", build_runtime_config)
        self.assertTrue(
            torch.equal(build_runtime_config["_mm_startup_shared"]["input_ids"], torch.tensor([[1, 2, 3]], dtype=torch.long))
        )
        self.assertIn("_mm_startup_stage_handoffs", build_runtime_config)
        self.assertEqual(sorted(build_runtime_config["_mm_startup_stage_handoffs"]), [1])
        self.assertIn("_mm_startup_root_input", build_runtime_config)
        self.assertTrue(
            torch.equal(
                build_runtime_config["_mm_startup_root_input"],
                torch.zeros(1, 3, 4),
            )
        )
        self.assertIn("_mm_startup_stage_visuals", build_runtime_config)
        self.assertEqual(sorted(build_runtime_config["_mm_startup_stage_visuals"]), [1])
        self.assertNotIn("_mm_startup_boundaries", build_runtime_config)
        self.assertNotIn("_mm_startup_visual_pos_masks", build_runtime_config)
        self.assertNotIn("_mm_startup_deepstack_by_layer", build_runtime_config)
        self.assertNotIn("_mm_frontend_plan", build_runtime_config)
        self.assertNotIn("_mm_frontend_state", build_runtime_config)
        self.assertNotIn("_mm_frontend_seed", build_runtime_config)
        self.assertNotIn("_mm_frontend_meta", build_runtime_config)
        self.assertEqual(build_runtime_config["_mm_num_frames"], 2)
        self.assertEqual(build_runtime_config["_mm_frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["start_idx"], 18)

    def test_multimodal_stage_zero_marks_frontend_active_in_direct_build(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], modality="multimodal")
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [],
        }
        startup_contract = _build_mm_startup_contract(num_stages=2)
        builder_instance = MagicMock()
        builder_instance.__enter__.return_value = builder_instance
        builder_instance.export_mm_startup_transport.side_effect = [
            pack_mm_startup_transport(
                select_mm_startup_contract(startup_contract, local_stage_indices=[0]),
            ),
            pack_mm_startup_transport(
                select_mm_startup_contract(startup_contract, local_stage_indices=[1]),
            ),
        ]

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.DirectStageBundleBuilder",
            return_value=builder_instance,
        ) as builder_cls, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.send_object_cpu",
        ) as send_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.send_tensor_payload_cpu",
        ) as send_tensor_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_available",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_initialized",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_rank(
                manifest,
                rank=0,
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
        self.assertEqual(builder_instance.export_mm_startup_transport.call_count, 2)
        send_meta_mock.assert_called_once()
        self.assertEqual(
            send_meta_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_meta stage_idx=1",
        )
        send_tensor_mock.assert_called_once()
        self.assertEqual(
            send_tensor_mock.call_args.kwargs["label"],
            "multimodal_startup_contract_tensors stage_idx=1",
        )
        sent_payload = restore_mm_startup_transport(
            send_meta_mock.call_args.args[0],
            send_tensor_mock.call_args.args[0],
        )
        self.assertEqual(sorted(sent_payload["stage_handoffs"]), [1])
        self.assertEqual(sorted(sent_payload["stage_visuals"]), [1])
        self.assertTrue(build_mock.call_args.kwargs["mm_activate_frontend"])
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

    def test_rank_zero_seeds_runtime_only_prompt_metadata_before_local_build(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], modality="text")
        manifest.runtime_config["include_runtime_reference"] = False
        prompt_metadata = {
            "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        }
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 0,
            "start_idx": 0,
            "end_idx": 17,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_text_prompt_meta",
            return_value=prompt_metadata,
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.broadcast_tensor_payload_cpu",
            side_effect=lambda payload, **_kwargs: payload,
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_available",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_initialized",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_rank(
                manifest,
                rank=0,
                device=torch.device("cpu"),
                compute_dtype_arg="float32",
            )

        prepare_meta_mock.assert_called_once_with(manifest.runtime_config)
        self.assertTrue(torch.equal(bcast_mock.call_args.args[0]["input_ids"], prompt_metadata["input_ids"]))
        self.assertTrue(
            torch.equal(
                bcast_mock.call_args.args[0]["attention_mask"],
                prompt_metadata["attention_mask"],
            )
        )
        build_runtime_config = build_mock.call_args.kwargs["runtime_config"]
        self.assertTrue(torch.equal(build_runtime_config["_runtime_only_input_ids"], prompt_metadata["input_ids"]))
        self.assertTrue(
            torch.equal(build_runtime_config["_runtime_only_attention_mask"], prompt_metadata["attention_mask"])
        )
        barrier_mock.assert_called_once()
        self.assertEqual(compute_dtype, torch.float32)
        self.assertEqual(bundle["end_idx"], 17)

    def test_nonzero_rank_restores_compact_prompt_metadata_before_local_build(self) -> None:
        manifest = _build_manifest(stage_ranges=[(0, 17), (18, 35)], modality="text")
        manifest.runtime_config["include_runtime_reference"] = False
        direct_bundle = {
            "save_dtype": "float32",
            "stage_idx": 1,
            "start_idx": 18,
            "end_idx": 35,
            "layers": [],
        }

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_text_prompt_meta",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.broadcast_tensor_payload_cpu",
            return_value={"input_ids": torch.tensor([[7, 8, 9]], dtype=torch.int64)},
        ) as bcast_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.build_direct_stage_bundle",
            return_value=direct_bundle,
        ) as build_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_available",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.is_initialized",
            return_value=True,
        ), patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.dist.barrier",
        ) as barrier_mock:
            bundle, compute_dtype = load_stage_bundle_for_rank(
                manifest,
                rank=1,
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
