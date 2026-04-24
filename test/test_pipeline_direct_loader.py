from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import load_stage_bundle_for_rank
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec, TextPipelineManifest


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
        bundle_dir="<direct>",
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


class PipelineDirectLoaderTest(unittest.TestCase):
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
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_runtime_only_text_generate_prompt_metadata",
            return_value=prompt_metadata,
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.broadcast_object_cpu",
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
        self.assertEqual(
            bcast_mock.call_args.args[0],
            {
                "input_ids_list": [7, 8, 9],
                "attention_mask_list": [1, 1, 0],
            },
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
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.prepare_runtime_only_text_generate_prompt_metadata",
        ) as prepare_meta_mock, patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel.broadcast_object_cpu",
            return_value={"input_ids_list": [7, 8, 9]},
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
