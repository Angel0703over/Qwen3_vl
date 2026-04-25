from __future__ import annotations

import unittest

from qwen3vl_tp_runtime.hexgen_core.schema import TextPipelineManifest
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    build_direct_hybrid_manifest,
    build_direct_pipeline_manifest,
)


class SchemaDirectManifestTest(unittest.TestCase):
    def test_build_direct_pipeline_manifest_uses_none_bundle_dir(self) -> None:
        manifest = build_direct_pipeline_manifest(
            modality="text",
            mode="generate",
            stage_ranges=[(0, 17), (18, 35)],
            model_path="/tmp/fake-model",
            save_dtype="float32",
            prompt="hello",
            max_new_tokens=4,
            include_runtime_reference=False,
        )

        self.assertIsNone(manifest.bundle_dir)
        self.assertTrue(manifest.is_direct)
        self.assertTrue(all(stage.is_direct for stage in manifest.stages))

    def test_build_direct_hybrid_manifest_keeps_direct_schema(self) -> None:
        manifest = build_direct_hybrid_manifest(
            modality="text",
            mode="generate",
            stage_ranges=[(0, 17), (18, 35)],
            tp_degrees=[2, 1],
            model_path="/tmp/fake-model",
            save_dtype="float32",
            prompt="hello",
            max_new_tokens=4,
            include_runtime_reference=False,
        )

        self.assertIsNone(manifest.bundle_dir)
        self.assertTrue(manifest.is_direct)
        self.assertTrue(all(stage.is_direct for stage in manifest.stages))

    def test_pipeline_manifest_from_dict_defaults_missing_bundle_dir_to_none(self) -> None:
        manifest = TextPipelineManifest.from_dict(
            {
                "pipeline_type": "text_generate",
                "num_stages": 1,
                "stage_ranges": [[0, 17]],
                "stages": [
                    {
                        "stage_idx": 0,
                        "start_idx": 0,
                        "end_idx": 17,
                        "num_layers": 18,
                        "save_dtype": "float32",
                        "bundle_path": None,
                    }
                ],
                "boundaries": [],
                "num_frames": 0,
                "save_dtype": "float32",
                "runtime_config": {"model_path": "/tmp/fake-model"},
            }
        )

        self.assertIsNone(manifest.bundle_dir)
        self.assertTrue(manifest.is_direct)


if __name__ == "__main__":
    unittest.main()
