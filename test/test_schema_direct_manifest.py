from __future__ import annotations

import unittest

from qwen3vl_tp_runtime.hexgen_core.schema import TextHybridManifest, TextPipelineManifest
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
        payload = manifest.to_dict()
        self.assertNotIn("bundle_dir", payload)
        self.assertNotIn("replay", payload)
        self.assertNotIn("bundle_path", payload["stages"][0])
        self.assertNotIn("replay", payload["stages"][0])

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
        payload = manifest.to_dict()
        self.assertNotIn("bundle_dir", payload)
        self.assertNotIn("replay", payload)
        self.assertNotIn("bundle_path", payload["stages"][0])
        self.assertNotIn("replay", payload["stages"][0])

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

    def test_pipeline_manifest_from_legacy_replay_fields_normalizes_schema(self) -> None:
        manifest = TextPipelineManifest.from_dict(
            {
                "pipeline_type": "text_generate",
                "num_stages": 1,
                "stage_ranges": [[0, 17]],
                "bundle_dir": "/tmp/bundles",
                "stages": [
                    {
                        "stage_idx": 0,
                        "start_idx": 0,
                        "end_idx": 17,
                        "num_layers": 18,
                        "save_dtype": "float32",
                        "bundle_path": "/tmp/bundles/stage_00.pt",
                    }
                ],
                "boundaries": [],
                "num_frames": 0,
                "save_dtype": "float32",
                "runtime_config": {"model_path": "/tmp/fake-model"},
            }
        )

        self.assertFalse(manifest.is_direct)
        self.assertEqual(manifest.bundle_dir, "/tmp/bundles")
        self.assertEqual(manifest.replay_bundle_dir, "/tmp/bundles")
        self.assertEqual(manifest.stages[0].bundle_path, "/tmp/bundles/stage_00.pt")
        self.assertEqual(manifest.stages[0].replay_bundle_path, "/tmp/bundles/stage_00.pt")

        payload = manifest.to_dict()
        self.assertNotIn("bundle_dir", payload)
        self.assertEqual(payload["replay"], {"bundle_dir": "/tmp/bundles"})
        self.assertNotIn("bundle_path", payload["stages"][0])
        self.assertEqual(payload["stages"][0]["replay"], {"bundle_path": "/tmp/bundles/stage_00.pt"})

    def test_hybrid_manifest_from_replay_schema_keeps_legacy_properties(self) -> None:
        manifest = TextHybridManifest.from_dict(
            {
                "runtime": "text_generate_hybrid",
                "tp_degrees": [1],
                "stage_rank_groups": [[0]],
                "pp_rank_groups": [[0]],
                "world_size": 1,
                "num_stages": 1,
                "send_list": [[]],
                "recv_list": [[]],
                "send_empty_list": [[]],
                "recv_empty_list": [[]],
                "pipeline_type": "text_generate",
                "stage_ranges": [[0, 17]],
                "replay": {"bundle_dir": "/tmp/hybrid-bundles"},
                "stages": [
                    {
                        "stage_idx": 0,
                        "start_idx": 0,
                        "end_idx": 17,
                        "num_layers": 18,
                        "save_dtype": "float32",
                        "replay": {"bundle_path": "/tmp/hybrid-bundles/stage_00.pt"},
                    }
                ],
                "boundaries": [],
                "num_frames": 0,
                "save_dtype": "float32",
                "runtime_config": {"model_path": "/tmp/fake-model"},
            }
        )

        self.assertFalse(manifest.is_direct)
        self.assertEqual(manifest.bundle_dir, "/tmp/hybrid-bundles")
        self.assertEqual(manifest.stages[0].bundle_path, "/tmp/hybrid-bundles/stage_00.pt")


if __name__ == "__main__":
    unittest.main()
