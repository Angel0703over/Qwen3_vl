from __future__ import annotations

import unittest

import qwen3vl_tp_runtime.hexgen_core as core_pkg
import qwen3vl_tp_runtime.hexgen_core.modules as core_modules_pkg
import qwen3vl_tp_runtime.models as models_pkg
import qwen3vl_tp_runtime.models.qwen3vl as qwen3vl_pkg
import qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel as hybrid_module
import qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel as pipeline_module
import qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel as tensor_module
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import prepare_text_generate_hybrid
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import (
    prepare_text_generate_pipeline,
    run_text_generate_pipeline_rank,
)


class CompatPackageExportsTest(unittest.TestCase):
    def test_models_package_keeps_lazy_main_and_legacy_exports(self) -> None:
        self.assertIs(models_pkg.build_direct_pipeline_manifest, qwen3vl_pkg.build_direct_pipeline_manifest)
        self.assertIs(models_pkg.load_bundle, qwen3vl_pkg.load_bundle)
        self.assertIn("build_direct_pipeline_manifest", models_pkg.__all__)
        self.assertIn("build_direct_pipeline_manifest", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("load_bundle", models_pkg.__all__)
        self.assertIn("load_bundle", models_pkg.LEGACY_CAPTURE_EXPORTS)

    def test_hexgen_core_modules_keeps_main_and_legacy_exports(self) -> None:
        self.assertIs(core_modules_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_modules_pkg.prepare_text_generate_pipeline, prepare_text_generate_pipeline)
        self.assertIn("run_text_generate_pipeline_rank", core_modules_pkg.__all__)
        self.assertIn("run_text_generate_pipeline_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("prepare_text_generate_pipeline", core_modules_pkg.__all__)
        self.assertIn("prepare_text_generate_pipeline", core_modules_pkg.LEGACY_REPLAY_EXPORTS)

    def test_hexgen_core_root_keeps_lazy_compat_exports(self) -> None:
        self.assertIs(core_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_pkg.prepare_text_generate_hybrid, prepare_text_generate_hybrid)
        self.assertIn("run_text_generate_pipeline_rank", core_pkg.__all__)
        self.assertIn("run_text_generate_pipeline_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("prepare_text_generate_hybrid", core_pkg.__all__)
        self.assertIn("prepare_text_generate_hybrid", core_pkg.LEGACY_REPLAY_EXPORTS)

    def test_concrete_runtime_modules_separate_direct_and_replay_exports(self) -> None:
        self.assertIn("run_text_generate_pipeline_rank", pipeline_module.__all__)
        self.assertIn("run_text_generate_pipeline_rank", pipeline_module.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("prepare_text_generate_pipeline", pipeline_module.__all__)
        self.assertIn("prepare_text_generate_pipeline", pipeline_module.LEGACY_REPLAY_EXPORTS)

        self.assertIn("run_text_hybrid_rank", hybrid_module.__all__)
        self.assertIn("run_text_hybrid_rank", hybrid_module.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("prepare_text_generate_hybrid", hybrid_module.__all__)
        self.assertIn("prepare_text_generate_hybrid", hybrid_module.LEGACY_REPLAY_EXPORTS)

        self.assertEqual(tensor_module.__all__, [])
        self.assertIn("run_text_tensor_parallel_rank", tensor_module.DEBUG_REPLAY_EXPORTS)


if __name__ == "__main__":
    unittest.main()
