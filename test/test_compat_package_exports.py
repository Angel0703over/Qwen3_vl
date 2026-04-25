from __future__ import annotations

import unittest

import qwen3vl_tp_runtime.hexgen_core as core_pkg
import qwen3vl_tp_runtime.hexgen_core.modules as core_modules_pkg
import qwen3vl_tp_runtime.models as models_pkg
import qwen3vl_tp_runtime.models.qwen3vl as qwen3vl_pkg
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
        self.assertIn("load_bundle", models_pkg.__all__)

    def test_hexgen_core_modules_keeps_main_and_legacy_exports(self) -> None:
        self.assertIs(core_modules_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_modules_pkg.prepare_text_generate_pipeline, prepare_text_generate_pipeline)
        self.assertIn("run_text_generate_pipeline_rank", core_modules_pkg.__all__)
        self.assertIn("prepare_text_generate_pipeline", core_modules_pkg.__all__)

    def test_hexgen_core_root_keeps_lazy_compat_exports(self) -> None:
        self.assertIs(core_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_pkg.prepare_text_generate_hybrid, prepare_text_generate_hybrid)
        self.assertIn("run_text_generate_pipeline_rank", core_pkg.__all__)
        self.assertIn("prepare_text_generate_hybrid", core_pkg.__all__)


if __name__ == "__main__":
    unittest.main()
