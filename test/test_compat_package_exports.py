from __future__ import annotations

import unittest

import qwen3vl_tp_runtime.hexgen_core as core_pkg
import qwen3vl_tp_runtime.hexgen_core.modules as core_modules_pkg
import qwen3vl_tp_runtime.models as models_pkg
import qwen3vl_tp_runtime.models.qwen3vl as qwen3vl_pkg
import qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel as hybrid_module
import qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel as pipeline_module
import qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel as tensor_module
from qwen3vl_tp_runtime.hexgen_core.schema import StageState
from qwen3vl_tp_runtime.hexgen_core.transport import StageCommunicator, StageHandoffTransport
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import prepare_text_generate_hybrid
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import (
    prepare_text_generate_pipeline,
    run_text_generate_pipeline_rank,
)


class CompatPackageExportsTest(unittest.TestCase):
    def test_models_package_keeps_lazy_main_and_legacy_exports(self) -> None:
        self.assertIs(models_pkg.StageState, StageState)
        self.assertIs(models_pkg.build_direct_stage_state, qwen3vl_pkg.build_direct_stage_state)
        self.assertIs(models_pkg.StageStateLoader, qwen3vl_pkg.StageStateLoader)
        self.assertIs(models_pkg.build_direct_pipeline_manifest, qwen3vl_pkg.build_direct_pipeline_manifest)
        self.assertIs(models_pkg.build_direct_tp_manifest, qwen3vl_pkg.build_direct_tp_manifest)
        self.assertIs(models_pkg.load_bundle, qwen3vl_pkg.load_bundle)
        self.assertIn("StageState", models_pkg.__all__)
        self.assertIn("build_direct_stage_state", models_pkg.__all__)
        self.assertIn("StageStateLoader", models_pkg.__all__)
        self.assertIn("build_direct_pipeline_manifest", models_pkg.__all__)
        self.assertIn("build_direct_tp_manifest", models_pkg.__all__)
        self.assertIn("StageState", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("build_direct_stage_state", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("StageStateLoader", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("build_direct_pipeline_manifest", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("build_direct_tp_manifest", models_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("DirectStageBundleBuilder", models_pkg.__all__)
        self.assertNotIn("build_direct_stage_bundle", models_pkg.__all__)
        self.assertIn("DirectStageBundleBuilder", models_pkg.LEGACY_STAGE_BUNDLE_EXPORTS)
        self.assertIn("build_direct_stage_bundle", models_pkg.LEGACY_STAGE_BUNDLE_EXPORTS)
        self.assertNotIn("load_bundle", models_pkg.__all__)
        self.assertIn("load_bundle", models_pkg.LEGACY_CAPTURE_EXPORTS)

    def test_hexgen_core_modules_keeps_main_and_legacy_exports(self) -> None:
        self.assertIs(core_modules_pkg.StageState, StageState)
        self.assertIs(core_modules_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_modules_pkg.prepare_text_generate_pipeline, prepare_text_generate_pipeline)
        self.assertIs(core_modules_pkg.load_stage_state_for_rank, pipeline_module.load_stage_state_for_rank)
        self.assertIs(
            core_modules_pkg.load_stage_state_for_hybrid_rank,
            hybrid_module.load_stage_state_for_hybrid_rank,
        )
        self.assertIs(core_modules_pkg.load_stage_state_for_tp_rank, tensor_module.load_stage_state_for_tp_rank)
        self.assertIs(core_modules_pkg.load_tp_manifest, tensor_module.load_tp_manifest)
        self.assertIs(core_modules_pkg.run_tensor_parallel_rank, tensor_module.run_tensor_parallel_rank)
        self.assertIn("StageState", core_modules_pkg.__all__)
        self.assertIn("load_stage_state_for_rank", core_modules_pkg.__all__)
        self.assertIn("load_stage_state_for_hybrid_rank", core_modules_pkg.__all__)
        self.assertIn("load_stage_state_for_tp_rank", core_modules_pkg.__all__)
        self.assertIn("load_tp_manifest", core_modules_pkg.__all__)
        self.assertIn("run_tensor_parallel_rank", core_modules_pkg.__all__)
        self.assertIn("run_text_generate_pipeline_rank", core_modules_pkg.__all__)
        self.assertIn("StageState", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_hybrid_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_tp_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_tp_manifest", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("TensorParallelManifest", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_tensor_parallel_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_text_generate_pipeline_rank", core_modules_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("load_stage_bundle_for_rank", core_modules_pkg.__all__)
        self.assertIn("load_stage_bundle_for_rank", core_modules_pkg.LEGACY_REPLAY_EXPORTS)
        self.assertNotIn("prepare_text_generate_pipeline", core_modules_pkg.__all__)
        self.assertIn("prepare_text_generate_pipeline", core_modules_pkg.LEGACY_REPLAY_EXPORTS)

    def test_hexgen_core_root_keeps_lazy_compat_exports(self) -> None:
        self.assertIs(core_pkg.StageState, StageState)
        self.assertIs(core_pkg.run_text_generate_pipeline_rank, run_text_generate_pipeline_rank)
        self.assertIs(core_pkg.prepare_text_generate_hybrid, prepare_text_generate_hybrid)
        self.assertIs(core_pkg.load_stage_state_by_index, pipeline_module.load_stage_state_by_index)
        self.assertIs(core_pkg.load_stage_state_for_rank, pipeline_module.load_stage_state_for_rank)
        self.assertIs(core_pkg.load_stage_state_for_hybrid_rank, hybrid_module.load_stage_state_for_hybrid_rank)
        self.assertIs(core_pkg.load_stage_state_for_tp_rank, tensor_module.load_stage_state_for_tp_rank)
        self.assertIs(core_pkg.load_tp_manifest, tensor_module.load_tp_manifest)
        self.assertIs(core_pkg.run_tensor_parallel_rank, tensor_module.run_tensor_parallel_rank)
        self.assertIs(core_pkg.StageCommunicator, StageCommunicator)
        self.assertTrue(issubclass(core_pkg.StageHandoffTransport, StageCommunicator))
        self.assertTrue(issubclass(StageHandoffTransport, StageCommunicator))
        self.assertIn("StageState", core_pkg.__all__)
        self.assertIn("StageStateView", core_pkg.__all__)
        self.assertIn("as_stage_state_view", core_pkg.__all__)
        self.assertIn("build_stage_state", core_pkg.__all__)
        self.assertIn("load_stage_state_by_index", core_pkg.__all__)
        self.assertIn("load_stage_state_for_rank", core_pkg.__all__)
        self.assertIn("load_stage_state_for_hybrid_rank", core_pkg.__all__)
        self.assertIn("load_stage_state_for_tp_rank", core_pkg.__all__)
        self.assertIn("load_tp_manifest", core_pkg.__all__)
        self.assertIn("TensorParallelManifest", core_pkg.__all__)
        self.assertIn("run_tensor_parallel_rank", core_pkg.__all__)
        self.assertIn("run_text_generate_pipeline_rank", core_pkg.__all__)
        self.assertIn("StageCommunicator", core_pkg.__all__)
        self.assertIn("StageState", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_hybrid_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_tp_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_tp_manifest", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("TensorParallelManifest", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_tensor_parallel_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_text_generate_pipeline_rank", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("StageCommunicator", core_pkg.DIRECT_RUNTIME_EXPORTS)
        self.assertNotIn("StageBundleView", core_pkg.__all__)
        self.assertNotIn("as_stage_bundle_view", core_pkg.__all__)
        self.assertNotIn("build_stage_bundle", core_pkg.__all__)
        self.assertNotIn("load_stage_bundle_for_rank", core_pkg.__all__)
        self.assertIn("StageBundleView", core_pkg.LEGACY_REPLAY_EXPORTS)
        self.assertIn("build_stage_bundle", core_pkg.LEGACY_REPLAY_EXPORTS)
        self.assertIn("load_stage_bundle_for_rank", core_pkg.LEGACY_REPLAY_EXPORTS)
        self.assertNotIn("prepare_text_generate_hybrid", core_pkg.__all__)
        self.assertIn("prepare_text_generate_hybrid", core_pkg.LEGACY_REPLAY_EXPORTS)

    def test_concrete_runtime_modules_separate_direct_and_replay_exports(self) -> None:
        self.assertIn("load_stage_state_by_index", pipeline_module.__all__)
        self.assertIn("load_stage_state_for_rank", pipeline_module.__all__)
        self.assertIn("load_stage_state_by_index", pipeline_module.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("load_stage_state_for_rank", pipeline_module.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_text_generate_pipeline_rank", pipeline_module.__all__)
        self.assertIn("run_text_generate_pipeline_rank", pipeline_module.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("StageRunner", pipeline_module.__all__)
        self.assertIn("GenerateWorker", pipeline_module.__all__)
        self.assertIn("DecodeWorker", pipeline_module.__all__)
        self.assertNotIn("load_stage_bundle_by_index", pipeline_module.__all__)
        self.assertNotIn("load_stage_bundle_for_rank", pipeline_module.__all__)
        self.assertIn("load_stage_bundle_by_index", pipeline_module.LEGACY_REPLAY_EXPORTS)
        self.assertIn("load_stage_bundle_for_rank", pipeline_module.LEGACY_REPLAY_EXPORTS)
        self.assertNotIn("prepare_text_generate_pipeline", pipeline_module.__all__)
        self.assertIn("prepare_text_generate_pipeline", pipeline_module.LEGACY_REPLAY_EXPORTS)

        self.assertIn("load_stage_state_for_hybrid_rank", hybrid_module.__all__)
        self.assertIn("load_stage_state_for_hybrid_rank", hybrid_module.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("run_text_hybrid_rank", hybrid_module.__all__)
        self.assertIn("run_text_hybrid_rank", hybrid_module.DIRECT_RUNTIME_EXPORTS)
        self.assertIn("StageRunner", hybrid_module.__all__)
        self.assertIn("GenerateWorker", hybrid_module.__all__)
        self.assertIn("DecodeWorker", hybrid_module.__all__)
        self.assertNotIn("prepare_text_generate_hybrid", hybrid_module.__all__)
        self.assertIn("prepare_text_generate_hybrid", hybrid_module.LEGACY_REPLAY_EXPORTS)

        self.assertIn("TensorParallelManifest", tensor_module.__all__)
        self.assertIn("load_tp_manifest", tensor_module.__all__)
        self.assertIn("load_stage_state_for_tp_rank", tensor_module.__all__)
        self.assertIn("StageRunner", tensor_module.__all__)
        self.assertIn("GenerateWorker", tensor_module.__all__)
        self.assertIn("DecodeWorker", tensor_module.__all__)
        self.assertIn("TensorParallelRunner", tensor_module.__all__)
        self.assertIn("run_tensor_parallel_rank", tensor_module.__all__)
        self.assertIn("run_stage_state_tp", tensor_module.__all__)
        self.assertTrue(issubclass(tensor_module.TensorParallelRunner, tensor_module.GenerateWorker))
        self.assertIn("run_text_tensor_parallel_rank", tensor_module.DEBUG_REPLAY_EXPORTS)


if __name__ == "__main__":
    unittest.main()
