from __future__ import annotations

import unittest

import qwen3vl_tp_runtime.models.qwen3vl as qwen3vl
import qwen3vl_tp_runtime.models.qwen3vl.capture as capture_pkg
from qwen3vl_tp_runtime.hexgen_core.schema import StageState
from qwen3vl_tp_runtime.models.qwen3vl.capture import (
    capture_text_prefill_bundle as capture_text_prefill_bundle_impl,
    load_bundle as load_bundle_impl,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    DirectStageBundleBuilder,
    DirectStageStateBuilder,
    StageStateLoader,
    build_direct_tp_manifest as build_direct_tp_manifest_impl,
    build_direct_stage_state as build_direct_stage_state_impl,
    materialize_text_stage_state as materialize_text_stage_state_impl,
)


class Qwen3vlExportsTest(unittest.TestCase):
    def test_root_package_exports_stage_state_compat_names(self) -> None:
        self.assertIs(qwen3vl.StageState, StageState)
        self.assertIs(qwen3vl.DirectStageStateBuilder, DirectStageStateBuilder)
        self.assertIs(qwen3vl.DirectStageStateBuilder, DirectStageBundleBuilder)
        self.assertIs(qwen3vl.StageStateLoader, StageStateLoader)
        self.assertIs(qwen3vl.build_direct_tp_manifest, build_direct_tp_manifest_impl)
        self.assertIs(qwen3vl.build_direct_stage_state, build_direct_stage_state_impl)
        self.assertIs(qwen3vl.materialize_text_stage_state, materialize_text_stage_state_impl)
        self.assertIn("StageState", qwen3vl.__all__)
        self.assertIn("DirectStageStateBuilder", qwen3vl.__all__)
        self.assertIn("StageStateLoader", qwen3vl.__all__)
        self.assertIn("build_direct_tp_manifest", qwen3vl.__all__)
        self.assertIn("build_direct_stage_state", qwen3vl.__all__)
        self.assertIn("materialize_text_stage_state", qwen3vl.__all__)
        self.assertNotIn("DirectStageBundleBuilder", qwen3vl.__all__)
        self.assertNotIn("build_direct_stage_bundle", qwen3vl.__all__)
        self.assertIn("DirectStageBundleBuilder", qwen3vl.LEGACY_STAGE_BUNDLE_EXPORTS)
        self.assertIn("build_direct_stage_bundle", qwen3vl.LEGACY_STAGE_BUNDLE_EXPORTS)

    def test_root_package_keeps_legacy_capture_exports_via_lazy_compat(self) -> None:
        self.assertIs(qwen3vl.load_bundle, load_bundle_impl)
        self.assertIs(qwen3vl.capture_text_prefill_bundle, capture_text_prefill_bundle_impl)
        self.assertNotIn("load_bundle", qwen3vl.__all__)
        self.assertNotIn("capture_text_prefill_bundle", qwen3vl.__all__)
        self.assertIn("load_bundle", qwen3vl.LEGACY_CAPTURE_EXPORTS)
        self.assertIn("capture_text_prefill_bundle", qwen3vl.LEGACY_CAPTURE_EXPORTS)
        self.assertIn("load_bundle", capture_pkg.LEGACY_CAPTURE_EXPORTS)
        self.assertIn("capture_text_prefill_bundle", capture_pkg.LEGACY_CAPTURE_EXPORTS)


if __name__ == "__main__":
    unittest.main()
