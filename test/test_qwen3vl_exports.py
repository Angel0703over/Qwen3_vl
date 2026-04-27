from __future__ import annotations

import unittest

import qwen3vl_tp_runtime.models.qwen3vl as qwen3vl
import qwen3vl_tp_runtime.models.qwen3vl.capture as capture_pkg
from qwen3vl_tp_runtime.models.qwen3vl.capture import (
    capture_text_prefill_bundle as capture_text_prefill_bundle_impl,
    load_bundle as load_bundle_impl,
)


class Qwen3vlExportsTest(unittest.TestCase):
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
