from __future__ import annotations

import unittest

from qwen3vl_tp_runtime.hexgen_core.modules.tp_debug import TpDebugConfig


class TpDebugConfigTest(unittest.TestCase):
    def test_default_config_is_disabled(self) -> None:
        config = TpDebugConfig()

        self.assertFalse(config.debug_mode)
        self.assertFalse(config.needs_direct_output)
        self.assertFalse(config.needs_layer_trace)
        self.assertEqual(
            config.to_summary_fields(),
            {
                "debug_mode": False,
                "compare_direct": False,
                "trace_layers": False,
                "dump_layer": None,
                "dump_topk": 5,
            },
        )

    def test_compare_direct_only_requires_direct_output(self) -> None:
        config = TpDebugConfig(compare_direct=True)

        self.assertTrue(config.debug_mode)
        self.assertTrue(config.needs_direct_output)
        self.assertFalse(config.needs_layer_trace)

    def test_trace_or_dump_enables_layer_trace(self) -> None:
        trace_config = TpDebugConfig(trace_layers=True)
        dump_config = TpDebugConfig(dump_layer=7, dump_topk=3)

        self.assertTrue(trace_config.needs_direct_output)
        self.assertTrue(trace_config.needs_layer_trace)
        self.assertTrue(dump_config.needs_direct_output)
        self.assertTrue(dump_config.needs_layer_trace)
        self.assertEqual(dump_config.to_summary_fields()["dump_topk"], 3)


if __name__ == "__main__":
    unittest.main()
