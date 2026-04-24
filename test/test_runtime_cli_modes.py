from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr

from qwen3vl_tp_runtime.scripts.runtime import build_parser
from qwen3vl_tp_runtime.scripts.runtime_cli import (
    _debug_path_warnings,
    _require_debug_path_opt_in,
)


class RuntimeCliModesTest(unittest.TestCase):
    def test_parser_help_marks_manifest_replay_as_debug_only(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("Recommended main path", help_text)
        self.assertIn("--allow-debug-paths", help_text)
        self.assertIn("--manifest-path", help_text)
        self.assertIn("debug/regression workflows", help_text)
        self.assertNotIn("--action", help_text)

    def test_manifest_path_run_emits_debug_warning(self) -> None:
        args = argparse.Namespace(backend="hybrid", manifest_path="/tmp/replay_manifest.pt")

        warnings = _debug_path_warnings(args)

        self.assertEqual(len(warnings), 1)
        self.assertIn("backend=hybrid", warnings[0])
        self.assertIn("--manifest-path", warnings[0])
        self.assertIn("调试/回放路径", warnings[0])

    def test_run_main_path_emits_no_warning(self) -> None:
        args = argparse.Namespace(backend="hybrid", manifest_path=None)

        warnings = _debug_path_warnings(args)

        self.assertEqual(warnings, [])

    def test_manifest_path_run_requires_debug_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "hybrid",
                "--manifest-path",
                "/tmp/replay_manifest.pt",
            ]
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                _require_debug_path_opt_in(parser, args)

        self.assertIn("backend=hybrid --manifest-path", stderr.getvalue())
        self.assertIn("--allow-debug-paths", stderr.getvalue())

    def test_removed_action_flag_is_rejected(self) -> None:
        parser = build_parser()
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                parser.parse_args(
                    [
                        "--modality",
                        "text",
                        "--mode",
                        "generate",
                        "--backend",
                        "pp",
                        "--action",
                        "run",
                    ]
                )

        self.assertIn("--action", stderr.getvalue())

    def test_allow_debug_paths_flag_allows_manifest_replay_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "hybrid",
                "--manifest-path",
                "/tmp/replay_manifest.pt",
                "--allow-debug-paths",
            ]
        )

        _require_debug_path_opt_in(parser, args)


if __name__ == "__main__":
    unittest.main()
