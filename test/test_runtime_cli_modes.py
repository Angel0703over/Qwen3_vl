from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import qwen3vl_tp_runtime.scripts.runtime as runtime_script
from qwen3vl_tp_runtime.scripts.runtime import build_parser
from qwen3vl_tp_runtime.scripts.runtime_cli import (
    _build_direct_manifest_kwargs,
    _debug_path_warnings,
    _load_hybrid_manifest_for_args,
    _load_pipeline_manifest_for_args,
    _reject_unsupported_debug_transport_backend,
    _reject_unsupported_generate_debug_flags,
    _require_debug_path_opt_in,
)


class RuntimeCliModesTest(unittest.TestCase):
    def test_parser_help_marks_manifest_replay_as_debug_only(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("Recommended main path", help_text)
        self.assertIn("--allow-debug-paths", help_text)
        self.assertIn("--manifest-path", help_text)
        self.assertIn("debug/regression workflows", help_text)
        self.assertIn("non-generate runs", help_text)
        self.assertNotIn("--action", help_text)

    def test_manifest_path_run_emits_debug_warning(self) -> None:
        args = argparse.Namespace(backend="hybrid", manifest_path="/tmp/replay_manifest.pt")

        warnings = _debug_path_warnings(args)

        self.assertEqual(len(warnings), 1)
        self.assertIn("backend=hybrid", warnings[0])
        self.assertIn("--manifest-path", warnings[0])
        self.assertIn("调试/回放路径", warnings[0])

    def test_runtime_main_import_surface_excludes_manifest_replay_loaders(self) -> None:
        self.assertFalse(hasattr(runtime_script, "load_pipeline_manifest"))
        self.assertFalse(hasattr(runtime_script, "load_hybrid_manifest"))

    def test_manifest_path_uses_debug_pipeline_replay_helper(self) -> None:
        args = argparse.Namespace(manifest_path="/tmp/replay_manifest.pt")
        sentinel = object()

        with patch(
            "qwen3vl_tp_runtime.scripts.runtime_cli.load_debug_pipeline_manifest",
            return_value=sentinel,
        ) as load_debug_mock:
            manifest = _load_pipeline_manifest_for_args(args)

        self.assertIs(manifest, sentinel)
        load_debug_mock.assert_called_once_with("/tmp/replay_manifest.pt")

    def test_manifest_path_uses_debug_hybrid_replay_helper(self) -> None:
        args = argparse.Namespace(manifest_path="/tmp/replay_manifest.pt")
        sentinel = object()

        with patch(
            "qwen3vl_tp_runtime.scripts.runtime_cli.load_debug_hybrid_manifest",
            return_value=sentinel,
        ) as load_debug_mock:
            manifest = _load_hybrid_manifest_for_args(args, backend="hybrid")

        self.assertIs(manifest, sentinel)
        load_debug_mock.assert_called_once_with("/tmp/replay_manifest.pt")

    def test_backend_tp_main_path_builds_direct_hybrid_manifest(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "tp",
                "--stage-ranges",
                "0:35",
                "--tp-degrees",
                "2",
            ]
        )
        sentinel = object()

        with patch.object(
            runtime_script,
            "build_direct_hybrid_manifest",
            return_value=sentinel,
        ) as build_mock:
            manifest = _load_hybrid_manifest_for_args(args, backend="tp")

        self.assertIs(manifest, sentinel)
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.kwargs["backend"], "tp")
        self.assertEqual(build_mock.call_args.kwargs["stage_ranges"], [(0, 35)])
        self.assertEqual(build_mock.call_args.kwargs["tp_degrees"], [2])
        self.assertFalse(build_mock.call_args.kwargs["include_runtime_reference"])

    def test_run_main_path_emits_no_warning(self) -> None:
        args = argparse.Namespace(backend="hybrid", manifest_path=None)

        warnings = _debug_path_warnings(args)

        self.assertEqual(warnings, [])

    def test_compare_direct_run_emits_debug_warning(self) -> None:
        args = argparse.Namespace(
            backend="hybrid",
            manifest_path=None,
            compare_direct=True,
            trace_layers=False,
            dump_layer=None,
        )

        warnings = _debug_path_warnings(args)

        self.assertEqual(len(warnings), 1)
        self.assertIn("backend=hybrid", warnings[0])
        self.assertIn("--compare-direct", warnings[0])
        self.assertIn("reference/trace transport", warnings[0])

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

    def test_compare_direct_run_requires_debug_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "prefill",
                "--backend",
                "hybrid",
                "--compare-direct",
            ]
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                _require_debug_path_opt_in(parser, args)

        self.assertIn("backend=hybrid --compare-direct", stderr.getvalue())
        self.assertIn("--allow-debug-paths", stderr.getvalue())

    def test_trace_layers_and_dump_layer_require_debug_opt_in(self) -> None:
        parser = build_parser()
        cases = [
            ("--trace-layers",),
            ("--dump-layer", "3"),
        ]

        for extra_args in cases:
            with self.subTest(extra_args=extra_args):
                args = parser.parse_args(
                    [
                        "--modality",
                        "text",
                        "--mode",
                        "prefill",
                        "--backend",
                        "hybrid",
                        *extra_args,
                    ]
                )
                stderr = io.StringIO()

                with self.assertRaises(SystemExit):
                    with redirect_stderr(stderr):
                        _require_debug_path_opt_in(parser, args)

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

    def test_allow_debug_paths_flag_allows_compare_direct_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "prefill",
                "--backend",
                "hybrid",
                "--compare-direct",
                "--allow-debug-paths",
            ]
        )

        _require_debug_path_opt_in(parser, args)

    def test_pp_compare_direct_is_rejected_even_with_debug_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "prefill",
                "--backend",
                "pp",
                "--compare-direct",
                "--allow-debug-paths",
            ]
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                _reject_unsupported_debug_transport_backend(parser, args)

        self.assertIn("backend=pp 当前不支持 --compare-direct", stderr.getvalue())

    def test_pp_trace_and_dump_are_rejected_even_with_debug_opt_in(self) -> None:
        parser = build_parser()
        cases = [
            ("--trace-layers",),
            ("--dump-layer", "3"),
        ]

        for extra_args in cases:
            with self.subTest(extra_args=extra_args):
                args = parser.parse_args(
                    [
                        "--modality",
                        "text",
                        "--mode",
                        "decode",
                        "--backend",
                        "pp",
                        *extra_args,
                        "--allow-debug-paths",
                    ]
                )
                stderr = io.StringIO()

                with self.assertRaises(SystemExit):
                    with redirect_stderr(stderr):
                        _reject_unsupported_debug_transport_backend(parser, args)

                self.assertIn("backend=pp 当前不支持", stderr.getvalue())

    def test_main_path_manifest_kwargs_keep_runtime_reference_disabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "hybrid",
            ]
        )

        kwargs = _build_direct_manifest_kwargs(args)

        self.assertFalse(kwargs["include_runtime_reference"])

    def test_compare_direct_manifest_kwargs_enable_runtime_reference(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "prefill",
                "--backend",
                "hybrid",
                "--compare-direct",
                "--allow-debug-paths",
            ]
        )

        kwargs = _build_direct_manifest_kwargs(args)

        self.assertTrue(kwargs["include_runtime_reference"])

    def test_pp_debug_flags_do_not_enable_runtime_reference(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "prefill",
                "--backend",
                "pp",
                "--compare-direct",
                "--allow-debug-paths",
            ]
        )

        kwargs = _build_direct_manifest_kwargs(args)

        self.assertFalse(kwargs["include_runtime_reference"])

    def test_generate_compare_direct_is_rejected_even_with_debug_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "hybrid",
                "--compare-direct",
                "--allow-debug-paths",
            ]
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                _reject_unsupported_generate_debug_flags(parser, args)

        self.assertIn("--mode generate 当前不支持 --compare-direct", stderr.getvalue())

    def test_generate_trace_and_dump_are_rejected_even_with_debug_opt_in(self) -> None:
        parser = build_parser()
        cases = [
            ("--trace-layers",),
            ("--dump-layer", "3"),
        ]

        for extra_args in cases:
            with self.subTest(extra_args=extra_args):
                args = parser.parse_args(
                    [
                        "--modality",
                        "text",
                        "--mode",
                        "generate",
                        "--backend",
                        "pp",
                        *extra_args,
                        "--allow-debug-paths",
                    ]
                )
                stderr = io.StringIO()

                with self.assertRaises(SystemExit):
                    with redirect_stderr(stderr):
                        _reject_unsupported_generate_debug_flags(parser, args)

                self.assertIn("--mode generate 当前不支持", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
