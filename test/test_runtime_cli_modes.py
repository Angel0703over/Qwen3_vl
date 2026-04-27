from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import qwen3vl_tp_runtime.scripts.runtime as runtime_script
from qwen3vl_tp_runtime.scripts.runtime import build_parser
from qwen3vl_tp_runtime.scripts.runtime_cli import (
    ParallelConfig,
    _build_direct_manifest_kwargs,
    _debug_path_warnings,
    _load_hybrid_manifest_for_args,
    _load_pipeline_manifest_for_args,
    _load_tp_manifest_for_args,
    _reject_unsupported_debug_transport_backend,
    _reject_unsupported_generate_debug_flags,
    _require_debug_path_opt_in,
    build_even_stage_ranges,
)


class RuntimeCliModesTest(unittest.TestCase):
    def _write_config(self, model_dir: Path, *, num_hidden_layers: int = 36) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps({"text_config": {"num_hidden_layers": num_hidden_layers}})
        )

    def test_parser_help_marks_manifest_replay_as_debug_only(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("Recommended main path", help_text)
        self.assertIn("--allow-debug-paths", help_text)
        self.assertIn("--manifest-path", help_text)
        self.assertIn("--pp", help_text)
        self.assertIn("--tp", help_text)
        self.assertIn("debug/regression workflows", help_text)
        self.assertIn("non-generate runs", help_text)
        self.assertNotIn("--action", help_text)

    def test_even_stage_ranges_split_model_layers_by_pp_degree(self) -> None:
        self.assertEqual(build_even_stage_ranges(num_layers=36, pp_degree=2), ["0:17", "18:35"])
        self.assertEqual(build_even_stage_ranges(num_layers=37, pp_degree=2), ["0:18", "19:36"])

    def test_parallel_config_expands_pp_and_tp_shortcuts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_config(Path(tmp_dir), num_hidden_layers=36)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--modality",
                    "text",
                    "--mode",
                    "generate",
                    "--backend",
                    "hybrid",
                    "--model-path",
                    tmp_dir,
                    "--pp",
                    "2",
                    "--tp",
                    "2",
                ]
            )

            parallel_config = ParallelConfig.from_args(args)

        self.assertEqual(parallel_config.stage_ranges, ["0:17", "18:35"])
        self.assertEqual(parallel_config.tp_degrees, [2, 2])
        self.assertTrue(parallel_config.resolved_from_pp)
        self.assertTrue(parallel_config.resolved_from_tp)

    def test_parallel_config_can_be_reapplied_without_shortcut_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_config(Path(tmp_dir), num_hidden_layers=36)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--modality",
                    "text",
                    "--mode",
                    "generate",
                    "--backend",
                    "pp",
                    "--model-path",
                    tmp_dir,
                    "--pp",
                    "2",
                ]
            )

            ParallelConfig.from_args(args).apply(args)
            parallel_config = ParallelConfig.from_args(args)

        self.assertEqual(parallel_config.stage_ranges, ["0:17", "18:35"])
        self.assertEqual(parallel_config.tp_degrees, [1, 1])

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

    def test_tp_manifest_path_uses_debug_tp_replay_helper(self) -> None:
        args = argparse.Namespace(manifest_path="/tmp/tp_manifest.pt")
        sentinel = object()

        with patch(
            "qwen3vl_tp_runtime.scripts.runtime_cli.load_debug_tp_manifest",
            return_value=sentinel,
        ) as load_debug_mock:
            manifest = _load_tp_manifest_for_args(args)

        self.assertIs(manifest, sentinel)
        load_debug_mock.assert_called_once_with("/tmp/tp_manifest.pt")

    def test_backend_tp_main_path_builds_direct_tp_manifest(self) -> None:
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
            "build_direct_tp_manifest",
            return_value=sentinel,
        ) as build_mock:
            manifest = _load_tp_manifest_for_args(args)

        self.assertIs(manifest, sentinel)
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.kwargs["stage_ranges"], [(0, 35)])
        self.assertEqual(build_mock.call_args.kwargs["tp_degrees"], [2])
        self.assertFalse(build_mock.call_args.kwargs["include_runtime_reference"])

    def test_backend_tp_accepts_tp_shortcut(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "tp",
                "--tp",
                "2",
            ]
        )
        sentinel = object()

        with patch.object(
            runtime_script,
            "build_direct_tp_manifest",
            return_value=sentinel,
        ) as build_mock:
            manifest = _load_tp_manifest_for_args(args)

        self.assertIs(manifest, sentinel)
        self.assertEqual(build_mock.call_args.kwargs["stage_ranges"], [(0, 35)])
        self.assertEqual(build_mock.call_args.kwargs["tp_degrees"], [2])

    def test_backend_pp_accepts_pp_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_config(Path(tmp_dir), num_hidden_layers=36)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--modality",
                    "text",
                    "--mode",
                    "generate",
                    "--backend",
                    "pp",
                    "--model-path",
                    tmp_dir,
                    "--pp",
                    "2",
                ]
            )

            kwargs = _build_direct_manifest_kwargs(args)

        self.assertEqual(kwargs["stage_ranges"], [(0, 17), (18, 35)])

    def test_backend_hybrid_accepts_pp_and_tp_shortcuts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_config(Path(tmp_dir), num_hidden_layers=36)
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--modality",
                    "text",
                    "--mode",
                    "generate",
                    "--backend",
                    "hybrid",
                    "--model-path",
                    tmp_dir,
                    "--pp",
                    "2",
                    "--tp",
                    "2",
                ]
            )
            sentinel = object()

            with patch.object(
                runtime_script,
                "build_direct_hybrid_manifest",
                return_value=sentinel,
            ) as build_mock:
                manifest = _load_hybrid_manifest_for_args(args, backend="hybrid")

        self.assertIs(manifest, sentinel)
        self.assertEqual(build_mock.call_args.kwargs["stage_ranges"], [(0, 17), (18, 35)])
        self.assertEqual(build_mock.call_args.kwargs["tp_degrees"], [2, 2])

    def test_parallel_shortcuts_reject_explicit_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--modality",
                "text",
                "--mode",
                "generate",
                "--backend",
                "pp",
                "--pp",
                "2",
                "--stage-ranges",
                "0:17",
                "18:35",
            ]
        )

        with self.assertRaises(ValueError):
            _build_direct_manifest_kwargs(args)

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
