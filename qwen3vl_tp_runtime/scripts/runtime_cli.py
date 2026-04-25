"""CLI defaults, validation, and debug-path gating for the unified runtime."""

from __future__ import annotations

import argparse
import sys

from qwen3vl_tp_runtime.hexgen_core import parse_stage_ranges


def _runtime_dep(name: str, fallback=None):
    runtime_mod = sys.modules.get("qwen3vl_tp_runtime.scripts.runtime")
    if runtime_mod is not None and hasattr(runtime_mod, name):
        return getattr(runtime_mod, name)
    if fallback is not None:
        return fallback
    raise AttributeError(name)


def _resolve_defaults(args: argparse.Namespace) -> None:
    if args.dump_topk is None:
        args.dump_topk = args.topk

    default_stage_ranges = _runtime_dep("DEFAULT_STAGE_RANGES")
    tp_single_stage_ranges = _runtime_dep("TP_SINGLE_STAGE_RANGES")
    default_tp_degrees = _runtime_dep("DEFAULT_TP_DEGREES")
    tp_single_stage_degrees = _runtime_dep("TP_SINGLE_STAGE_DEGREES")

    if args.backend == "tp":
        if args.stage_ranges == default_stage_ranges:
            args.stage_ranges = tp_single_stage_ranges.copy()
        if args.tp_degrees == default_tp_degrees:
            args.tp_degrees = tp_single_stage_degrees.copy()


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    live_runners = _runtime_dep("LIVE_RUNNERS")

    if args.backend == "hf":
        if args.mode != "generate":
            parser.error("backend=hf 当前只支持 --mode generate。")
        return

    if args.backend == "live":
        if (args.modality, args.mode) not in live_runners:
            parser.error("当前 live 入口只支持 multimodal 的 prefill/decode/generate。")
        return

    if args.backend == "pp":
        return

    if args.backend == "hybrid":
        return

    if args.backend == "tp":
        if len(args.stage_ranges) != 1:
            parser.error("backend=tp 要求恰好一个 --stage-ranges，也就是单 stage 无 PP。")
        if len(args.tp_degrees) != 1:
            parser.error("backend=tp 要求恰好一个 --tp-degrees。")
        if args.tp_degrees[0] <= 1:
            parser.error("backend=tp 要求 TP 度数大于 1。")
        return

    parser.error(f"未知 backend={args.backend!r}")


def _debug_path_warnings(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if args.backend in {"pp", "tp", "hybrid"} and getattr(args, "manifest_path", None) is not None:
        warnings.append(
            f"[warning] backend={args.backend} --manifest-path 已降级为调试/回放路径；"
            "推荐省略 --manifest-path，直接从 model_path 构建运行参数。"
        )
    return warnings


def _is_debug_path(args: argparse.Namespace) -> bool:
    return args.backend in {"pp", "tp", "hybrid"} and getattr(args, "manifest_path", None) is not None


def _debug_path_label(args: argparse.Namespace) -> str | None:
    if args.backend in {"pp", "tp", "hybrid"} and getattr(args, "manifest_path", None) is not None:
        return f"backend={args.backend} --manifest-path"
    return None


def _require_debug_path_opt_in(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not _is_debug_path(args) or getattr(args, "allow_debug_paths", False):
        return
    label = _debug_path_label(args)
    if label is None:
        return
    parser.error(
        f"{label} 已降级为调试路径；如确需使用，请显式传 --allow-debug-paths。"
        "推荐使用 backend=pp|tp|hybrid 直接从 model_path 构建运行参数。"
    )


def _emit_debug_path_warnings(args: argparse.Namespace) -> None:
    for warning in _debug_path_warnings(args):
        print(warning, file=sys.stderr, flush=True)


def _build_direct_manifest_kwargs(args: argparse.Namespace) -> dict:
    include_runtime_reference = args.compare_direct or args.trace_layers or args.dump_layer is not None
    kwargs = {
        "modality": args.modality,
        "mode": args.mode,
        "stage_ranges": _runtime_dep("parse_stage_ranges", parse_stage_ranges)(args.stage_ranges),
        "model_path": args.model_path,
        "save_dtype": args.save_dtype,
        "include_runtime_reference": include_runtime_reference,
    }
    if args.modality == "text":
        kwargs["prompt"] = args.prompt
        if args.mode == "decode":
            kwargs["decode_token_id"] = args.decode_token_id
        elif args.mode == "generate":
            kwargs["max_new_tokens"] = args.max_new_tokens
    else:
        kwargs["num_frames"] = args.num_frames
        kwargs["frame_dir"] = args.frame_dir
        if args.mode == "decode":
            kwargs["decode_token_id"] = args.decode_token_id
        elif args.mode == "generate":
            kwargs["max_new_tokens"] = args.max_new_tokens
    return kwargs


def _load_pipeline_manifest_for_args(args: argparse.Namespace):
    if args.manifest_path is None:
        return _runtime_dep("build_direct_pipeline_manifest")(**_build_direct_manifest_kwargs(args))
    return _runtime_dep("load_pipeline_manifest")(args.manifest_path)


def _load_hybrid_manifest_for_args(args: argparse.Namespace, *, backend: str):
    if args.manifest_path is None:
        return _runtime_dep("build_direct_hybrid_manifest")(
            **_build_direct_manifest_kwargs(args),
            tp_degrees=args.tp_degrees,
            backend=backend,
        )
    return _runtime_dep("load_hybrid_manifest")(args.manifest_path)


__all__ = [
    "_build_direct_manifest_kwargs",
    "_debug_path_warnings",
    "_emit_debug_path_warnings",
    "_load_hybrid_manifest_for_args",
    "_load_pipeline_manifest_for_args",
    "_require_debug_path_opt_in",
    "_resolve_defaults",
    "_validate_args",
]
