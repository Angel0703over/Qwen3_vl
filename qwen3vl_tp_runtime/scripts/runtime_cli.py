"""CLI defaults, validation, and debug-path gating for the unified runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from ..hexgen_core import parse_stage_ranges
from .runtime_replay import (
    load_debug_hybrid_manifest,
    load_debug_pipeline_manifest,
    load_debug_tp_manifest,
)


def _runtime_dep(name: str, fallback=None):
    runtime_mod = sys.modules.get("qwen3vl_tp_runtime.scripts.runtime")
    if runtime_mod is not None and hasattr(runtime_mod, name):
        return getattr(runtime_mod, name)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, name):
        return getattr(main_mod, name)
    if fallback is not None:
        return fallback
    raise AttributeError(name)


@dataclass(slots=True)
class ParallelConfig:
    """Resolved PP/TP layout from user-facing parallel CLI shortcuts."""

    stage_ranges: list[str]
    tp_degrees: list[int]
    resolved_from_pp: bool = False
    resolved_from_tp: bool = False

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser | None = None,
    ) -> "ParallelConfig":
        _reject_parallel_shortcut_conflicts(args, parser)

        default_stage_ranges = _runtime_dep("DEFAULT_STAGE_RANGES")
        tp_single_stage_ranges = _runtime_dep("TP_SINGLE_STAGE_RANGES")
        default_tp_degrees = _runtime_dep("DEFAULT_TP_DEGREES")
        tp_single_stage_degrees = _runtime_dep("TP_SINGLE_STAGE_DEGREES")

        stage_ranges = args.stage_ranges
        resolved_from_pp = bool(getattr(args, "_stage_ranges_resolved_from_pp", False))
        if stage_ranges is None:
            if getattr(args, "pp", None) is not None:
                try:
                    stage_ranges = build_even_stage_ranges(
                        num_layers=_read_text_num_hidden_layers(args.model_path),
                        pp_degree=args.pp,
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    _raise_or_parser_error(parser, str(exc))
                resolved_from_pp = True
            elif args.backend == "tp":
                stage_ranges = tp_single_stage_ranges.copy()
            else:
                stage_ranges = default_stage_ranges.copy()

        tp_degrees = args.tp_degrees
        resolved_from_tp = bool(getattr(args, "_tp_degrees_resolved_from_tp", False))
        if tp_degrees is None:
            if getattr(args, "tp", None) is not None:
                if args.backend == "hybrid":
                    tp_degrees = [args.tp for _ in stage_ranges]
                else:
                    tp_degrees = [args.tp]
                resolved_from_tp = True
            elif args.backend == "tp":
                tp_degrees = tp_single_stage_degrees.copy()
            elif args.backend == "pp":
                tp_degrees = [1 for _ in stage_ranges]
            else:
                tp_degrees = default_tp_degrees.copy()

        return cls(
            stage_ranges=stage_ranges,
            tp_degrees=tp_degrees,
            resolved_from_pp=resolved_from_pp,
            resolved_from_tp=resolved_from_tp,
        )

    def apply(self, args: argparse.Namespace) -> None:
        args.stage_ranges = self.stage_ranges
        args.tp_degrees = self.tp_degrees
        if self.resolved_from_pp:
            args._stage_ranges_resolved_from_pp = True
        if self.resolved_from_tp:
            args._tp_degrees_resolved_from_tp = True


def _resolve_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    if args.dump_topk is None:
        args.dump_topk = args.topk
    ParallelConfig.from_args(args, parser).apply(args)
    if (
        getattr(args, "video_kv_compression", "none") != "none"
        and getattr(args, "video_kv_keep_ratio", None) is None
        and getattr(args, "video_kv_keep_tokens_per_window", None) is None
    ):
        args.video_kv_keep_ratio = 0.5


def _raise_or_parser_error(parser: argparse.ArgumentParser | None, message: str) -> None:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def _reject_parallel_shortcut_conflicts(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    if (
        getattr(args, "pp", None) is not None
        and args.stage_ranges is not None
        and not getattr(args, "_stage_ranges_resolved_from_pp", False)
    ):
        _raise_or_parser_error(parser, "不能同时传 --pp 和 --stage-ranges；请二选一。")
    if (
        getattr(args, "tp", None) is not None
        and args.tp_degrees is not None
        and not getattr(args, "_tp_degrees_resolved_from_tp", False)
    ):
        _raise_or_parser_error(parser, "不能同时传 --tp 和 --tp-degrees；请二选一。")


def _read_text_num_hidden_layers(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"无法根据 --pp 自动切分：没有找到 config.json: {config_path}")
    payload = json.loads(config_path.read_text())
    text_config = payload.get("text_config", payload)
    return int(text_config["num_hidden_layers"])


def build_even_stage_ranges(*, num_layers: int, pp_degree: int) -> list[str]:
    if pp_degree <= 0:
        raise ValueError(f"--pp 必须大于 0，当前拿到 {pp_degree}。")
    if num_layers <= 0:
        raise ValueError(f"num_layers 必须大于 0，当前拿到 {num_layers}。")
    if pp_degree > num_layers:
        raise ValueError(f"--pp={pp_degree} 不能大于模型层数 {num_layers}。")

    base, remainder = divmod(num_layers, pp_degree)
    ranges: list[str] = []
    start_idx = 0
    for stage_idx in range(pp_degree):
        stage_layers = base + (1 if stage_idx < remainder else 0)
        end_idx = start_idx + stage_layers - 1
        ranges.append(f"{start_idx}:{end_idx}")
        start_idx = end_idx + 1
    return ranges


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _reject_unsupported_debug_transport_backend(parser, args)
    _reject_unsupported_generate_debug_flags(parser, args)
    _validate_video_input_args(parser, args)
    _validate_video_kv_selector_args(parser, args)
    live_runners = _runtime_dep("LIVE_RUNNERS")
    if getattr(args, "tp", None) is not None and args.tp <= 0:
        parser.error(f"--tp 必须大于 0，当前拿到 {args.tp}。")
    if getattr(args, "tp_degrees", None) is not None and any(tp_degree <= 0 for tp_degree in args.tp_degrees):
        parser.error(f"--tp-degrees 每一项都必须大于 0，当前拿到 {args.tp_degrees!r}。")

    if args.backend == "hf":
        if args.mode != "generate":
            parser.error("backend=hf 当前只支持 --mode generate。")
        return

    if args.backend == "live":
        if (args.modality, args.mode) not in live_runners:
            parser.error("当前 live 入口只支持 multimodal 的 prefill/decode/generate。")
        return

    if args.backend == "pp":
        if getattr(args, "tp", None) not in (None, 1):
            parser.error("backend=pp 不支持 --tp > 1；如需 PP+TP，请使用 backend=hybrid。")
        if any(tp_degree != 1 for tp_degree in args.tp_degrees):
            parser.error("backend=pp 不使用 TP；如需 TP，请使用 backend=tp 或 backend=hybrid。")
        return

    if args.backend == "hybrid":
        if len(args.tp_degrees) != len(args.stage_ranges):
            parser.error(
                "backend=hybrid 要求 TP 度数数量和 PP stage 数一致，"
                f"当前 stage_ranges={args.stage_ranges!r} tp_degrees={args.tp_degrees!r}。"
            )
        return

    if args.backend == "tp":
        if getattr(args, "pp", None) not in (None, 1):
            parser.error("backend=tp 是单 stage TP；如需同时指定 --pp 和 --tp，请使用 backend=hybrid。")
        if len(args.stage_ranges) != 1:
            parser.error("backend=tp 要求恰好一个 --stage-ranges，也就是单 stage 无 PP。")
        if len(args.tp_degrees) != 1:
            parser.error("backend=tp 要求恰好一个 --tp-degrees。")
        if args.tp_degrees[0] <= 1:
            parser.error("backend=tp 要求 TP 度数大于 1。")
        return

    parser.error(f"未知 backend={args.backend!r}")


def _validate_video_kv_selector_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    method = getattr(args, "video_kv_compression", "none")
    keep_ratio = getattr(args, "video_kv_keep_ratio", None)
    keep_tokens = getattr(args, "video_kv_keep_tokens_per_window", None)
    if keep_ratio is not None and keep_tokens is not None:
        parser.error("--video-kv-keep-ratio 和 --video-kv-keep-tokens-per-window 不能同时使用。")
    if keep_ratio is not None and not (0.0 < float(keep_ratio) <= 1.0):
        parser.error(f"--video-kv-keep-ratio 必须在 (0, 1]，当前拿到 {keep_ratio}。")
    if keep_tokens is not None and int(keep_tokens) <= 0:
        parser.error(f"--video-kv-keep-tokens-per-window 必须大于 0，当前拿到 {keep_tokens}。")
    if method != "none":
        if args.modality != "multimodal" or args.mode != "generate":
            parser.error("--video-kv-compression 当前只支持 multimodal generate 主路径。")


def _validate_video_input_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    video_path = getattr(args, "video_path", None)
    video_url = getattr(args, "video_url", None)
    has_full_video_input = video_path is not None or video_url is not None
    has_video_sampling = any(
        getattr(args, name, None) is not None
        for name in (
            "video_fps",
            "video_nframes",
            "video_start",
            "video_end",
            "video_min_frames",
            "video_max_frames",
        )
    )
    if (has_full_video_input or has_video_sampling) and args.modality != "multimodal":
        parser.error("完整视频输入参数当前只支持 --modality multimodal。")
    if video_path is not None and video_url is not None:
        parser.error("--video-path 和 --video-url 不能同时使用。")
    if has_video_sampling and not has_full_video_input:
        parser.error("video_* 采样参数只用于 --video-path/--video-url；frame-dir 路径请继续使用 --sample-fps。")
    if getattr(args, "video_fps", None) is not None and getattr(args, "video_nframes", None) is not None:
        parser.error("--video-fps 和 --video-nframes 不能同时使用。")
    if getattr(args, "video_nframes", None) is not None and int(args.video_nframes) <= 0:
        parser.error(f"--video-nframes 必须大于 0，当前拿到 {args.video_nframes}。")
    if getattr(args, "video_min_frames", None) is not None and int(args.video_min_frames) <= 0:
        parser.error(f"--video-min-frames 必须大于 0，当前拿到 {args.video_min_frames}。")
    if getattr(args, "video_max_frames", None) is not None and int(args.video_max_frames) <= 0:
        parser.error(f"--video-max-frames 必须大于 0，当前拿到 {args.video_max_frames}。")
    if (
        getattr(args, "video_min_frames", None) is not None
        and getattr(args, "video_max_frames", None) is not None
        and int(args.video_min_frames) > int(args.video_max_frames)
    ):
        parser.error("--video-min-frames 不能大于 --video-max-frames。")


def _debug_path_warnings(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if args.backend in {"pp", "tp", "hybrid"} and getattr(args, "manifest_path", None) is not None:
        warnings.append(
            f"[warning] backend={args.backend} --manifest-path 已降级为调试/回放路径；"
            "推荐省略 --manifest-path，直接从 model_path 构建运行参数。"
        )
    debug_flags = _debug_transport_flags(args)
    if _supports_debug_transport_backend(args) and debug_flags:
        warnings.append(
            f"[warning] backend={args.backend} {' '.join(debug_flags)} 已归类为调试路径；"
            "这会重新启用更重的 reference/trace transport。"
            "主路径建议省略这些标志。"
        )
    return warnings


def _debug_transport_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if getattr(args, "compare_direct", False):
        flags.append("--compare-direct")
    if getattr(args, "trace_layers", False):
        flags.append("--trace-layers")
    if getattr(args, "dump_layer", None) is not None:
        flags.append("--dump-layer")
    return flags


def _supports_debug_transport_backend(args: argparse.Namespace) -> bool:
    return args.backend in {"tp", "hybrid"}


def _reject_unsupported_debug_transport_backend(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    debug_flags = _debug_transport_flags(args)
    if not debug_flags or _supports_debug_transport_backend(args):
        return
    parser.error(
        f"backend={args.backend} 当前不支持 {' '.join(debug_flags)}；"
        "这些标志目前只在 backend=tp|hybrid 的调试路径上有效。"
    )


def _reject_unsupported_generate_debug_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    debug_flags = _debug_transport_flags(args)
    if args.mode != "generate" or not debug_flags:
        return
    parser.error(
        f"--mode generate 当前不支持 {' '.join(debug_flags)}；"
        "这些标志目前只在非 generate 的 TP/hybrid 调试路径上有效。"
    )


def _is_debug_path(args: argparse.Namespace) -> bool:
    if args.backend in {"pp", "tp", "hybrid"} and getattr(args, "manifest_path", None) is not None:
        return True
    return _supports_debug_transport_backend(args) and bool(_debug_transport_flags(args))


def _debug_path_label(args: argparse.Namespace) -> str | None:
    if args.backend in {"pp", "tp", "hybrid"}:
        debug_parts: list[str] = []
        if getattr(args, "manifest_path", None) is not None:
            debug_parts.append("--manifest-path")
        if _supports_debug_transport_backend(args):
            debug_parts.extend(_debug_transport_flags(args))
        if debug_parts:
            return f"backend={args.backend} {' '.join(debug_parts)}"
    return None


def _require_debug_path_opt_in(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not _is_debug_path(args) or getattr(args, "allow_debug_paths", False):
        return
    label = _debug_path_label(args)
    if label is None:
        return
    parser.error(
        f"{label} 已降级为调试路径；如确需使用，请显式传 --allow-debug-paths。"
        "推荐省略这些调试入口，直接使用 backend=pp|tp|hybrid 从 model_path 构建运行参数。"
    )


def _emit_debug_path_warnings(args: argparse.Namespace) -> None:
    for warning in _debug_path_warnings(args):
        print(warning, file=sys.stderr, flush=True)


def _build_direct_manifest_kwargs(args: argparse.Namespace) -> dict:
    _resolve_defaults(args)
    include_runtime_reference = _supports_debug_transport_backend(args) and (
        args.compare_direct or args.trace_layers or args.dump_layer is not None
    )
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
        kwargs["sample_fps"] = args.sample_fps
        kwargs["video_path"] = args.video_path
        kwargs["video_url"] = args.video_url
        kwargs["video_fps"] = args.video_fps
        kwargs["video_nframes"] = args.video_nframes
        kwargs["video_start"] = args.video_start
        kwargs["video_end"] = args.video_end
        kwargs["video_min_frames"] = args.video_min_frames
        kwargs["video_max_frames"] = args.video_max_frames
        kwargs["video_kv_compression"] = args.video_kv_compression
        kwargs["video_kv_keep_ratio"] = args.video_kv_keep_ratio
        kwargs["video_kv_keep_tokens_per_window"] = args.video_kv_keep_tokens_per_window
        if args.mode == "decode":
            kwargs["decode_token_id"] = args.decode_token_id
        elif args.mode == "generate":
            kwargs["max_new_tokens"] = args.max_new_tokens
    return kwargs


def _load_pipeline_manifest_for_args(args: argparse.Namespace):
    if args.manifest_path is None:
        return _runtime_dep("build_direct_pipeline_manifest")(**_build_direct_manifest_kwargs(args))
    return load_debug_pipeline_manifest(args.manifest_path)


def _load_hybrid_manifest_for_args(args: argparse.Namespace, *, backend: str):
    if args.manifest_path is None:
        return _runtime_dep("build_direct_hybrid_manifest")(
            **_build_direct_manifest_kwargs(args),
            tp_degrees=args.tp_degrees,
            backend=backend,
        )
    return load_debug_hybrid_manifest(args.manifest_path)


def _load_tp_manifest_for_args(args: argparse.Namespace):
    if args.manifest_path is None:
        return _runtime_dep("build_direct_tp_manifest")(
            **_build_direct_manifest_kwargs(args),
            tp_degrees=args.tp_degrees,
        )
    return load_debug_tp_manifest(args.manifest_path)


__all__ = [
    "ParallelConfig",
    "_build_direct_manifest_kwargs",
    "build_even_stage_ranges",
    "_debug_path_warnings",
    "_emit_debug_path_warnings",
    "_load_hybrid_manifest_for_args",
    "_load_pipeline_manifest_for_args",
    "_load_tp_manifest_for_args",
    "_reject_unsupported_debug_transport_backend",
    "_reject_unsupported_generate_debug_flags",
    "_require_debug_path_opt_in",
    "_resolve_defaults",
    "_supports_debug_transport_backend",
    "_validate_args",
]
