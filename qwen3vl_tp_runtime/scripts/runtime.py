"""Unified runtime CLI for live, bundle, pipeline-parallel, tensor-parallel, and hybrid execution."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3vl_tp_runtime.hexgen_core import (
    FRAME_DIR,
    MODEL_PATH,
    MULTIMODAL_DECODE_BUNDLE_PATH,
    MULTIMODAL_DECODE_HYBRID_BUNDLE_DIR,
    MULTIMODAL_DECODE_HYBRID_MANIFEST_PATH,
    MULTIMODAL_DECODE_PIPELINE_BUNDLE_DIR,
    MULTIMODAL_DECODE_PIPELINE_MANIFEST_PATH,
    MULTIMODAL_GENERATE_HYBRID_BUNDLE_DIR,
    MULTIMODAL_GENERATE_HYBRID_MANIFEST_PATH,
    MULTIMODAL_GENERATE_PIPELINE_BUNDLE_DIR,
    MULTIMODAL_GENERATE_PIPELINE_MANIFEST_PATH,
    MULTIMODAL_PREFILL_BUNDLE_PATH,
    MULTIMODAL_PREFILL_HYBRID_BUNDLE_DIR,
    MULTIMODAL_PREFILL_HYBRID_MANIFEST_PATH,
    MULTIMODAL_PREFILL_PIPELINE_BUNDLE_DIR,
    MULTIMODAL_PREFILL_PIPELINE_MANIFEST_PATH,
    TEXT_DECODE_BUNDLE_PATH,
    TEXT_DECODE_HYBRID_BUNDLE_DIR,
    TEXT_DECODE_HYBRID_MANIFEST_PATH,
    TEXT_DECODE_PIPELINE_BUNDLE_DIR,
    TEXT_DECODE_PIPELINE_MANIFEST_PATH,
    TEXT_GENERATE_BUNDLE_PATH,
    TEXT_GENERATE_HYBRID_BUNDLE_DIR,
    TEXT_GENERATE_HYBRID_MANIFEST_PATH,
    TEXT_GENERATE_PIPELINE_BUNDLE_DIR,
    TEXT_GENERATE_PIPELINE_MANIFEST_PATH,
    TEXT_PREFILL_BUNDLE_PATH,
    TEXT_PREFILL_HYBRID_BUNDLE_DIR,
    TEXT_PREFILL_HYBRID_MANIFEST_PATH,
    TEXT_PREFILL_PIPELINE_BUNDLE_DIR,
    TEXT_PREFILL_PIPELINE_MANIFEST_PATH,
    get_device,
    init_dist,
    load_hybrid_manifest,
    load_pipeline_manifest,
    parse_stage_ranges,
    prepare_multimodal_decode_hybrid,
    prepare_multimodal_decode_pipeline,
    prepare_multimodal_generate_hybrid,
    prepare_multimodal_generate_pipeline,
    prepare_multimodal_prefill_hybrid,
    prepare_multimodal_prefill_pipeline,
    prepare_text_decode_hybrid,
    prepare_text_decode_pipeline,
    prepare_text_generate_hybrid,
    prepare_text_generate_pipeline,
    prepare_text_prefill_hybrid,
    prepare_text_prefill_pipeline,
    run_text_generate_pipeline_rank,
    run_text_pipeline_rank,
)
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import TextHybridRunner
from qwen3vl_tp_runtime.models.qwen3vl import (
    build_inputs,
    build_direct_hybrid_manifest,
    build_direct_pipeline_manifest,
    build_text_inputs,
    list_frames,
    load_bundle,
    load_model,
    load_processor,
)
from qwen3vl_tp_runtime.scripts.bundle import (
    multimodal_decode as bundle_multimodal_decode,
    multimodal_prefill as bundle_multimodal_prefill,
    text_decode as bundle_text_decode,
    text_generate as bundle_text_generate,
    text_prefill as bundle_text_prefill,
)
from qwen3vl_tp_runtime.scripts.common import summarize_last_token_topk
from qwen3vl_tp_runtime.scripts.live import live_multimodal_runtime


Combo = tuple[str, str]


LIVE_RUNNERS = {
    ("multimodal", "prefill"): live_multimodal_runtime.run_prefill,
    ("multimodal", "decode"): live_multimodal_runtime.run_decode,
    ("multimodal", "generate"): live_multimodal_runtime.run_generate,
}

BUNDLE_PREPARE_RUNNERS = {
    ("text", "prefill"): bundle_text_prefill.run_prepare,
    ("text", "decode"): bundle_text_decode.run_prepare,
    ("text", "generate"): bundle_text_generate.run_prepare,
    ("multimodal", "prefill"): bundle_multimodal_prefill.run_prepare,
    ("multimodal", "decode"): bundle_multimodal_decode.run_prepare,
}

BUNDLE_RUN_RUNNERS = {
    ("text", "prefill"): bundle_text_prefill.run_direct,
    ("text", "decode"): bundle_text_decode.run_direct,
    ("text", "generate"): bundle_text_generate.run_direct,
    ("multimodal", "prefill"): bundle_multimodal_prefill.run_direct,
    ("multimodal", "decode"): bundle_multimodal_decode.run_direct,
}

PIPELINE_PREPARE_RUNNERS = {
    ("text", "prefill"): prepare_text_prefill_pipeline,
    ("text", "decode"): prepare_text_decode_pipeline,
    ("text", "generate"): prepare_text_generate_pipeline,
    ("multimodal", "prefill"): prepare_multimodal_prefill_pipeline,
    ("multimodal", "decode"): prepare_multimodal_decode_pipeline,
    ("multimodal", "generate"): prepare_multimodal_generate_pipeline,
}

HYBRID_PREPARE_RUNNERS = {
    ("text", "prefill"): prepare_text_prefill_hybrid,
    ("text", "decode"): prepare_text_decode_hybrid,
    ("text", "generate"): prepare_text_generate_hybrid,
    ("multimodal", "prefill"): prepare_multimodal_prefill_hybrid,
    ("multimodal", "decode"): prepare_multimodal_decode_hybrid,
    ("multimodal", "generate"): prepare_multimodal_generate_hybrid,
}

DEFAULT_BUNDLE_PATHS = {
    ("text", "prefill"): TEXT_PREFILL_BUNDLE_PATH,
    ("text", "decode"): TEXT_DECODE_BUNDLE_PATH,
    ("text", "generate"): TEXT_GENERATE_BUNDLE_PATH,
    ("multimodal", "prefill"): MULTIMODAL_PREFILL_BUNDLE_PATH,
    ("multimodal", "decode"): MULTIMODAL_DECODE_BUNDLE_PATH,
}

DEFAULT_PIPELINE_BUNDLE_DIRS = {
    ("text", "prefill"): TEXT_PREFILL_PIPELINE_BUNDLE_DIR,
    ("text", "decode"): TEXT_DECODE_PIPELINE_BUNDLE_DIR,
    ("text", "generate"): TEXT_GENERATE_PIPELINE_BUNDLE_DIR,
    ("multimodal", "prefill"): MULTIMODAL_PREFILL_PIPELINE_BUNDLE_DIR,
    ("multimodal", "decode"): MULTIMODAL_DECODE_PIPELINE_BUNDLE_DIR,
    ("multimodal", "generate"): MULTIMODAL_GENERATE_PIPELINE_BUNDLE_DIR,
}

DEFAULT_PIPELINE_MANIFEST_PATHS = {
    ("text", "prefill"): TEXT_PREFILL_PIPELINE_MANIFEST_PATH,
    ("text", "decode"): TEXT_DECODE_PIPELINE_MANIFEST_PATH,
    ("text", "generate"): TEXT_GENERATE_PIPELINE_MANIFEST_PATH,
    ("multimodal", "prefill"): MULTIMODAL_PREFILL_PIPELINE_MANIFEST_PATH,
    ("multimodal", "decode"): MULTIMODAL_DECODE_PIPELINE_MANIFEST_PATH,
    ("multimodal", "generate"): MULTIMODAL_GENERATE_PIPELINE_MANIFEST_PATH,
}

DEFAULT_HYBRID_BUNDLE_DIRS = {
    ("text", "prefill"): TEXT_PREFILL_HYBRID_BUNDLE_DIR,
    ("text", "decode"): TEXT_DECODE_HYBRID_BUNDLE_DIR,
    ("text", "generate"): TEXT_GENERATE_HYBRID_BUNDLE_DIR,
    ("multimodal", "prefill"): MULTIMODAL_PREFILL_HYBRID_BUNDLE_DIR,
    ("multimodal", "decode"): MULTIMODAL_DECODE_HYBRID_BUNDLE_DIR,
    ("multimodal", "generate"): MULTIMODAL_GENERATE_HYBRID_BUNDLE_DIR,
}

DEFAULT_HYBRID_MANIFEST_PATHS = {
    ("text", "prefill"): TEXT_PREFILL_HYBRID_MANIFEST_PATH,
    ("text", "decode"): TEXT_DECODE_HYBRID_MANIFEST_PATH,
    ("text", "generate"): TEXT_GENERATE_HYBRID_MANIFEST_PATH,
    ("multimodal", "prefill"): MULTIMODAL_PREFILL_HYBRID_MANIFEST_PATH,
    ("multimodal", "decode"): MULTIMODAL_DECODE_HYBRID_MANIFEST_PATH,
    ("multimodal", "generate"): MULTIMODAL_GENERATE_HYBRID_MANIFEST_PATH,
}

GENERATE_PIPELINE_TYPES = {"text_generate", "multimodal_generate"}
DEFAULT_STAGE_RANGES = ["0:17", "18:35"]
DEFAULT_TP_DEGREES = [2, 2]
TP_SINGLE_STAGE_RANGES = ["0:35"]
TP_SINGLE_STAGE_DEGREES = [2]


def _tensor_shape_map_to_json(payload: dict[str, tuple[int, ...] | None]) -> dict[str, list[int] | None]:
    return {
        key: (None if value is None else list(value))
        for key, value in payload.items()
    }


def _manifest_boundaries_to_json(manifest) -> list[dict[str, float | int]]:
    return [
        {
            "src_stage_idx": boundary.src_stage_idx,
            "dst_stage_idx": boundary.dst_stage_idx,
            "max_diff": boundary.max_diff,
            "mean_diff": boundary.mean_diff,
        }
        for boundary in manifest.boundaries
    ]


def _derive_tp_output_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path.replace("_hybrid_", "_tp_").replace("_hybrid", "_tp")


def _normalize_tp_runtime_name(runtime_name: str) -> str:
    return runtime_name.replace("_hybrid", "_tp")


@lru_cache(maxsize=4)
def _load_processor_cached(model_path: str):
    return load_processor(model_path)


def _decode_generated_token_ids(
    token_ids: list[int],
    *,
    model_path: str,
    keep_special_tokens: bool,
    clean_up_tokenization_spaces: bool,
) -> tuple[str | None, str | None]:
    if not token_ids:
        return "", None

    try:
        processor = _load_processor_cached(model_path)
        decoded_texts = processor.post_process_image_text_to_text(
            [torch.tensor(token_ids, dtype=torch.long)],
            skip_special_tokens=not keep_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return (decoded_texts[0] if decoded_texts else ""), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _attach_generated_texts(summary: dict, args: argparse.Namespace) -> dict:
    if args.mode != "generate":
        return summary
    if "generated_token_ids" not in summary:
        return summary

    generated_text, generated_error = _decode_generated_token_ids(
        summary["generated_token_ids"],
        model_path=args.model_path,
        keep_special_tokens=args.keep_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
    )
    reference_generated_token_ids = summary.get("reference_generated_token_ids")
    reference_text = None
    reference_error = None
    if reference_generated_token_ids is not None:
        reference_text, reference_error = _decode_generated_token_ids(
            reference_generated_token_ids,
            model_path=args.model_path,
            keep_special_tokens=args.keep_special_tokens,
            clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        )

    if generated_text is not None:
        summary["generated_text"] = generated_text
    if reference_text is not None:
        summary["reference_generated_text"] = reference_text
    if generated_error is not None or reference_error is not None:
        summary["generated_text_decode_error"] = {
            "runtime": generated_error,
            "reference": reference_error,
        }
    return summary


def _build_script_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        prompt=args.prompt,
        num_frames=args.num_frames,
        frame_dir=args.frame_dir,
        decode_token_id=args.decode_token_id,
        max_new_tokens=args.max_new_tokens,
        bundle_path=args.bundle_path,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        stage_ranges=args.stage_ranges,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
        device=args.device,
        compute_dtype=args.compute_dtype,
        comm_dtype=args.comm_dtype,
        topk=args.topk,
    )


def _build_hf_generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
        kwargs["top_k"] = args.sample_top_k
    return kwargs


def _trim_generated_ids(
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> list[torch.Tensor]:
    return [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]


def _run_hf_generate(args: argparse.Namespace) -> None:
    model = load_model(
        args.model_path,
        attn_implementation=args.attn_implementation,
    )
    processor = load_processor(args.model_path)
    device = next(model.parameters()).device

    if args.modality == "multimodal":
        frame_paths = list_frames(args.num_frames, args.frame_dir)
        inputs = build_inputs(
            processor,
            frame_paths,
            prompt=args.prompt,
            sample_fps=args.sample_fps,
            add_generation_prompt=True,
        )
    else:
        frame_paths = []
        inputs = build_text_inputs(
            processor,
            args.prompt,
            add_generation_prompt=True,
        )

    inputs = inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **_build_hf_generation_kwargs(args),
        )

    input_ids = inputs["input_ids"]
    trimmed_ids = _trim_generated_ids(input_ids, generated_ids)
    decoded_texts = processor.post_process_image_text_to_text(
        trimmed_ids,
        skip_special_tokens=not args.keep_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
    )

    summary = {
        "backend": "hf",
        "mode": "generate",
        "modality": args.modality,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.sample_top_k if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "attn_implementation": args.attn_implementation,
        "input_token_count": int(input_ids.shape[-1]),
        "generated_token_count": int(trimmed_ids[0].shape[-1]) if trimmed_ids else 0,
        "generated_token_ids": trimmed_ids[0].tolist() if trimmed_ids else [],
        "generated_text": decoded_texts[0] if decoded_texts else "",
    }
    if args.modality == "multimodal":
        summary["num_frames"] = len(frame_paths)
        summary["frame_paths"] = frame_paths
        summary["sample_fps"] = args.sample_fps

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _resolve_defaults(args: argparse.Namespace) -> None:
    combo = (args.modality, args.mode)
    if args.dump_topk is None:
        args.dump_topk = args.topk

    if args.backend == "bundle" and args.bundle_path is None:
        args.bundle_path = DEFAULT_BUNDLE_PATHS.get(combo)
    if args.backend == "pp" and args.action == "prepare":
        if args.bundle_dir is None:
            args.bundle_dir = DEFAULT_PIPELINE_BUNDLE_DIRS.get(combo)
        if args.manifest_path is None:
            args.manifest_path = DEFAULT_PIPELINE_MANIFEST_PATHS.get(combo)
    if args.backend == "hybrid" and args.action == "prepare":
        if args.bundle_dir is None:
            args.bundle_dir = DEFAULT_HYBRID_BUNDLE_DIRS.get(combo)
        if args.manifest_path is None:
            args.manifest_path = DEFAULT_HYBRID_MANIFEST_PATHS.get(combo)
    if args.backend == "tp":
        if args.stage_ranges == DEFAULT_STAGE_RANGES:
            args.stage_ranges = TP_SINGLE_STAGE_RANGES.copy()
        if args.tp_degrees == DEFAULT_TP_DEGREES:
            args.tp_degrees = TP_SINGLE_STAGE_DEGREES.copy()
        if args.action == "prepare" and args.bundle_dir is None:
            args.bundle_dir = _derive_tp_output_path(DEFAULT_HYBRID_BUNDLE_DIRS.get(combo))
        if args.action == "prepare" and args.manifest_path is None:
            args.manifest_path = _derive_tp_output_path(DEFAULT_HYBRID_MANIFEST_PATHS.get(combo))


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    combo = (args.modality, args.mode)
    if args.backend == "hf":
        if args.action != "run":
            parser.error("backend=hf 只支持 --action run。")
        if args.mode != "generate":
            parser.error("backend=hf 当前只支持 --mode generate。")
        return

    if args.backend == "live":
        if args.action != "run":
            parser.error("backend=live 只支持 --action run。")
        if combo not in LIVE_RUNNERS:
            parser.error("当前 live 入口只支持 multimodal 的 prefill/decode/generate。")
        return

    if args.backend == "bundle":
        if args.action == "prepare" and combo not in BUNDLE_PREPARE_RUNNERS:
            parser.error("当前 bundle 入口还不支持这个 modality/mode 组合。")
        if args.action == "run" and combo not in BUNDLE_RUN_RUNNERS:
            parser.error("当前 bundle 入口还不支持这个 modality/mode 组合。")
        if args.bundle_path is None:
            parser.error("这个组合没有默认 bundle_path，请显式传 --bundle-path。")
        return

    if args.backend == "pp":
        if combo not in PIPELINE_PREPARE_RUNNERS:
            parser.error("当前 pp 入口还不支持这个 modality/mode 组合。")
        if args.action == "prepare" and (args.bundle_dir is None or args.manifest_path is None):
            parser.error("pp 入口需要 bundle_dir 和 manifest_path。")
        return

    if args.backend == "hybrid":
        if combo not in HYBRID_PREPARE_RUNNERS:
            parser.error("当前 hybrid 入口还不支持这个 modality/mode 组合。")
        if args.action == "prepare" and (args.bundle_dir is None or args.manifest_path is None):
            parser.error("hybrid 入口需要 bundle_dir 和 manifest_path。")
        return

    if args.backend == "tp":
        if combo not in HYBRID_PREPARE_RUNNERS:
            parser.error("当前 tp 入口还不支持这个 modality/mode 组合。")
        if args.action == "prepare" and (args.bundle_dir is None or args.manifest_path is None):
            parser.error("tp 入口需要 bundle_dir 和 manifest_path。")
        if len(args.stage_ranges) != 1:
            parser.error("backend=tp 要求恰好一个 --stage-ranges，也就是单 stage 无 PP。")
        if len(args.tp_degrees) != 1:
            parser.error("backend=tp 要求恰好一个 --tp-degrees。")
        if args.tp_degrees[0] <= 1:
            parser.error("backend=tp 要求 TP 度数大于 1。")
        return

    parser.error(f"未知 backend={args.backend!r}")


def _build_prepare_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "stage_ranges": parse_stage_ranges(args.stage_ranges),
        "bundle_dir": args.bundle_dir,
        "manifest_path": args.manifest_path,
        "save_dtype": args.save_dtype,
        "model_path": args.model_path,
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


def _build_direct_manifest_kwargs(args: argparse.Namespace) -> dict:
    include_runtime_reference = args.compare_direct or args.trace_layers or args.dump_layer is not None
    kwargs = {
        "modality": args.modality,
        "mode": args.mode,
        "stage_ranges": parse_stage_ranges(args.stage_ranges),
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


def _summarize_prepare_manifest(manifest, args: argparse.Namespace, *, include_hybrid_layout: bool) -> dict:
    summary = {
        "backend": args.backend,
        "modality": args.modality,
        "mode": args.mode,
        "pipeline_type": manifest.pipeline_type,
        "bundle_dir": manifest.bundle_dir,
        "manifest_path": args.manifest_path,
        "num_stages": manifest.num_stages,
        "stage_ranges": manifest.stage_ranges,
        "boundaries": _manifest_boundaries_to_json(manifest),
        "save_dtype": manifest.save_dtype,
    }
    if hasattr(manifest, "num_frames"):
        summary["num_frames"] = manifest.num_frames
    if include_hybrid_layout:
        summary.update(
            {
                "runtime": manifest.runtime,
                "tp_degrees": manifest.tp_degrees,
                "stage_rank_groups": manifest.stage_rank_groups,
                "pp_rank_groups": manifest.pp_rank_groups,
                "world_size": manifest.world_size,
            }
        )
    return summary


def _summarize_pipeline_run(stats: dict, manifest, topk: int) -> dict:
    summary = {
        "rank": stats["rank"],
        "backend": "pp",
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "num_stages": stats["num_stages"],
        "start_idx": stats["start_idx"],
        "end_idx": stats["end_idx"],
        "num_layers": stats["num_layers"],
        "device": stats["device"],
        "comm_dtype": stats["comm_dtype"],
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "received_payload_keys": stats["received_payload_keys"],
        "sent_payload_keys": stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(stats["sent_tensor_shapes"]),
        "boundary_max_diff": stats["boundary_max_diff"],
        "boundary_mean_diff": stats["boundary_mean_diff"],
        "stage_max_diff": stats["stage_max_diff"],
        "stage_mean_diff": stats["stage_mean_diff"],
    }

    if "stage_output" in stats and "reference_output" in stats:
        stage_output = stats["stage_output"]
        reference_output = stats["reference_output"]
        summary["last_stage_topk"] = summarize_last_token_topk(stage_output, topk)
        summary["reference_topk"] = summarize_last_token_topk(reference_output, topk)
    return summary


def _summarize_generate_phase_stats(phase_stats: dict) -> dict:
    return {
        "input_shape": list(phase_stats["input_shape"]),
        "output_shape": list(phase_stats["output_shape"]),
        "received_payload_keys": phase_stats["received_payload_keys"],
        "sent_payload_keys": phase_stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(phase_stats["sent_tensor_shapes"]),
        "boundary_max_diff": phase_stats["boundary_max_diff"],
        "boundary_mean_diff": phase_stats["boundary_mean_diff"],
        "embedding_max_diff": phase_stats["embedding_max_diff"],
        "embedding_mean_diff": phase_stats["embedding_mean_diff"],
        "hidden_stage_max_diff": phase_stats["hidden_stage_max_diff"],
        "hidden_stage_mean_diff": phase_stats["hidden_stage_mean_diff"],
        "norm_max_diff": phase_stats["norm_max_diff"],
        "norm_mean_diff": phase_stats["norm_mean_diff"],
        "stage_max_diff": phase_stats["stage_max_diff"],
        "stage_mean_diff": phase_stats["stage_mean_diff"],
        "predicted_token_id": phase_stats["predicted_token_id"],
        "reference_token_id": phase_stats["reference_token_id"],
    }


def _maybe_load_generate_reference_logits(bundle_path: str | None) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    if not bundle_path:
        return None, None
    last_bundle = load_bundle(bundle_path)
    if last_bundle.get("runtime_only_generate"):
        return None, None
    prefill_bundle = last_bundle.get("prefill")
    decode_steps = last_bundle.get("decode_steps")
    if not isinstance(prefill_bundle, dict) or decode_steps is None:
        return None, None
    reference_prefill = prefill_bundle.get("logits")
    if reference_prefill is None:
        return None, None
    reference_steps = []
    for step_payload in decode_steps:
        if not isinstance(step_payload, dict):
            return None, None
        logits = step_payload.get("logits")
        if logits is None:
            return None, None
        reference_steps.append(logits)
    return reference_prefill, reference_steps


def _summarize_pipeline_generate_run(stats: dict, manifest, topk: int) -> dict:
    summary = {
        "rank": stats["rank"],
        "backend": "pp",
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "num_stages": stats["num_stages"],
        "start_idx": stats["start_idx"],
        "end_idx": stats["end_idx"],
        "num_layers": stats["num_layers"],
        "device": stats["device"],
        "comm_dtype": stats["comm_dtype"],
        "prefill_seq_len": stats["prefill_seq_len"],
        "max_new_tokens": stats["max_new_tokens"],
        "prefill": _summarize_generate_phase_stats(stats["prefill"]),
        "steps": [_summarize_generate_phase_stats(step) for step in stats["steps"]],
        "generated_token_ids": stats["generated_token_ids"],
    }
    if stats.get("reference_generated_token_ids") is not None:
        summary["reference_generated_token_ids"] = stats["reference_generated_token_ids"]
        summary["token_match"] = stats["generated_token_ids"] == stats["reference_generated_token_ids"]

    if "prefill_output_tensor" in stats:
        summary["prefill_topk"] = summarize_last_token_topk(stats["prefill_output_tensor"], topk)
        reference_prefill = stats.get("reference_prefill_output_tensor")
        reference_steps = stats.get("reference_step_output_tensors")
        if (reference_prefill is None or reference_steps is None) and manifest.stages[-1].bundle_path:
            bundle_reference_prefill, bundle_reference_steps = _maybe_load_generate_reference_logits(
                manifest.stages[-1].bundle_path
            )
            if reference_prefill is None:
                reference_prefill = bundle_reference_prefill
            if reference_steps is None:
                reference_steps = bundle_reference_steps
        if reference_prefill is not None and reference_steps is not None:
            summary["reference_prefill_topk"] = summarize_last_token_topk(reference_prefill, topk)
            summary["step_topks"] = [
                {
                    "step_idx": step_idx,
                    "topk": summarize_last_token_topk(runtime_logits, topk),
                    "reference_topk": summarize_last_token_topk(reference_logits, topk),
                }
                for step_idx, (runtime_logits, reference_logits) in enumerate(
                    zip(stats["step_output_tensors"], reference_steps)
                )
            ]
        else:
            summary["step_topks"] = [
                {
                    "step_idx": step_idx,
                    "topk": summarize_last_token_topk(runtime_logits, topk),
                }
                for step_idx, runtime_logits in enumerate(stats["step_output_tensors"])
            ]
    return summary


def _summarize_hybrid_run(
    stats: dict,
    manifest,
    *,
    backend: str,
    topk: int,
    compare_direct: bool,
    trace_layers: bool,
    dump_layer: int | None,
) -> dict:
    debug_mode = compare_direct or trace_layers or dump_layer is not None
    if manifest.pipeline_type in GENERATE_PIPELINE_TYPES:
        summary = {
            "rank": stats["rank"],
            "backend": backend,
            "pipeline_type": manifest.pipeline_type,
            "stage_idx": stats["stage_idx"],
            "stage_ranks": stats["stage_ranks"],
            "local_rank": stats["local_rank"],
            "tp_degree": stats["tp_degree"],
            "leader_rank": stats["leader_rank"],
            "current_pp_group": stats["current_pp_group"],
            "debug_mode": debug_mode,
            "compare_direct": compare_direct,
            "trace_layers": trace_layers,
            "dump_layer": dump_layer,
            "dump_topk": topk,
            "prefill_seq_len": stats["prefill_seq_len"],
            "max_new_tokens": stats["max_new_tokens"],
            "prefill": _summarize_generate_phase_stats(stats["prefill"]),
            "steps": [_summarize_generate_phase_stats(step) for step in stats["steps"]],
            "generated_token_ids": stats["generated_token_ids"],
        }
        if stats.get("reference_generated_token_ids") is not None:
            summary["reference_generated_token_ids"] = stats["reference_generated_token_ids"]
            summary["token_match"] = stats["generated_token_ids"] == stats["reference_generated_token_ids"]
        if "prefill_output_tensor" in stats and stats["stage_idx"] == stats["num_stages"] - 1 and stats["local_rank"] == 0:
            summary["prefill_topk"] = summarize_last_token_topk(stats["prefill_output_tensor"], topk)
            reference_prefill = stats.get("reference_prefill_output_tensor")
            reference_steps = stats.get("reference_step_output_tensors")
            if (reference_prefill is None or reference_steps is None) and manifest.stages[-1].bundle_path:
                bundle_reference_prefill, bundle_reference_steps = _maybe_load_generate_reference_logits(
                    manifest.stages[-1].bundle_path
                )
                if reference_prefill is None:
                    reference_prefill = bundle_reference_prefill
                if reference_steps is None:
                    reference_steps = bundle_reference_steps
            if reference_prefill is not None and reference_steps is not None:
                summary["reference_prefill_topk"] = summarize_last_token_topk(reference_prefill, topk)
                summary["step_topks"] = [
                    {
                        "step_idx": step_idx,
                        "topk": summarize_last_token_topk(runtime_logits, topk),
                        "reference_topk": summarize_last_token_topk(reference_logits, topk),
                    }
                    for step_idx, (runtime_logits, reference_logits) in enumerate(
                        zip(stats["step_output_tensors"], reference_steps)
                    )
                ]
            else:
                summary["step_topks"] = [
                    {
                        "step_idx": step_idx,
                        "topk": summarize_last_token_topk(runtime_logits, topk),
                    }
                    for step_idx, runtime_logits in enumerate(stats["step_output_tensors"])
                ]
        return summary

    summary = {
        "rank": stats["rank"],
        "backend": backend,
        "pipeline_type": manifest.pipeline_type,
        "stage_idx": stats["stage_idx"],
        "stage_ranks": stats["stage_ranks"],
        "local_rank": stats["local_rank"],
        "tp_degree": stats["tp_degree"],
        "leader_rank": stats["leader_rank"],
        "current_pp_group": stats["current_pp_group"],
        "input_shape": list(stats["input_shape"]),
        "output_shape": list(stats["output_shape"]),
        "debug_mode": debug_mode,
        "compare_direct": compare_direct,
        "trace_layers": trace_layers,
        "dump_layer": dump_layer,
        "dump_topk": topk,
        "received_payload_keys": stats["received_payload_keys"],
        "sent_payload_keys": stats["sent_payload_keys"],
        "sent_tensor_shapes": _tensor_shape_map_to_json(stats["sent_tensor_shapes"]),
        "boundary_max_diff": stats["boundary_max_diff"],
        "boundary_mean_diff": stats["boundary_mean_diff"],
        "direct_max_diff": stats["direct_max_diff"],
        "direct_mean_diff": stats["direct_mean_diff"],
        "stage_max_diff": stats["stage_max_diff"],
        "stage_mean_diff": stats["stage_mean_diff"],
        "tp_direct_max_diff": stats["tp_direct_max_diff"],
        "tp_direct_mean_diff": stats["tp_direct_mean_diff"],
        "trace_summary": stats["trace_summary"],
        "num_traces": len(stats["traces"] or []),
        "outlier_dump": stats["outlier_dump"],
    }
    if (
        "stage_output" in stats
        and "reference_output" in stats
        and stats["stage_idx"] == stats["num_stages"] - 1
        and stats["local_rank"] == 0
    ):
        summary["last_stage_topk"] = summarize_last_token_topk(stats["stage_output"], topk)
        summary["reference_topk"] = summarize_last_token_topk(stats["reference_output"], topk)
    return summary


def _run_live(args: argparse.Namespace) -> None:
    LIVE_RUNNERS[(args.modality, args.mode)](_build_script_namespace(args))


def _run_bundle(args: argparse.Namespace) -> None:
    runners = BUNDLE_PREPARE_RUNNERS if args.action == "prepare" else BUNDLE_RUN_RUNNERS
    runners[(args.modality, args.mode)](_build_script_namespace(args))


def _run_pp_prepare(args: argparse.Namespace) -> None:
    manifest = PIPELINE_PREPARE_RUNNERS[(args.modality, args.mode)](**_build_prepare_kwargs(args))
    print(json.dumps(_summarize_prepare_manifest(manifest, args, include_hybrid_layout=False), ensure_ascii=False, indent=2))


def _run_pp(args: argparse.Namespace) -> None:
    rank, world_size = init_dist()
    device = get_device(args.device)
    if args.manifest_path is None:
        manifest = build_direct_pipeline_manifest(**_build_direct_manifest_kwargs(args))
    else:
        manifest = load_pipeline_manifest(args.manifest_path)
    if manifest.pipeline_type in GENERATE_PIPELINE_TYPES:
        stats = run_text_generate_pipeline_rank(
            rank=rank,
            world_size=world_size,
            manifest=manifest,
            device=device,
            compute_dtype_arg=args.compute_dtype,
            comm_dtype_arg=args.comm_dtype,
            return_tensors=rank == manifest.num_stages - 1,
        )
        summary = _summarize_pipeline_generate_run(stats, manifest, args.topk)
    else:
        stats = run_text_pipeline_rank(
            rank=rank,
            world_size=world_size,
            manifest=manifest,
            device=device,
            compute_dtype_arg=args.compute_dtype,
            comm_dtype_arg=args.comm_dtype,
            return_tensors=rank == manifest.num_stages - 1,
        )
        summary = _summarize_pipeline_run(stats, manifest, args.topk)
    summary = _attach_generated_texts(summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    dist.barrier()


def _run_hybrid_prepare(args: argparse.Namespace) -> None:
    manifest = HYBRID_PREPARE_RUNNERS[(args.modality, args.mode)](
        **_build_prepare_kwargs(args),
        tp_degrees=args.tp_degrees,
    )
    print(json.dumps(_summarize_prepare_manifest(manifest, args, include_hybrid_layout=True), ensure_ascii=False, indent=2))


def _run_tp_prepare(args: argparse.Namespace) -> None:
    manifest = HYBRID_PREPARE_RUNNERS[(args.modality, args.mode)](
        **_build_prepare_kwargs(args),
        tp_degrees=args.tp_degrees,
    )
    manifest.runtime = _normalize_tp_runtime_name(manifest.runtime)
    torch.save(manifest.to_dict(), args.manifest_path)
    print(json.dumps(_summarize_prepare_manifest(manifest, args, include_hybrid_layout=True), ensure_ascii=False, indent=2))


def _run_hybrid_family(args: argparse.Namespace, *, backend: str) -> None:
    if args.manifest_path is None:
        manifest = build_direct_hybrid_manifest(
            **_build_direct_manifest_kwargs(args),
            tp_degrees=args.tp_degrees,
            backend=backend,
        )
    else:
        manifest = load_hybrid_manifest(args.manifest_path)
    rank, world_size = init_dist()
    device = get_device(args.device)
    runner = TextHybridRunner(
        manifest=manifest,
        device=device,
        compute_dtype_arg=args.compute_dtype,
        comm_dtype_arg=args.comm_dtype,
        tp_attn_math_mode=args.tp_attn_math,
        tp_mlp_math_mode=args.tp_mlp_math,
        compare_direct=args.compare_direct,
        trace_layers=args.trace_layers,
        dump_layer=args.dump_layer,
        dump_topk=args.dump_topk,
        return_tensors=True,
    )
    stats = runner.run_rank(rank, world_size)
    summary = _summarize_hybrid_run(
        stats,
        manifest,
        backend=backend,
        topk=args.dump_topk,
        compare_direct=args.compare_direct,
        trace_layers=args.trace_layers,
        dump_layer=args.dump_layer,
    )
    summary = _attach_generated_texts(summary, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    dist.barrier()


def _run_hybrid(args: argparse.Namespace) -> None:
    _run_hybrid_family(args, backend="hybrid")


def _run_tp(args: argparse.Namespace) -> None:
    _run_hybrid_family(args, backend="tp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL unified runtime entrypoint for live/bundle/PP/TP/hybrid paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modality", choices=["text", "multimodal"], required=True)
    parser.add_argument("--mode", choices=["prefill", "decode", "generate"], required=True)
    parser.add_argument("--backend", choices=["hf", "live", "bundle", "pp", "tp", "hybrid"], required=True)
    parser.add_argument("--action", choices=["prepare", "run"], default="run")

    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sample-fps", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="请用中文简要介绍一下人工智能。")
    parser.add_argument("--decode-token-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)

    parser.add_argument("--bundle-path", type=str, default=None)
    parser.add_argument("--bundle-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--stage-ranges", nargs="+", default=DEFAULT_STAGE_RANGES.copy())
    parser.add_argument("--tp-degrees", type=int, nargs="+", default=DEFAULT_TP_DEGREES.copy())

    parser.add_argument("--save-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--comm-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument("--tp-attn-math", choices=["orig", "float32", "bfloat16"], default="orig")
    parser.add_argument("--tp-mlp-math", choices=["orig", "float32", "bfloat16"], default="orig")
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--dump-layer", type=int, default=None)
    parser.add_argument("--dump-topk", type=int, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--keep-special-tokens", action="store_true")
    parser.add_argument(
        "--clean-up-tokenization-spaces",
        action="store_true",
        default=False,
    )
    return parser


def run_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    if parser is None:
        parser = build_parser()
    _resolve_defaults(args)
    _validate_args(parser, args)

    if args.backend == "hf":
        _run_hf_generate(args)
    elif args.backend == "live":
        _run_live(args)
    elif args.backend == "bundle":
        _run_bundle(args)
    elif args.backend == "pp":
        if args.action == "prepare":
            _run_pp_prepare(args)
        else:
            _run_pp(args)
    elif args.backend == "hybrid":
        if args.action == "prepare":
            _run_hybrid_prepare(args)
        else:
            _run_hybrid(args)
    elif args.backend == "tp":
        if args.action == "prepare":
            _run_tp_prepare(args)
        else:
            _run_tp(args)
    else:
        parser.error(f"未知 backend={args.backend!r}")


@record
def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_args(args, parser)


if __name__ == "__main__":
    main()
