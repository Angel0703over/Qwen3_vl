"""Unified runtime CLI for direct runtime execution plus manifest-replay debug paths."""

from __future__ import annotations

import argparse
import json
import sys
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
    get_device,
    init_dist,
    run_text_generate_pipeline_rank,
    run_text_pipeline_rank,
)
from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import TextHybridRunner
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs,
    build_text_inputs,
    list_frames,
    load_model,
    load_processor,
)
from qwen3vl_tp_runtime.scripts.runtime_cli import (
    _emit_debug_path_warnings,
    _load_hybrid_manifest_for_args,
    _load_pipeline_manifest_for_args,
    _require_debug_path_opt_in,
    _resolve_defaults,
    _validate_args,
)
from qwen3vl_tp_runtime.scripts.runtime_summary import (
    _attach_generated_texts,
    _summarize_hybrid_run,
    _summarize_pipeline_generate_run,
    _summarize_pipeline_run,
)
from qwen3vl_tp_runtime.scripts.live import live_multimodal_runtime


LIVE_RUNNERS = {
    ("multimodal", "prefill"): live_multimodal_runtime.run_prefill,
    ("multimodal", "decode"): live_multimodal_runtime.run_decode,
    ("multimodal", "generate"): live_multimodal_runtime.run_generate,
}

GENERATE_PIPELINE_TYPES = {"text_generate", "multimodal_generate"}
DEFAULT_STAGE_RANGES = ["0:17", "18:35"]
DEFAULT_TP_DEGREES = [2, 2]
TP_SINGLE_STAGE_RANGES = ["0:35"]
TP_SINGLE_STAGE_DEGREES = [2]


def _build_script_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        prompt=args.prompt,
        num_frames=args.num_frames,
        frame_dir=args.frame_dir,
        decode_token_id=args.decode_token_id,
        max_new_tokens=args.max_new_tokens,
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


def _run_live(args: argparse.Namespace) -> None:
    LIVE_RUNNERS[(args.modality, args.mode)](_build_script_namespace(args))


def _run_pp(args: argparse.Namespace) -> None:
    rank, world_size = init_dist()
    device = get_device(args.device)
    manifest = _load_pipeline_manifest_for_args(args)
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


def _run_hybrid_family(args: argparse.Namespace, *, backend: str) -> None:
    manifest = _load_hybrid_manifest_for_args(args, backend=backend)
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
        description=(
            "Qwen3-VL unified runtime entrypoint.\n"
            "Recommended main path: backend=pp|tp|hybrid "
            "(directly builds runtime state from model_path).\n"
            "--manifest-path replay is kept for debug/regression workflows and "
            "requires --allow-debug-paths."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modality", choices=["text", "multimodal"], required=True)
    parser.add_argument("--mode", choices=["prefill", "decode", "generate"], required=True)
    parser.add_argument(
        "--backend",
        choices=["hf", "live", "pp", "tp", "hybrid"],
        required=True,
        help=(
            "Execution backend. Recommended main runtimes are pp/tp/hybrid. "
            "hf/live are auxiliary paths."
        ),
    )

    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sample-fps", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="请用中文简要介绍一下人工智能。")
    parser.add_argument("--decode-token-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)

    parser.add_argument(
        "--allow-debug-paths",
        action="store_true",
        help=(
            "Acknowledge use of debug-only runtime paths such as "
            "--manifest-path replay."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help=(
            "Prepared manifest replay path for debug-only runs. Leave unset on the "
            "main run path to build runtime state directly from model_path."
        ),
    )
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
    _require_debug_path_opt_in(parser, args)
    _validate_args(parser, args)
    _emit_debug_path_warnings(args)

    if args.backend == "hf":
        _run_hf_generate(args)
    elif args.backend == "live":
        _run_live(args)
    elif args.backend == "pp":
        _run_pp(args)
    elif args.backend == "hybrid":
        _run_hybrid(args)
    elif args.backend == "tp":
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
