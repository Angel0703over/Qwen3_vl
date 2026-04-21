"""Script for capturing and replaying multimodal direct/PP decode logits bundles with KV cache."""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import (
    FRAME_DIR,
    MODEL_PATH,
    MULTIMODAL_DECODE_BUNDLE_PATH,
    MULTIMODAL_DECODE_PIPELINE_BUNDLE_DIR,
    MULTIMODAL_DECODE_PIPELINE_MANIFEST_PATH,
    get_device,
    init_dist,
    load_pipeline_manifest,
    parse_stage_ranges,
    prepare_multimodal_decode_pipeline,
)
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import run_text_pipeline_rank
from qwen3vl_tp_runtime.models.qwen3vl import (
    capture_multimodal_decode_bundle,
    dtype_from_name,
    forward_text_embeddings,
    load_bundle,
    move_bundle,
    trace_text_decode_logits,
)


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def summarize_last_token_topk(logits: torch.Tensor, topk: int) -> list[dict]:
    last_token_logits = logits[0, -1].to(torch.float32)
    k = min(topk, last_token_logits.numel())
    values, indices = torch.topk(last_token_logits, k=k)
    return [
        {
            "token_id": int(token_id),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def _release_unused_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_decode_bundle(bundle_path: str, device: torch.device, compute_dtype_arg: str) -> tuple[dict, torch.dtype]:
    bundle = load_bundle(bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def run_prepare(args) -> None:
    bundle = capture_multimodal_decode_bundle(
        num_frames=args.num_frames,
        decode_token_id=args.decode_token_id,
        bundle_path=args.bundle_path,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
        frame_dir=args.frame_dir,
    )

    print(f"[prepare] bundle saved to {args.bundle_path}")
    print(
        f"[prepare] num_frames={bundle['num_frames']} "
        f"prefill_seq_len={bundle['prefill_seq_len']} "
        f"total_seq_len={bundle['total_seq_len']} "
        f"decode_token_id={bundle['decode_token_id']} "
        f"decode_source={bundle['decode_source']!r} "
        f"num_layers={len(bundle['layers'])} "
        f"save_dtype={bundle['save_dtype']}"
    )
    print(
        f"[prepare] decode_input_shape={tuple(bundle['decode_input_ids'].shape)} "
        f"layer_input_shape={tuple(bundle['layer_input'].shape)} "
        f"stage_output_shape={tuple(bundle['stage_output'].shape)} "
        f"logits_shape={tuple(bundle['logits'].shape)}"
    )
    print(
        f"[prepare] stage_sanity max_diff={bundle['stage_sanity_max_diff']} "
        f"mean_diff={bundle['stage_sanity_mean_diff']}"
    )
    print(
        f"[prepare] norm_sanity max_diff={bundle['norm_sanity_max_diff']} "
        f"mean_diff={bundle['norm_sanity_mean_diff']}"
    )
    print(
        f"[prepare] logits_sanity max_diff={bundle['logits_sanity_max_diff']} "
        f"mean_diff={bundle['logits_sanity_mean_diff']}"
    )
    print(f"[prepare] last_token_topk={summarize_last_token_topk(bundle['logits'], args.topk)}")


def run_direct(args) -> None:
    device = get_device(args.device)
    bundle, compute_dtype = load_decode_bundle(args.bundle_path, device, args.compute_dtype)

    decode_input_ids = bundle["decode_input_ids"]
    embedded_input = forward_text_embeddings(decode_input_ids, bundle)
    trace = trace_text_decode_logits(embedded_input, bundle)

    reference_layer_input = bundle["layer_input"]
    reference_stage_output = bundle["stage_output"]
    reference_norm_output = bundle["norm_output"]
    reference_logits = bundle["logits"]

    embedding_max, embedding_mean = tensor_diff_stats(embedded_input, reference_layer_input)
    stage_max, stage_mean = tensor_diff_stats(trace["stage_output"], reference_stage_output)
    norm_max, norm_mean = tensor_diff_stats(trace["norm_output"], reference_norm_output)
    logits_max, logits_mean = tensor_diff_stats(trace["logits"], reference_logits)

    print(
        f"[run] device={device} compute_dtype={compute_dtype} "
        f"num_frames={bundle['num_frames']} prefill_seq_len={bundle['prefill_seq_len']} "
        f"total_seq_len={bundle['total_seq_len']} decode_token_id={bundle['decode_token_id']}"
    )
    print(
        f"[run] decode_input_shape={tuple(decode_input_ids.shape)} "
        f"embedded_input_shape={tuple(embedded_input.shape)} "
        f"logits_shape={tuple(trace['logits'].shape)}"
    )
    print(f"[run] embedding_vs_reference max_diff={embedding_max} mean_diff={embedding_mean}")
    print(f"[run] stage_vs_reference max_diff={stage_max} mean_diff={stage_mean}")
    print(f"[run] norm_vs_reference max_diff={norm_max} mean_diff={norm_mean}")
    print(f"[run] logits_vs_reference max_diff={logits_max} mean_diff={logits_mean}")
    print(f"[run] direct_last_token_topk={summarize_last_token_topk(trace['logits'], args.topk)}")
    print(f"[run] reference_last_token_topk={summarize_last_token_topk(reference_logits, args.topk)}")


def run_prepare_pp(args) -> None:
    stage_ranges = parse_stage_ranges(args.stage_ranges)
    manifest = prepare_multimodal_decode_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        num_frames=args.num_frames,
        decode_token_id=args.decode_token_id,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
        frame_dir=args.frame_dir,
    )
    _release_unused_memory()

    stage0_bundle = load_bundle(manifest.stages[0].bundle_path)
    num_frames = int(stage0_bundle["num_frames"])
    prefill_seq_len = int(stage0_bundle["prefill_seq_len"])
    total_seq_len = int(stage0_bundle["total_seq_len"])
    decode_token_id = int(stage0_bundle["decode_token_id"])
    decode_source = str(stage0_bundle["decode_source"])
    stage0_input_shape = tuple(stage0_bundle["stage_input"].shape)
    stage0_sanity_max_diff = stage0_bundle["sanity_max_diff"]
    stage0_sanity_mean_diff = stage0_bundle["sanity_mean_diff"]
    del stage0_bundle
    _release_unused_memory()

    last_bundle = load_bundle(manifest.stages[-1].bundle_path)
    last_output_shape = tuple(last_bundle["stage_output"].shape)
    last_hidden_sanity = None
    if "hidden_stage_sanity_max_diff" in last_bundle:
        last_hidden_sanity = (
            last_bundle["hidden_stage_sanity_max_diff"],
            last_bundle["hidden_stage_sanity_mean_diff"],
        )
    last_norm_sanity = None
    if "norm_sanity_max_diff" in last_bundle:
        last_norm_sanity = (
            last_bundle["norm_sanity_max_diff"],
            last_bundle["norm_sanity_mean_diff"],
        )
    last_logits_sanity = None
    reference_last_token_topk = None
    if "logits_sanity_max_diff" in last_bundle:
        last_logits_sanity = (
            last_bundle["logits_sanity_max_diff"],
            last_bundle["logits_sanity_mean_diff"],
        )
        reference_last_token_topk = summarize_last_token_topk(last_bundle["logits"], args.topk)
    del last_bundle
    _release_unused_memory()

    print(f"[prepare-pp] manifest saved to {args.manifest_path}")
    print(f"[prepare-pp] bundle_dir={args.bundle_dir}")
    print(
        f"[prepare-pp] num_frames={num_frames} "
        f"prefill_seq_len={prefill_seq_len} "
        f"total_seq_len={total_seq_len} "
        f"decode_token_id={decode_token_id} "
        f"decode_source={decode_source!r} "
        f"num_stages={manifest.num_stages} "
        f"stage_ranges={manifest.stage_ranges} "
        f"save_dtype={manifest.save_dtype}"
    )
    print(
        f"[prepare-pp] stage0_input_shape={stage0_input_shape} "
        f"last_output_shape={last_output_shape}"
    )
    for boundary in manifest.boundaries:
        print(
            f"[prepare-pp] boundary stage{boundary.src_stage_idx}->stage{boundary.dst_stage_idx} "
            f"max_diff={boundary.max_diff} mean_diff={boundary.mean_diff}"
        )
    print(
        f"[prepare-pp] stage0_sanity max_diff={stage0_sanity_max_diff} "
        f"mean_diff={stage0_sanity_mean_diff}"
    )
    if last_hidden_sanity is not None:
        print(
            f"[prepare-pp] last_hidden_sanity max_diff={last_hidden_sanity[0]} "
            f"mean_diff={last_hidden_sanity[1]}"
        )
    if last_norm_sanity is not None:
        print(
            f"[prepare-pp] last_norm_sanity max_diff={last_norm_sanity[0]} "
            f"mean_diff={last_norm_sanity[1]}"
        )
    if last_logits_sanity is not None and reference_last_token_topk is not None:
        print(
            f"[prepare-pp] last_logits_sanity max_diff={last_logits_sanity[0]} "
            f"mean_diff={last_logits_sanity[1]}"
        )
        print(f"[prepare-pp] reference_last_token_topk={reference_last_token_topk}")


def run_pp(args) -> None:
    rank, world_size = init_dist()
    device = get_device(args.device)
    manifest = load_pipeline_manifest(args.manifest_path)
    stats = run_text_pipeline_rank(
        rank=rank,
        world_size=world_size,
        manifest=manifest,
        device=device,
        compute_dtype_arg=args.compute_dtype,
        comm_dtype_arg=args.comm_dtype,
        return_tensors=rank == world_size - 1,
    )

    print(
        f"[config] rank={rank} device={device} world_size={world_size} "
        f"pipeline_type={manifest.pipeline_type} start_idx={stats['start_idx']} "
        f"end_idx={stats['end_idx']} num_layers={stats['num_layers']} "
        f"comm_dtype={stats['comm_dtype']}"
    )
    if stats["boundary_max_diff"] is not None:
        print(
            f"[handoff] rank={rank} input_shape={stats['input_shape']} "
            f"boundary_max_diff={stats['boundary_max_diff']} "
            f"boundary_mean_diff={stats['boundary_mean_diff']} "
            f"payload_keys={stats['received_payload_keys']}"
        )
    else:
        print(f"[handoff] rank={rank} input_shape={stats['input_shape']} payload_keys={stats['received_payload_keys']}")
    print(
        f"[stage] rank={rank} output_shape={stats['output_shape']} "
        f"pp_vs_reference max_diff={stats['stage_max_diff']} "
        f"mean_diff={stats['stage_mean_diff']}"
    )
    if stats["sent_shape"] is not None:
        print(
            f"[send] rank={rank} sent_shape={stats['sent_shape']} "
            f"payload_keys={stats['sent_payload_keys']} tensor_shapes={stats['sent_tensor_shapes']}"
        )

    if rank == world_size - 1:
        stage_output = stats.pop("stage_output")
        reference_output = stats.pop("reference_output")
        print(f"[pp] last_stage_topk={summarize_last_token_topk(stage_output, args.topk)}")
        print(f"[pp] reference_topk={summarize_last_token_topk(reference_output, args.topk)}")

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL multimodal direct/PP decode logits baseline with KV cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取 multimodal decode logits bundle。")
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--decode-token-id", type=int, default=None)
    prepare_parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    prepare_parser.add_argument("--bundle-path", type=str, default=MULTIMODAL_DECODE_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    prepare_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_parser.add_argument("--topk", type=int, default=5)

    run_parser = subparsers.add_parser("run", help="运行 multimodal direct decode logits replay。")
    run_parser.add_argument("--bundle-path", type=str, default=MULTIMODAL_DECODE_BUNDLE_PATH)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    run_parser.add_argument("--topk", type=int, default=5)

    prepare_pp_parser = subparsers.add_parser("prepare-pp", help="抓取 multimodal PP decode logits pipeline。")
    prepare_pp_parser.add_argument("--num-frames", type=int, default=8)
    prepare_pp_parser.add_argument("--decode-token-id", type=int, default=None)
    prepare_pp_parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    prepare_pp_parser.add_argument("--stage-ranges", nargs="+", default=["0:17", "18:35"])
    prepare_pp_parser.add_argument("--bundle-dir", type=str, default=MULTIMODAL_DECODE_PIPELINE_BUNDLE_DIR)
    prepare_pp_parser.add_argument("--manifest-path", type=str, default=MULTIMODAL_DECODE_PIPELINE_MANIFEST_PATH)
    prepare_pp_parser.add_argument(
        "--save-dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
    )
    prepare_pp_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_pp_parser.add_argument("--topk", type=int, default=5)

    pp_parser = subparsers.add_parser("pp", help="运行 multimodal PP decode logits replay。")
    pp_parser.add_argument("--manifest-path", type=str, default=MULTIMODAL_DECODE_PIPELINE_MANIFEST_PATH)
    pp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pp_parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    pp_parser.add_argument("--comm-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="float32")
    pp_parser.add_argument("--topk", type=int, default=5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "run":
        run_direct(args)
    elif args.command == "prepare-pp":
        run_prepare_pp(args)
    elif args.command == "pp":
        run_pp(args)
    else:
        parser.error(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
