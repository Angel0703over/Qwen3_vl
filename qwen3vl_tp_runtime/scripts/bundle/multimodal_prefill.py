"""Script for capturing and replaying multimodal direct/PP prefill logits bundles."""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import (
    FRAME_DIR,
    MODEL_PATH,
    MULTIMODAL_PREFILL_BUNDLE_PATH,
    MULTIMODAL_PREFILL_PIPELINE_BUNDLE_DIR,
    MULTIMODAL_PREFILL_PIPELINE_MANIFEST_PATH,
    get_device,
    init_dist,
    load_pipeline_manifest,
    parse_stage_ranges,
    prepare_multimodal_prefill_pipeline,
)
from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import run_text_pipeline_rank
from qwen3vl_tp_runtime.models.qwen3vl import (
    capture_multimodal_prefill_bundle,
    load_bundle,
    trace_text_prefill_logits,
)
from qwen3vl_tp_runtime.scripts.common import (
    load_runtime_bundle,
    release_unused_memory,
    summarize_last_token_topk,
    tensor_diff_stats,
)


def run_prepare(args) -> None:
    bundle = capture_multimodal_prefill_bundle(
        num_frames=args.num_frames,
        bundle_path=args.bundle_path,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
        frame_dir=args.frame_dir,
    )

    print(f"[prepare] bundle saved to {args.bundle_path}")
    print(
        f"[prepare] num_frames={bundle['num_frames']} "
        f"seq_len={bundle['input_ids'].shape[-1]} "
        f"num_layers={len(bundle['layers'])} "
        f"save_dtype={bundle['save_dtype']}"
    )
    print(
        f"[prepare] layer_input_shape={tuple(bundle['layer_input'].shape)} "
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
    bundle, compute_dtype = load_runtime_bundle(args.bundle_path, device, args.compute_dtype)

    stage_input = bundle["layer_input"]
    trace = trace_text_prefill_logits(stage_input, bundle)

    reference_stage_output = bundle["stage_output"]
    reference_norm_output = bundle["norm_output"]
    reference_logits = bundle["logits"]

    stage_max, stage_mean = tensor_diff_stats(trace["stage_output"], reference_stage_output)
    norm_max, norm_mean = tensor_diff_stats(trace["norm_output"], reference_norm_output)
    logits_max, logits_mean = tensor_diff_stats(trace["logits"], reference_logits)

    print(
        f"[run] device={device} compute_dtype={compute_dtype} "
        f"num_frames={bundle['num_frames']} seq_len={bundle['input_ids'].shape[-1]}"
    )
    print(
        f"[run] stage_input_shape={tuple(stage_input.shape)} "
        f"logits_shape={tuple(trace['logits'].shape)}"
    )
    print(f"[run] stage_vs_reference max_diff={stage_max} mean_diff={stage_mean}")
    print(f"[run] norm_vs_reference max_diff={norm_max} mean_diff={norm_mean}")
    print(f"[run] logits_vs_reference max_diff={logits_max} mean_diff={logits_mean}")
    print(f"[run] direct_last_token_topk={summarize_last_token_topk(trace['logits'], args.topk)}")
    print(f"[run] reference_last_token_topk={summarize_last_token_topk(reference_logits, args.topk)}")


def run_prepare_pp(args) -> None:
    stage_ranges = parse_stage_ranges(args.stage_ranges)
    manifest = prepare_multimodal_prefill_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        num_frames=args.num_frames,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
        frame_dir=args.frame_dir,
    )
    release_unused_memory()

    stage0_bundle = load_bundle(manifest.stages[0].bundle_path)
    num_frames = int(stage0_bundle["num_frames"])
    seq_len = int(stage0_bundle["input_ids"].shape[-1])
    stage0_input_shape = tuple(stage0_bundle["stage_input"].shape)
    stage0_sanity_max_diff = stage0_bundle["sanity_max_diff"]
    stage0_sanity_mean_diff = stage0_bundle["sanity_mean_diff"]
    del stage0_bundle
    release_unused_memory()

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
    release_unused_memory()

    print(f"[prepare-pp] manifest saved to {args.manifest_path}")
    print(f"[prepare-pp] bundle_dir={args.bundle_dir}")
    print(
        f"[prepare-pp] num_frames={num_frames} "
        f"seq_len={seq_len} "
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
    parser = argparse.ArgumentParser(description="Qwen3-VL multimodal direct/PP prefill logits baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取 multimodal prefill logits bundle。")
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    prepare_parser.add_argument("--bundle-path", type=str, default=MULTIMODAL_PREFILL_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    prepare_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_parser.add_argument("--topk", type=int, default=5)

    run_parser = subparsers.add_parser("run", help="运行 multimodal direct prefill logits replay。")
    run_parser.add_argument("--bundle-path", type=str, default=MULTIMODAL_PREFILL_BUNDLE_PATH)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    run_parser.add_argument("--topk", type=int, default=5)

    prepare_pp_parser = subparsers.add_parser("prepare-pp", help="抓取 multimodal PP prefill logits pipeline。")
    prepare_pp_parser.add_argument("--num-frames", type=int, default=8)
    prepare_pp_parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    prepare_pp_parser.add_argument("--stage-ranges", nargs="+", default=["0:17", "18:35"])
    prepare_pp_parser.add_argument("--bundle-dir", type=str, default=MULTIMODAL_PREFILL_PIPELINE_BUNDLE_DIR)
    prepare_pp_parser.add_argument("--manifest-path", type=str, default=MULTIMODAL_PREFILL_PIPELINE_MANIFEST_PATH)
    prepare_pp_parser.add_argument(
        "--save-dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
    )
    prepare_pp_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_pp_parser.add_argument("--topk", type=int, default=5)

    pp_parser = subparsers.add_parser("pp", help="运行 multimodal PP prefill logits replay。")
    pp_parser.add_argument("--manifest-path", type=str, default=MULTIMODAL_PREFILL_PIPELINE_MANIFEST_PATH)
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
