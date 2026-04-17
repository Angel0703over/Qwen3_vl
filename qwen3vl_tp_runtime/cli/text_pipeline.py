"""CLI for preparing and executing a plain sequential text pipeline prototype."""

import argparse
import sys
from pathlib import Path

import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import (
    TEXT_PIPELINE_BUNDLE_DIR,
    TEXT_PIPELINE_MANIFEST_PATH,
    get_device,
    init_dist,
    load_pipeline_manifest,
    parse_stage_ranges,
    prepare_text_pipeline,
    run_text_pipeline_rank,
)


def run_prepare(args):
    stage_ranges = parse_stage_ranges(args.ranges)
    manifest = prepare_text_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        num_frames=args.num_frames,
        save_dtype=args.save_dtype,
    )

    print(f"[prepare] manifest saved to {args.manifest_path}")
    print(
        f"[prepare] num_stages={manifest.num_stages} "
        f"stage_ranges={manifest.stage_ranges} save_dtype={manifest.save_dtype}"
    )
    for stage_meta in manifest.stages:
        print(
            f"[prepare-stage] stage={stage_meta.stage_idx} "
            f"start_idx={stage_meta.start_idx} end_idx={stage_meta.end_idx} "
            f"num_layers={stage_meta.num_layers} bundle_path={stage_meta.bundle_path}"
        )
    for boundary in manifest.boundaries:
        print(
            f"[prepare-boundary] src_stage={boundary.src_stage_idx} dst_stage={boundary.dst_stage_idx} "
            f"max_diff={boundary.max_diff} mean_diff={boundary.mean_diff}"
        )


def run_pp(args):
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
    )

    print(
        f"[config] rank={stats['rank']} device={stats['device']} "
        f"stage={stats['stage_idx']}/{stats['num_stages'] - 1} "
        f"start_idx={stats['start_idx']} end_idx={stats['end_idx']} "
        f"num_layers={stats['num_layers']} comm_dtype={stats['comm_dtype']}"
    )
    if stats["boundary_max_diff"] is not None:
        print(
            f"[boundary] rank={stats['rank']} input_shape={stats['input_shape']} "
            f"max_diff={stats['boundary_max_diff']} mean_diff={stats['boundary_mean_diff']}"
        )
    print(
        f"[stage] rank={stats['rank']} output_shape={stats['output_shape']} "
        f"max_diff={stats['stage_max_diff']} mean_diff={stats['stage_mean_diff']}"
    )
    if stats["sent_shape"] is not None:
        print(
            f"[send] rank={stats['rank']} dst={stats['rank'] + 1} "
            f"sent_shape={stats['sent_shape']}"
        )

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL 多段 text pipeline 原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取多段连续 text stage 的 bundle。")
    prepare_parser.add_argument("--ranges", nargs="+", required=True, help="例如 0:5 6:11 12:17")
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-dir", type=str, default=TEXT_PIPELINE_BUNDLE_DIR)
    prepare_parser.add_argument("--manifest-path", type=str, default=TEXT_PIPELINE_MANIFEST_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    pp_parser = subparsers.add_parser("pp", help="按 rank 顺序跑多段 text pipeline handoff。")
    pp_parser.add_argument("--manifest-path", type=str, default=TEXT_PIPELINE_MANIFEST_PATH)
    pp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    pp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "pp":
        run_pp(args)


if __name__ == "__main__":
    main()
