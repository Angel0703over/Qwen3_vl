import argparse
import sys
from pathlib import Path

import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.core.config import TEXT_HYBRID_BUNDLE_DIR, TEXT_HYBRID_MANIFEST_PATH
from qwen3vl_tp_runtime.core.dist import get_device, init_dist
from qwen3vl_tp_runtime.core.hybrid import (
    load_hybrid_manifest,
    parse_tp_degrees,
    prepare_text_hybrid,
    run_text_hybrid_rank,
)
from qwen3vl_tp_runtime.core.pipeline import parse_stage_ranges


def format_trace_stat(name, stats):
    return f"{name} max_diff={stats[0]} mean_diff={stats[1]}"


def print_stage_traces(rank: int, traces: list[dict] | None) -> None:
    if traces is None:
        return
    for trace in traces:
        print(
            f"[trace-direct] rank={rank} layer={trace['layer_idx']} deepstack={trace['deepstack_applied']} "
            f"{format_trace_stat('layer_input', trace['direct_vs_ref']['layer_input'])} "
            f"{format_trace_stat('attn_output', trace['direct_vs_ref']['attn_output'])} "
            f"{format_trace_stat('gate_out', trace['direct_vs_ref']['gate_out'])} "
            f"{format_trace_stat('up_out', trace['direct_vs_ref']['up_out'])} "
            f"{format_trace_stat('fused', trace['direct_vs_ref']['fused'])} "
            f"{format_trace_stat('mlp_output', trace['direct_vs_ref']['mlp_output'])} "
            f"{format_trace_stat('layer_output', trace['direct_vs_ref']['layer_output'])} "
            f"{format_trace_stat('post_deepstack', trace['direct_vs_ref']['post_deepstack'])}"
        )
        print(
            f"[trace-tp] rank={rank} layer={trace['layer_idx']} deepstack={trace['deepstack_applied']} "
            f"{format_trace_stat('layer_input', trace['tp_vs_direct']['layer_input'])} "
            f"{format_trace_stat('attn_output', trace['tp_vs_direct']['attn_output'])} "
            f"{format_trace_stat('gate_out', trace['tp_vs_direct']['gate_out'])} "
            f"{format_trace_stat('up_out', trace['tp_vs_direct']['up_out'])} "
            f"{format_trace_stat('fused', trace['tp_vs_direct']['fused'])} "
            f"{format_trace_stat('mlp_output', trace['tp_vs_direct']['mlp_output'])} "
            f"{format_trace_stat('layer_output', trace['tp_vs_direct']['layer_output'])} "
            f"{format_trace_stat('post_deepstack', trace['tp_vs_direct']['post_deepstack'])}"
        )


def print_outlier_dump(rank: int, outlier_dump: dict | None) -> None:
    if outlier_dump is None:
        return

    layer_idx = outlier_dump["layer_idx"]
    deepstack_applied = outlier_dump["deepstack_applied"]
    for tensor_name, entries in outlier_dump["direct_vs_ref"].items():
        for entry in entries:
            print(
                f"[outlier-direct] rank={rank} layer={layer_idx} deepstack={deepstack_applied} "
                f"tensor={tensor_name} index={entry['index']} ref={entry['rhs']} "
                f"direct={entry['lhs']} abs_diff={entry['abs_diff']}"
            )
    for tensor_name, entries in outlier_dump["tp_vs_direct"].items():
        for entry in entries:
            print(
                f"[outlier-tp] rank={rank} layer={layer_idx} deepstack={deepstack_applied} "
                f"tensor={tensor_name} index={entry['index']} direct={entry['rhs']} "
                f"tp={entry['lhs']} abs_diff={entry['abs_diff']}"
            )


def run_prepare(args):
    stage_ranges = parse_stage_ranges(args.ranges)
    tp_degrees = parse_tp_degrees(args.tp)
    manifest = prepare_text_hybrid(
        stage_ranges=stage_ranges,
        tp_degrees=tp_degrees,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        num_frames=args.num_frames,
        save_dtype=args.save_dtype,
    )

    print(f"[prepare] manifest saved to {args.manifest_path}")
    print(
        f"[prepare] num_stages={manifest['num_stages']} world_size={manifest['world_size']} "
        f"stage_ranges={manifest['stage_ranges']} tp_degrees={manifest['tp_degrees']}"
    )
    for stage_meta, rank_group in zip(manifest["stages"], manifest["stage_rank_groups"]):
        print(
            f"[prepare-stage] stage={stage_meta['stage_idx']} "
            f"start_idx={stage_meta['start_idx']} end_idx={stage_meta['end_idx']} "
            f"num_layers={stage_meta['num_layers']} ranks={rank_group}"
        )
    for boundary in manifest["boundaries"]:
        print(
            f"[prepare-boundary] src_stage={boundary['src_stage_idx']} dst_stage={boundary['dst_stage_idx']} "
            f"max_diff={boundary['max_diff']} mean_diff={boundary['mean_diff']}"
        )


def run_hybrid(args):
    rank, world_size = init_dist()
    device = get_device(args.device)
    manifest = load_hybrid_manifest(args.manifest_path)

    stats = run_text_hybrid_rank(
        rank=rank,
        world_size=world_size,
        manifest=manifest,
        device=device,
        compute_dtype_arg=args.compute_dtype,
        comm_dtype_arg=args.comm_dtype,
        tp_attn_math_mode=args.tp_attn_math,
        tp_mlp_math_mode=args.tp_mlp_math,
        compare_direct=args.compare_direct or args.trace_layers or args.dump_layer is not None,
        trace_layers=args.trace_layers,
        dump_layer=args.dump_layer,
        dump_topk=args.dump_topk,
    )

    print(
        f"[config] rank={stats['rank']} device={stats['device']} "
        f"stage={stats['stage_idx']}/{stats['num_stages'] - 1} stage_ranks={stats['stage_ranks']} "
        f"local_rank={stats['local_rank']} tp_degree={stats['tp_degree']} "
        f"start_idx={stats['start_idx']} end_idx={stats['end_idx']} "
        f"num_layers={stats['num_layers']} comm_dtype={stats['comm_dtype']} "
        f"tp_attn_math={stats['tp_attn_math_mode']} tp_mlp_math={stats['tp_mlp_math_mode']}"
    )
    print(
        f"[boundary] rank={stats['rank']} input_shape={stats['input_shape']} "
        f"max_diff={stats['boundary_max_diff']} mean_diff={stats['boundary_mean_diff']}"
    )
    if stats["direct_max_diff"] is not None:
        print(
            f"[direct] rank={stats['rank']} direct_vs_reference max_diff={stats['direct_max_diff']} "
            f"mean_diff={stats['direct_mean_diff']}"
        )
    print(
        f"[stage] rank={stats['rank']} output_shape={stats['output_shape']} "
        f"tp_vs_reference max_diff={stats['stage_max_diff']} mean_diff={stats['stage_mean_diff']}"
    )
    if stats["tp_direct_max_diff"] is not None:
        print(
            f"[stage] rank={stats['rank']} tp_vs_direct max_diff={stats['tp_direct_max_diff']} "
            f"mean_diff={stats['tp_direct_mean_diff']}"
        )
    if stats["sent_shape"] is not None:
        print(
            f"[send] rank={stats['rank']} dst={stats['next_leader_rank']} "
            f"sent_shape={stats['sent_shape']}"
        )
    if args.trace_layers:
        print_stage_traces(stats["rank"], stats["traces"])
    if args.dump_layer is not None:
        print_outlier_dump(stats["rank"], stats["outlier_dump"])

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL text stage 的最小 PP+TP 混合原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取 hybrid text pipeline 的 manifest 和 bundle。")
    prepare_parser.add_argument("--ranges", nargs="+", required=True, help="例如 0:5 6:11")
    prepare_parser.add_argument("--tp", nargs="+", required=True, type=int, help="例如 2 2")
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-dir", type=str, default=TEXT_HYBRID_BUNDLE_DIR)
    prepare_parser.add_argument("--manifest-path", type=str, default=TEXT_HYBRID_MANIFEST_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    run_parser = subparsers.add_parser("run", help="按 hybrid manifest 跑 PP+TP 原型。")
    run_parser.add_argument("--manifest-path", type=str, default=TEXT_HYBRID_MANIFEST_PATH)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    run_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    run_parser.add_argument(
        "--tp-attn-math",
        choices=["float32", "orig"],
        default="orig",
        help="TP attention 本地数学计算使用的 dtype 策略。",
    )
    run_parser.add_argument(
        "--tp-mlp-math",
        choices=["float32", "orig"],
        default="float32",
        help="TP MLP 本地数学计算使用的 dtype 策略。",
    )
    run_parser.add_argument("--compare-direct", action="store_true")
    run_parser.add_argument("--trace-layers", action="store_true")
    run_parser.add_argument("--dump-layer", type=int, default=None)
    run_parser.add_argument("--dump-topk", type=int, default=5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "run":
        run_hybrid(args)


if __name__ == "__main__":
    main()
