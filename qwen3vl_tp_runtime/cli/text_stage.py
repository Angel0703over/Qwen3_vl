"""CLI for preparing and replaying an early text stage with TP and trace dumps."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import TEXT_STAGE_BUNDLE_PATH, get_device, init_dist
from qwen3vl_tp_runtime.hexgen_core.stage import (
    get_stage_input,
    get_stage_output,
    run_stage,
    run_stage_tp,
    trace_stage,
    trace_stage_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl import (
    capture_text_stage_bundle,
    dtype_from_name,
    load_bundle,
    move_bundle,
    resolve_comm_dtype,
)


def run_prepare(args):
    bundle = capture_text_stage_bundle(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_frames=args.num_frames,
        bundle_path=args.bundle_path,
        save_dtype=args.save_dtype,
    )

    first_layer = bundle["layers"][0]
    print(f"[prepare] bundle saved to {args.bundle_path}")
    print(
        f"[prepare] start_idx={bundle['start_idx']} end_idx={bundle['end_idx']} "
        f"num_layers={len(bundle['layers'])} save_dtype={bundle['save_dtype']} "
        f"num_heads={first_layer['num_attention_heads']} num_kv_heads={first_layer['num_key_value_heads']} "
        f"head_dim={first_layer['head_dim']} attn_impl={first_layer['attn_implementation']}"
    )
    print(
        f"[prepare] layer_input_shape={tuple(bundle['layer_input'].shape)} "
        f"layer_output_shape={tuple(bundle['layer_output'].shape)} "
        f"attention_mask_shape={tuple(bundle['attention_mask'].shape)} "
        f"visual_pos_masks_shape={None if bundle['visual_pos_masks'] is None else tuple(bundle['visual_pos_masks'].shape)}"
    )
    print(
        f"[prepare] deepstack_layer_indices={bundle['deepstack_layer_indices']} "
        f"input_device={bundle['original_input_device']} output_device={bundle['original_output_device']}"
    )
    print(
        f"[prepare] sanity max_diff={bundle['sanity_max_diff']} "
        f"mean_diff={bundle['sanity_mean_diff']}"
    )


def tensor_diff_stats(lhs, rhs):
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def format_diff(name, lhs, rhs):
    max_diff, mean_diff = tensor_diff_stats(lhs, rhs)
    return f"{name} max_diff={max_diff} mean_diff={mean_diff}"


def print_stage_trace(bundle, rank, world_size, comm_dtype, tp_attn_math, tp_mlp_math):
    stage_input = get_stage_input(bundle)
    direct_traces = trace_stage(stage_input, bundle)
    tp_traces = trace_stage_tp(
        stage_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_attn_math_mode=tp_attn_math,
        tp_mlp_math_mode=tp_mlp_math,
    )

    for direct_trace, tp_trace in zip(direct_traces, tp_traces):
        layer_idx = direct_trace["layer_idx"]
        deepstack_applied = direct_trace["deepstack_applied"]

        print(
            f"[trace-layer] rank={rank} layer={layer_idx} deepstack={deepstack_applied} "
            f"{format_diff('layer_input', tp_trace['layer_input'], direct_trace['layer_input'])} "
            f"{format_diff('attn_output', tp_trace['attn_output'], direct_trace['attn_output'])} "
            f"{format_diff('mlp_output', tp_trace['mlp_output'], direct_trace['mlp_output'])} "
            f"{format_diff('layer_output', tp_trace['layer_output'], direct_trace['layer_output'])} "
            f"{format_diff('post_deepstack', tp_trace['post_deepstack'], direct_trace['post_deepstack'])}"
        )


def run_tp(args):
    rank, world_size = init_dist()
    device = get_device(args.device)

    bundle = load_bundle(args.bundle_path)
    compute_dtype_name = bundle["save_dtype"] if args.compute_dtype == "auto" else args.compute_dtype
    compute_dtype = dtype_from_name(compute_dtype_name)
    comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)
    bundle = move_bundle(bundle, device, compute_dtype)

    layer_input = get_stage_input(bundle)
    reference_output = get_stage_output(bundle)
    direct_output = run_stage(layer_input, bundle)
    tp_output = run_stage_tp(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_attn_math_mode=args.tp_attn_math,
        tp_mlp_math_mode=args.tp_mlp_math,
    )

    direct_max = (direct_output - reference_output).abs().max().item()
    direct_mean = (direct_output - reference_output).abs().mean().item()
    tp_ref_max = (tp_output - reference_output).abs().max().item()
    tp_ref_mean = (tp_output - reference_output).abs().mean().item()
    tp_direct_max = (tp_output - direct_output).abs().max().item()
    tp_direct_mean = (tp_output - direct_output).abs().mean().item()

    first_layer = bundle["layers"][0]
    local_q_heads = first_layer["num_attention_heads"] // world_size
    local_kv_heads = first_layer["num_key_value_heads"] // world_size

    print(
        f"[config] rank={rank} device={device} world_size={world_size} "
        f"start_idx={bundle['start_idx']} end_idx={bundle['end_idx']} num_layers={len(bundle['layers'])} "
        f"num_heads={first_layer['num_attention_heads']} num_kv_heads={first_layer['num_key_value_heads']} "
        f"local_q_heads={local_q_heads} local_kv_heads={local_kv_heads} "
        f"head_dim={first_layer['head_dim']} hidden_act={first_layer['hidden_act']} "
        f"attn_impl={first_layer.get('attn_implementation', 'unknown')} "
        f"deepstack_layer_indices={bundle['deepstack_layer_indices']} "
        f"tp_attn_math={args.tp_attn_math} tp_mlp_math={args.tp_mlp_math}"
    )
    print(
        f"[config] layer_input_shape={tuple(layer_input.shape)} layer_output_shape={tuple(reference_output.shape)} "
        f"dtype={layer_input.dtype} bundle_dtype={bundle['save_dtype']} "
        f"original_input_dtype={bundle['original_input_dtype']} original_input_device={bundle['original_input_device']} "
        f"comm_dtype={comm_dtype}"
    )
    print(
        f"[direct] rank={rank} direct_text_stage_vs_reference max_diff={direct_max} "
        f"mean_diff={direct_mean}"
    )
    print(
        f"[text-stage] rank={rank} tp_vs_reference max_diff={tp_ref_max} "
        f"mean_diff={tp_ref_mean} tp_vs_direct max_diff={tp_direct_max} mean_diff={tp_direct_mean}"
    )
    if args.trace_layers:
        print_stage_trace(bundle, rank, world_size, comm_dtype, args.tp_attn_math, args.tp_mlp_math)


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL early text stage TP runtime 原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取真实 text stage bundle。")
    prepare_parser.add_argument("--start-idx", type=int, default=0)
    prepare_parser.add_argument("--end-idx", type=int, default=2)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=TEXT_STAGE_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    tp_parser = subparsers.add_parser("tp", help="跑 early text stage TP forward。")
    tp_parser.add_argument("--bundle-path", type=str, default=TEXT_STAGE_BUNDLE_PATH)
    tp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--tp-attn-math", choices=["float32", "orig"], default="orig")
    tp_parser.add_argument("--tp-mlp-math", choices=["float32", "orig"], default="float32")
    tp_parser.add_argument("--trace-layers", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "tp":
        run_tp(args)


if __name__ == "__main__":
    main()
