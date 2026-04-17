import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.core.capture import capture_layer_range_bundle, load_bundle, move_bundle
from qwen3vl_tp_runtime.core.config import LAYER_RANGE_BUNDLE_PATH
from qwen3vl_tp_runtime.core.forward import (
    compose_layer_bundle,
    forward_layer_range,
    forward_layer_range_tp,
    trace_decoder_layer,
    trace_decoder_layer_tp,
)
from qwen3vl_tp_runtime.core.dist import get_device, init_dist
from qwen3vl_tp_runtime.core.ops import dtype_from_name, resolve_comm_dtype


def run_prepare(args):
    bundle = capture_layer_range_bundle(
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
        f"attention_mask_shape={tuple(bundle['attention_mask'].shape)}"
    )
    print(
        f"[prepare] input_device={bundle['original_input_device']} "
        f"output_device={bundle['original_output_device']}"
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


def print_layer_trace(bundle, rank, world_size, comm_dtype):
    direct_hidden = bundle["layer_input"]
    tp_hidden = bundle["layer_input"]

    for layer_bundle in bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, bundle)
        direct_trace = trace_decoder_layer(direct_hidden, runtime_bundle)
        tp_trace = trace_decoder_layer_tp(tp_hidden, runtime_bundle, rank, world_size, comm_dtype)
        layer_idx = layer_bundle["layer_idx"]

        print(
            f"[trace-attn] rank={rank} layer={layer_idx} "
            f"{format_diff('layer_input', tp_trace['layer_input'], direct_trace['layer_input'])} "
            f"{format_diff('attn_input', tp_trace['attn_input'], direct_trace['attn_input'])} "
            f"{format_diff('attn_output', tp_trace['attn_output'], direct_trace['attn_output'])} "
            f"{format_diff('after_attn', tp_trace['after_attn'], direct_trace['after_attn'])}"
        )
        print(
            f"[trace-mlp] rank={rank} layer={layer_idx} "
            f"{format_diff('mlp_input', tp_trace['mlp_input'], direct_trace['mlp_input'])} "
            f"{format_diff('mlp_output', tp_trace['mlp_output'], direct_trace['mlp_output'])} "
            f"{format_diff('layer_output', tp_trace['layer_output'], direct_trace['layer_output'])}"
        )

        direct_hidden = direct_trace["layer_output"]
        tp_hidden = tp_trace["layer_output"]


def run_tp(args):
    rank, world_size = init_dist()
    device = get_device(args.device)

    bundle = load_bundle(args.bundle_path)
    compute_dtype_name = bundle["save_dtype"] if args.compute_dtype == "auto" else args.compute_dtype
    compute_dtype = dtype_from_name(compute_dtype_name)
    comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)
    bundle = move_bundle(bundle, device, compute_dtype)

    layer_input = bundle["layer_input"]
    reference_output = bundle["layer_output"]
    direct_output = forward_layer_range(layer_input, bundle)
    tp_output = forward_layer_range_tp(layer_input, bundle, rank, world_size, comm_dtype)

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
        f"attn_impl={first_layer.get('attn_implementation', 'unknown')}"
    )
    print(
        f"[config] layer_input_shape={tuple(layer_input.shape)} layer_output_shape={tuple(reference_output.shape)} "
        f"dtype={layer_input.dtype} bundle_dtype={bundle['save_dtype']} "
        f"original_input_dtype={bundle['original_input_dtype']} original_input_device={bundle['original_input_device']} "
        f"comm_dtype={comm_dtype}"
    )
    print(
        f"[direct] rank={rank} direct_range_vs_reference max_diff={direct_max} "
        f"mean_diff={direct_mean}"
    )
    print(
        f"[range] rank={rank} tp_vs_reference max_diff={tp_ref_max} "
        f"mean_diff={tp_ref_mean} tp_vs_direct max_diff={tp_direct_max} mean_diff={tp_direct_mean}"
    )
    if args.trace_layers:
        print_layer_trace(bundle, rank, world_size, comm_dtype)


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL 多层 TP runtime 原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取真实多层 range bundle。")
    prepare_parser.add_argument("--start-idx", type=int, default=11)
    prepare_parser.add_argument("--end-idx", type=int, default=12)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=LAYER_RANGE_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    tp_parser = subparsers.add_parser("tp", help="跑多层 TP forward。")
    tp_parser.add_argument("--bundle-path", type=str, default=LAYER_RANGE_BUNDLE_PATH)
    tp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
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
