"""CLI for preparing and replaying a single Qwen3-VL decoder layer under TP."""

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import FULL_LAYER_BUNDLE_PATH, get_device, init_dist
from qwen3vl_tp_runtime.models.qwen3vl import (
    capture_full_layer_bundle,
    dtype_from_name,
    forward_decoder_layer,
    forward_decoder_layer_tp,
    load_bundle,
    move_bundle,
    resolve_comm_dtype,
)


def run_prepare(args):
    bundle = capture_full_layer_bundle(
        layer_idx=args.layer_idx,
        num_frames=args.num_frames,
        bundle_path=args.bundle_path,
        save_dtype=args.save_dtype,
    )

    print(f"[prepare] bundle saved to {args.bundle_path}")
    print(
        f"[prepare] layer={bundle['layer_idx']} save_dtype={bundle['save_dtype']} "
        f"num_heads={bundle['num_attention_heads']} num_kv_heads={bundle['num_key_value_heads']} "
        f"head_dim={bundle['head_dim']} attn_impl={bundle['attn_implementation']} hidden_act={bundle['hidden_act']}"
    )
    print(
        f"[prepare] layer_input_shape={tuple(bundle['layer_input'].shape)} "
        f"attn_input_shape={tuple(bundle['attn_input'].shape)} "
        f"mlp_input_shape={tuple(bundle['mlp_input'].shape)} "
        f"layer_output_shape={tuple(bundle['layer_output'].shape)}"
    )
    print(
        f"[prepare] attention_mask_shape={tuple(bundle['attention_mask'].shape)} "
        f"q_weight_shape={tuple(bundle['q_weight'].shape)} "
        f"gate_weight_shape={tuple(bundle['gate_weight'].shape)}"
    )
    print(
        f"[prepare] input_device={bundle['original_input_device']} "
        f"output_device={bundle['original_output_device']}"
    )
    print(
        f"[prepare] sanity max_diff={bundle['sanity_max_diff']} "
        f"mean_diff={bundle['sanity_mean_diff']}"
    )


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
    direct_output = forward_decoder_layer(layer_input, bundle)
    tp_output = forward_decoder_layer_tp(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        attn_math_mode=args.tp_attn_math,
        mlp_math_mode=args.tp_mlp_math,
    )

    direct_max = (direct_output - reference_output).abs().max().item()
    direct_mean = (direct_output - reference_output).abs().mean().item()
    tp_ref_max = (tp_output - reference_output).abs().max().item()
    tp_ref_mean = (tp_output - reference_output).abs().mean().item()
    tp_direct_max = (tp_output - direct_output).abs().max().item()
    tp_direct_mean = (tp_output - direct_output).abs().mean().item()

    local_q_heads = bundle["num_attention_heads"] // world_size
    local_kv_heads = bundle["num_key_value_heads"] // world_size

    print(
        f"[config] rank={rank} device={device} world_size={world_size} layer={bundle['layer_idx']} "
        f"num_heads={bundle['num_attention_heads']} num_kv_heads={bundle['num_key_value_heads']} "
        f"local_q_heads={local_q_heads} local_kv_heads={local_kv_heads} head_dim={bundle['head_dim']} "
        f"hidden_act={bundle['hidden_act']} attn_impl={bundle.get('attn_implementation', 'unknown')} "
        f"tp_attn_math={args.tp_attn_math} tp_mlp_math={args.tp_mlp_math}"
    )
    print(
        f"[config] layer_input_shape={tuple(layer_input.shape)} layer_output_shape={tuple(reference_output.shape)} "
        f"dtype={layer_input.dtype} bundle_dtype={bundle['save_dtype']} "
        f"original_input_dtype={bundle['original_input_dtype']} original_input_device={bundle['original_input_device']} "
        f"comm_dtype={comm_dtype}"
    )
    print(
        f"[direct] rank={rank} direct_full_layer_vs_reference max_diff={direct_max} "
        f"mean_diff={direct_mean}"
    )
    print(
        f"[full-layer] rank={rank} tp_vs_reference max_diff={tp_ref_max} "
        f"mean_diff={tp_ref_mean} tp_vs_direct max_diff={tp_direct_max} mean_diff={tp_direct_mean}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL 单层 TP runtime 原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取真实 full layer bundle。")
    prepare_parser.add_argument("--layer-idx", type=int, default=11)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--bundle-path", type=str, default=FULL_LAYER_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    tp_parser = subparsers.add_parser("tp", help="跑完整 full layer TP forward。")
    tp_parser.add_argument("--bundle-path", type=str, default=FULL_LAYER_BUNDLE_PATH)
    tp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tp_parser.add_argument("--compute-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    tp_parser.add_argument("--tp-attn-math", choices=["float32", "orig"], default="orig")
    tp_parser.add_argument("--tp-mlp-math", choices=["float32", "orig"], default="float32")
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
