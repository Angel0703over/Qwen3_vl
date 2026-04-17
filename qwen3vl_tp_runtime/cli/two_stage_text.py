import argparse
import sys
from pathlib import Path

import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.core.capture import capture_text_stage_bundle, load_bundle, move_bundle
from qwen3vl_tp_runtime.core.config import TEXT_STAGE0_BUNDLE_PATH, TEXT_STAGE1_BUNDLE_PATH
from qwen3vl_tp_runtime.core.dist import get_device, init_dist
from qwen3vl_tp_runtime.core.ops import dtype_from_name, resolve_comm_dtype
from qwen3vl_tp_runtime.core.stage import get_stage_input, get_stage_output, run_stage
from qwen3vl_tp_runtime.core.transport import recv_hidden_states, send_hidden_states


def tensor_diff_stats(lhs, rhs):
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def validate_split_args(start_idx: int, split_idx: int, end_idx: int) -> None:
    if not start_idx <= split_idx < end_idx:
        raise ValueError("需要满足 start_idx <= split_idx < end_idx，才能切成两个连续 stage。")


def run_prepare(args):
    validate_split_args(args.start_idx, args.split_idx, args.end_idx)

    stage0_bundle = capture_text_stage_bundle(
        start_idx=args.start_idx,
        end_idx=args.split_idx,
        num_frames=args.num_frames,
        bundle_path=args.stage0_bundle_path,
        save_dtype=args.save_dtype,
    )
    stage1_bundle = capture_text_stage_bundle(
        start_idx=args.split_idx + 1,
        end_idx=args.end_idx,
        num_frames=args.num_frames,
        bundle_path=args.stage1_bundle_path,
        save_dtype=args.save_dtype,
    )

    boundary_max, boundary_mean = tensor_diff_stats(
        stage0_bundle["stage_output"],
        stage1_bundle["stage_input"],
    )

    print(f"[prepare] stage0_bundle saved to {args.stage0_bundle_path}")
    print(f"[prepare] stage1_bundle saved to {args.stage1_bundle_path}")
    print(
        f"[prepare] start_idx={args.start_idx} split_idx={args.split_idx} end_idx={args.end_idx} "
        f"stage0_layers={len(stage0_bundle['layers'])} stage1_layers={len(stage1_bundle['layers'])} "
        f"save_dtype={stage0_bundle['save_dtype']}"
    )
    print(
        f"[prepare] stage0_input_shape={tuple(stage0_bundle['stage_input'].shape)} "
        f"boundary_shape={tuple(stage0_bundle['stage_output'].shape)} "
        f"stage1_output_shape={tuple(stage1_bundle['stage_output'].shape)}"
    )
    print(
        f"[prepare] boundary stage0_output_vs_stage1_input max_diff={boundary_max} "
        f"mean_diff={boundary_mean}"
    )
    print(
        f"[prepare] stage0_sanity max_diff={stage0_bundle['sanity_max_diff']} "
        f"mean_diff={stage0_bundle['sanity_mean_diff']} "
        f"stage1_sanity max_diff={stage1_bundle['sanity_max_diff']} "
        f"mean_diff={stage1_bundle['sanity_mean_diff']}"
    )


def load_stage_bundle(bundle_path: str, device, compute_dtype_arg: str):
    bundle = load_bundle(bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def run_pp(args):
    rank, world_size = init_dist()
    if world_size != 2:
        raise ValueError("two_stage_text 当前只支持 WORLD_SIZE=2。")

    device = get_device(args.device)

    if rank == 0:
        bundle, compute_dtype = load_stage_bundle(args.stage0_bundle_path, device, args.compute_dtype)
        comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)

        stage_input = get_stage_input(bundle)
        reference_output = get_stage_output(bundle)
        stage_output = run_stage(stage_input, bundle)
        sent_shape = send_hidden_states(stage_output, dst=1, comm_dtype=comm_dtype)

        stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

        print(
            f"[config] rank={rank} device={device} world_size={world_size} "
            f"start_idx={bundle['start_idx']} end_idx={bundle['end_idx']} "
            f"num_layers={len(bundle['layers'])} comm_dtype={comm_dtype}"
        )
        print(
            f"[stage0] rank={rank} input_shape={tuple(stage_input.shape)} "
            f"output_shape={tuple(stage_output.shape)} sent_shape={sent_shape}"
        )
        print(
            f"[stage0] rank={rank} direct_vs_reference max_diff={stage_max} "
            f"mean_diff={stage_mean}"
        )
    else:
        bundle, compute_dtype = load_stage_bundle(args.stage1_bundle_path, device, args.compute_dtype)
        comm_dtype = resolve_comm_dtype(args.comm_dtype, compute_dtype)

        reference_input = get_stage_input(bundle)
        reference_output = get_stage_output(bundle)
        received_hidden = recv_hidden_states(
            src=0,
            device=device,
            hidden_dtype=reference_input.dtype,
            comm_dtype=comm_dtype,
        )
        boundary_max, boundary_mean = tensor_diff_stats(received_hidden, reference_input)

        stage_output = run_stage(received_hidden, bundle)
        stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

        print(
            f"[config] rank={rank} device={device} world_size={world_size} "
            f"start_idx={bundle['start_idx']} end_idx={bundle['end_idx']} "
            f"num_layers={len(bundle['layers'])} comm_dtype={comm_dtype}"
        )
        print(
            f"[handoff] rank={rank} recv_shape={tuple(received_hidden.shape)} "
            f"reference_input_shape={tuple(reference_input.shape)} "
            f"max_diff={boundary_max} mean_diff={boundary_mean}"
        )
        print(
            f"[stage1] rank={rank} pp_vs_reference max_diff={stage_max} "
            f"mean_diff={stage_mean}"
        )

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL 两段 text stage 的最小 PP 原型入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取两个连续 text stage 的 bundle。")
    prepare_parser.add_argument("--start-idx", type=int, default=0)
    prepare_parser.add_argument("--split-idx", type=int, default=5)
    prepare_parser.add_argument("--end-idx", type=int, default=11)
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--stage0-bundle-path", type=str, default=TEXT_STAGE0_BUNDLE_PATH)
    prepare_parser.add_argument("--stage1-bundle-path", type=str, default=TEXT_STAGE1_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float32", "bfloat16"], default="auto")

    pp_parser = subparsers.add_parser("pp", help="跑两段 text stage 的最小 pipeline handoff。")
    pp_parser.add_argument("--stage0-bundle-path", type=str, default=TEXT_STAGE0_BUNDLE_PATH)
    pp_parser.add_argument("--stage1-bundle-path", type=str, default=TEXT_STAGE1_BUNDLE_PATH)
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
