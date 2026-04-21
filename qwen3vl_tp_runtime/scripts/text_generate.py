"""Script for capturing and replaying a text-only direct greedy generation loop."""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen3vl_tp_runtime.hexgen_core import (
    MODEL_PATH,
    TEXT_GENERATE_BUNDLE_PATH,
    TEXT_GENERATE_PIPELINE_BUNDLE_DIR,
    TEXT_GENERATE_PIPELINE_MANIFEST_PATH,
    get_device,
    init_dist,
    load_pipeline_manifest,
    parse_stage_ranges,
    prepare_text_generate_pipeline,
    run_text_generate_pipeline_rank,
)
from qwen3vl_tp_runtime.models.qwen3vl import (
    capture_text_generate_bundle,
    dtype_from_name,
    forward_text_embeddings,
    load_bundle,
    move_bundle,
    trace_text_decode_logits_with_runtime_cache,
    trace_text_prefill_logits,
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


def load_generate_bundle(bundle_path: str, device: torch.device, compute_dtype_arg: str) -> tuple[dict, torch.dtype]:
    bundle = load_bundle(bundle_path)
    compute_dtype_name = bundle["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
    compute_dtype = dtype_from_name(compute_dtype_name)
    return move_bundle(bundle, device, compute_dtype), compute_dtype


def bundle_with_runtime_tensors(bundle: dict, runtime_payload: dict) -> dict:
    runtime_bundle = {
        key: value
        for key, value in bundle.items()
        if key not in {"prefill", "prefill_cache_layers", "generated_token_ids", "decode_steps"}
    }
    runtime_bundle.update(runtime_payload)
    return runtime_bundle


def build_prefill_cache_map(
    prefill_cache_layers: list[dict],
) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    return {
        int(layer_payload["layer_idx"]): (
            layer_payload["past_key"],
            layer_payload["past_value"],
        )
        for layer_payload in prefill_cache_layers
    }


def token_tensor_to_list(token_tensor: torch.Tensor) -> list[int]:
    if token_tensor.dim() == 2:
        token_tensor = token_tensor[0]
    return [int(token_id) for token_id in token_tensor.tolist()]


def run_prepare(args) -> None:
    bundle = capture_text_generate_bundle(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        bundle_path=args.bundle_path,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
    )

    generated_token_ids = token_tensor_to_list(bundle["generated_token_ids"])
    print(f"[prepare] bundle saved to {args.bundle_path}")
    print(
        f"[prepare] prompt={bundle['prompt']!r} "
        f"prefill_seq_len={bundle['prefill_seq_len']} "
        f"max_new_tokens={bundle['max_new_tokens']} "
        f"num_layers={len(bundle['layers'])} "
        f"save_dtype={bundle['save_dtype']}"
    )
    print(
        f"[prepare] prefill_input_shape={tuple(bundle['prefill']['layer_input'].shape)} "
        f"prefill_logits_shape={tuple(bundle['prefill']['logits'].shape)} "
        f"num_decode_steps={len(bundle['decode_steps'])}"
    )
    print(f"[prepare] reference_generated_token_ids={generated_token_ids}")
    print(f"[prepare] prefill_topk={summarize_last_token_topk(bundle['prefill']['logits'], args.topk)}")
    if bundle["decode_steps"]:
        print(f"[prepare] final_decode_topk={summarize_last_token_topk(bundle['decode_steps'][-1]['logits'], args.topk)}")


def run_prepare_pp(args) -> None:
    stage_ranges = parse_stage_ranges(args.stage_ranges)
    manifest = prepare_text_generate_pipeline(
        stage_ranges=stage_ranges,
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        save_dtype=args.save_dtype,
        model_path=args.model_path,
    )
    stage0_bundle = load_bundle(manifest.stages[0].bundle_path)
    last_bundle = load_bundle(manifest.stages[-1].bundle_path)
    reference_generated_token_ids = token_tensor_to_list(last_bundle["generated_token_ids"])

    print(f"[prepare-pp] manifest saved to {args.manifest_path}")
    print(f"[prepare-pp] bundle_dir={args.bundle_dir}")
    print(
        f"[prepare-pp] prompt={stage0_bundle['prompt']!r} "
        f"prefill_seq_len={stage0_bundle['prefill_seq_len']} "
        f"max_new_tokens={stage0_bundle['max_new_tokens']} "
        f"num_stages={manifest.num_stages} "
        f"stage_ranges={manifest.stage_ranges} "
        f"save_dtype={manifest.save_dtype}"
    )
    print(
        f"[prepare-pp] stage0_prefill_input_shape={tuple(stage0_bundle['prefill']['stage_input'].shape)} "
        f"last_prefill_output_shape={tuple(last_bundle['prefill']['stage_output'].shape)} "
        f"num_decode_steps={len(last_bundle['decode_steps'])}"
    )
    for boundary in manifest.boundaries:
        print(
            f"[prepare-pp] boundary stage{boundary.src_stage_idx}->stage{boundary.dst_stage_idx} "
            f"max_diff={boundary.max_diff} mean_diff={boundary.mean_diff}"
        )
    print(f"[prepare-pp] reference_generated_token_ids={reference_generated_token_ids}")
    if "logits" in last_bundle["prefill"]:
        print(f"[prepare-pp] prefill_topk={summarize_last_token_topk(last_bundle['prefill']['logits'], args.topk)}")
    if last_bundle["decode_steps"] and "logits" in last_bundle["decode_steps"][-1]:
        print(
            f"[prepare-pp] final_decode_topk="
            f"{summarize_last_token_topk(last_bundle['decode_steps'][-1]['logits'], args.topk)}"
        )


def run_direct(args) -> None:
    device = get_device(args.device)
    bundle, compute_dtype = load_generate_bundle(args.bundle_path, device, args.compute_dtype)

    prefill_bundle = bundle_with_runtime_tensors(bundle, bundle["prefill"])
    input_ids = bundle["input_ids"]
    embedded_input = forward_text_embeddings(input_ids, prefill_bundle)
    prefill_trace = trace_text_prefill_logits(embedded_input, prefill_bundle)

    prefill_embedding_max, prefill_embedding_mean = tensor_diff_stats(embedded_input, bundle["prefill"]["layer_input"])
    prefill_stage_max, prefill_stage_mean = tensor_diff_stats(
        prefill_trace["stage_output"],
        bundle["prefill"]["stage_output"],
    )
    prefill_norm_max, prefill_norm_mean = tensor_diff_stats(
        prefill_trace["norm_output"],
        bundle["prefill"]["norm_output"],
    )
    prefill_logits_max, prefill_logits_mean = tensor_diff_stats(
        prefill_trace["logits"],
        bundle["prefill"]["logits"],
    )

    generated_token_ids = [int(prefill_trace["logits"][0, -1].argmax().item())]
    reference_generated_token_ids = token_tensor_to_list(bundle["generated_token_ids"])
    cache_by_layer = build_prefill_cache_map(bundle["prefill_cache_layers"])
    current_token_id = generated_token_ids[0]

    print(
        f"[run] device={device} compute_dtype={compute_dtype} "
        f"prompt={bundle['prompt']!r} prefill_seq_len={bundle['prefill_seq_len']} "
        f"max_new_tokens={bundle['max_new_tokens']}"
    )
    print(
        f"[run] prefill_input_ids_shape={tuple(input_ids.shape)} "
        f"prefill_embedded_input_shape={tuple(embedded_input.shape)} "
        f"prefill_logits_shape={tuple(prefill_trace['logits'].shape)}"
    )
    print(
        f"[run] prefill_embedding_vs_reference max_diff={prefill_embedding_max} "
        f"mean_diff={prefill_embedding_mean}"
    )
    print(f"[run] prefill_stage_vs_reference max_diff={prefill_stage_max} mean_diff={prefill_stage_mean}")
    print(f"[run] prefill_norm_vs_reference max_diff={prefill_norm_max} mean_diff={prefill_norm_mean}")
    print(f"[run] prefill_logits_vs_reference max_diff={prefill_logits_max} mean_diff={prefill_logits_mean}")
    print(f"[run] prefill_topk={summarize_last_token_topk(prefill_trace['logits'], args.topk)}")

    for step in bundle["decode_steps"]:
        step_idx = int(step["step_idx"])
        step_bundle = bundle_with_runtime_tensors(bundle, step)
        decode_input_ids = torch.tensor(
            [[current_token_id]],
            device=device,
            dtype=input_ids.dtype,
        )
        embedded_decode_input = forward_text_embeddings(decode_input_ids, step_bundle)
        step_trace = trace_text_decode_logits_with_runtime_cache(
            embedded_decode_input,
            step_bundle,
            cache_by_layer=cache_by_layer,
        )

        embedding_max, embedding_mean = tensor_diff_stats(embedded_decode_input, step["layer_input"])
        stage_max, stage_mean = tensor_diff_stats(step_trace["stage_output"], step["stage_output"])
        norm_max, norm_mean = tensor_diff_stats(step_trace["norm_output"], step["norm_output"])
        logits_max, logits_mean = tensor_diff_stats(step_trace["logits"], step["logits"])
        predicted_token_id = int(step_trace["logits"][0, -1].argmax().item())
        reference_input_token_id = int(step["decode_input_ids"][0, 0].item())
        reference_output_token_id = int(step["output_token_id"])

        print(
            f"[step {step_idx}] input_token_id={current_token_id} "
            f"reference_input_token_id={reference_input_token_id} "
            f"predicted_next_token_id={predicted_token_id} "
            f"reference_next_token_id={reference_output_token_id}"
        )
        print(
            f"[step {step_idx}] embedding_vs_reference max_diff={embedding_max} "
            f"mean_diff={embedding_mean}"
        )
        print(f"[step {step_idx}] stage_vs_reference max_diff={stage_max} mean_diff={stage_mean}")
        print(f"[step {step_idx}] norm_vs_reference max_diff={norm_max} mean_diff={norm_mean}")
        print(f"[step {step_idx}] logits_vs_reference max_diff={logits_max} mean_diff={logits_mean}")
        print(f"[step {step_idx}] topk={summarize_last_token_topk(step_trace['logits'], args.topk)}")

        cache_by_layer = step_trace["cache_by_layer"]
        current_token_id = predicted_token_id
        generated_token_ids.append(predicted_token_id)

    print(f"[run] generated_token_ids={generated_token_ids}")
    print(f"[run] reference_generated_token_ids={reference_generated_token_ids}")
    print(f"[run] token_match={generated_token_ids == reference_generated_token_ids}")


def _print_generate_phase_stats(prefix: str, rank: int, phase_stats: dict) -> None:
    if phase_stats["boundary_max_diff"] is not None:
        print(
            f"[{prefix} handoff] rank={rank} input_shape={phase_stats['input_shape']} "
            f"boundary_max_diff={phase_stats['boundary_max_diff']} "
            f"boundary_mean_diff={phase_stats['boundary_mean_diff']} "
            f"payload_keys={phase_stats['received_payload_keys']}"
        )
    else:
        print(
            f"[{prefix} handoff] rank={rank} input_shape={phase_stats['input_shape']} "
            f"payload_keys={phase_stats['received_payload_keys']}"
        )
    if phase_stats["embedding_max_diff"] is not None:
        print(
            f"[{prefix}] rank={rank} embedding_vs_reference "
            f"max_diff={phase_stats['embedding_max_diff']} "
            f"mean_diff={phase_stats['embedding_mean_diff']}"
        )
    print(
        f"[{prefix}] rank={rank} output_shape={phase_stats['output_shape']} "
        f"stage_vs_reference max_diff={phase_stats['stage_max_diff']} "
        f"mean_diff={phase_stats['stage_mean_diff']}"
    )
    if phase_stats["hidden_stage_max_diff"] is not None:
        print(
            f"[{prefix}] rank={rank} hidden_stage_vs_reference "
            f"max_diff={phase_stats['hidden_stage_max_diff']} "
            f"mean_diff={phase_stats['hidden_stage_mean_diff']}"
        )
    if phase_stats["norm_max_diff"] is not None:
        print(
            f"[{prefix}] rank={rank} norm_vs_reference "
            f"max_diff={phase_stats['norm_max_diff']} "
            f"mean_diff={phase_stats['norm_mean_diff']}"
        )
    if phase_stats["predicted_token_id"] is not None:
        print(
            f"[{prefix}] rank={rank} predicted_next_token_id={phase_stats['predicted_token_id']} "
            f"reference_next_token_id={phase_stats['reference_token_id']}"
        )
    if phase_stats["sent_shape"] is not None:
        print(
            f"[{prefix} send] rank={rank} sent_shape={phase_stats['sent_shape']} "
            f"payload_keys={phase_stats['sent_payload_keys']} "
            f"tensor_shapes={phase_stats['sent_tensor_shapes']}"
        )


def run_pp(args) -> None:
    rank, world_size = init_dist()
    device = get_device(args.device)
    manifest = load_pipeline_manifest(args.manifest_path)
    stats = run_text_generate_pipeline_rank(
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
        f"prefill_seq_len={stats['prefill_seq_len']} max_new_tokens={stats['max_new_tokens']} "
        f"comm_dtype={stats['comm_dtype']}"
    )
    _print_generate_phase_stats("prefill", rank, stats["prefill"])
    for step_idx, step_stats in enumerate(stats["steps"]):
        _print_generate_phase_stats(f"step {step_idx}", rank, step_stats)

    if rank == world_size - 1:
        last_bundle = load_bundle(manifest.stages[-1].bundle_path)
        prefill_output_tensor = stats.pop("prefill_output_tensor")
        step_output_tensors = stats.pop("step_output_tensors")
        print(f"[pp] prefill_topk={summarize_last_token_topk(prefill_output_tensor, args.topk)}")
        print(f"[pp] reference_prefill_topk={summarize_last_token_topk(last_bundle['prefill']['logits'], args.topk)}")
        for step_idx, (runtime_logits, step_payload) in enumerate(zip(step_output_tensors, last_bundle["decode_steps"])):
            print(f"[pp step {step_idx}] topk={summarize_last_token_topk(runtime_logits, args.topk)}")
            print(f"[pp step {step_idx}] reference_topk={summarize_last_token_topk(step_payload['logits'], args.topk)}")
        print(f"[pp] generated_token_ids={stats['generated_token_ids']}")
        print(f"[pp] reference_generated_token_ids={stats['reference_generated_token_ids']}")
        print(f"[pp] token_match={stats['generated_token_ids'] == stats['reference_generated_token_ids']}")

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-VL text-only direct/PP greedy generation loop baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抓取 text-only greedy generation reference bundle。")
    prepare_parser.add_argument("--prompt", type=str, default="请用中文简要介绍一下人工智能。")
    prepare_parser.add_argument("--max-new-tokens", type=int, default=4)
    prepare_parser.add_argument("--bundle-path", type=str, default=TEXT_GENERATE_BUNDLE_PATH)
    prepare_parser.add_argument("--save-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    prepare_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_parser.add_argument("--topk", type=int, default=5)

    run_parser = subparsers.add_parser("run", help="运行 text-only direct greedy generation replay。")
    run_parser.add_argument("--bundle-path", type=str, default=TEXT_GENERATE_BUNDLE_PATH)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    run_parser.add_argument("--topk", type=int, default=5)

    prepare_pp_parser = subparsers.add_parser("prepare-pp", help="抓取 text-only PP greedy generation pipeline。")
    prepare_pp_parser.add_argument("--prompt", type=str, default="请用中文简要介绍一下人工智能。")
    prepare_pp_parser.add_argument("--max-new-tokens", type=int, default=4)
    prepare_pp_parser.add_argument("--stage-ranges", nargs="+", default=["0:17", "18:35"])
    prepare_pp_parser.add_argument("--bundle-dir", type=str, default=TEXT_GENERATE_PIPELINE_BUNDLE_DIR)
    prepare_pp_parser.add_argument("--manifest-path", type=str, default=TEXT_GENERATE_PIPELINE_MANIFEST_PATH)
    prepare_pp_parser.add_argument("--save-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    prepare_pp_parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    prepare_pp_parser.add_argument("--topk", type=int, default=5)

    pp_parser = subparsers.add_parser("pp", help="运行 text-only PP greedy generation replay。")
    pp_parser.add_argument("--manifest-path", type=str, default=TEXT_GENERATE_PIPELINE_MANIFEST_PATH)
    pp_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    pp_parser.add_argument("--compute-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    pp_parser.add_argument("--comm-dtype", choices=["auto", "float32", "bfloat16"], default="auto")
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


if __name__ == "__main__":
    main()
