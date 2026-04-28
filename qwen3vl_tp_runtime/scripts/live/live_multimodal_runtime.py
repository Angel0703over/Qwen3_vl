"""No-bundle multimodal runtime smoke tests built on the live Qwen3-VL front-end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3vl_tp_runtime.hexgen_core.config import FRAME_DIR, MODEL_PATH
from qwen3vl_tp_runtime.models.qwen3vl.execution import (
    forward_text_prefill_logits,
    trace_text_decode_logits_with_runtime_cache,
)
from qwen3vl_tp_runtime.models.qwen3vl.live import (
    build_cache_by_layer_from_past_key_values,
    build_live_multimodal_stage_bundle,
    prepare_multimodal_decode_runtime_inputs,
    prepare_multimodal_prefill_runtime_inputs,
)
from qwen3vl_tp_runtime.models.qwen3vl.processing import (
    build_inputs,
    list_frames,
    load_model,
    load_processor,
)
from qwen3vl_tp_runtime.scripts.common import summarize_last_token_topk, tensor_diff_stats


def ensure_attention_mask_2d(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    attention_mask_2d = inputs.get("attention_mask")
    if attention_mask_2d is None:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("inputs 里既没有 attention_mask，也没有 input_ids。")
        return torch.ones_like(input_ids)
    return attention_mask_2d


def load_live_multimodal_case(
    *,
    model_path: str,
    frame_dir: str,
    num_frames: int,
):
    model = load_model(model_path, attn_implementation="eager")
    processor = load_processor(model_path)
    frame_paths = list_frames(num_frames, frame_dir)
    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(next(model.parameters()).device)
    return model, frame_paths, inputs


def build_prefill_runtime(
    *,
    model,
    inputs: dict[str, torch.Tensor],
    compute_dtype_arg: str,
):
    runtime_inputs = prepare_multimodal_prefill_runtime_inputs(model, inputs)
    num_layers = len(model.model.language_model.layers)
    bundle, compute_dtype = build_live_multimodal_stage_bundle(
        model,
        start_idx=0,
        end_idx=num_layers - 1,
        runtime_inputs=runtime_inputs,
        phase="prefill",
        compute_dtype_arg=compute_dtype_arg,
    )
    stage_input = runtime_inputs.inputs_embeds.to(device=next(model.parameters()).device, dtype=compute_dtype)
    runtime_logits = forward_text_prefill_logits(stage_input, bundle)
    return runtime_inputs, bundle, compute_dtype, runtime_logits


def run_prefill(args) -> None:
    model, frame_paths, inputs = load_live_multimodal_case(
        model_path=args.model_path,
        frame_dir=args.frame_dir,
        num_frames=args.num_frames,
    )

    with torch.inference_mode():
        reference_outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    runtime_inputs, _bundle, compute_dtype, runtime_logits = build_prefill_runtime(
        model=model,
        inputs=inputs,
        compute_dtype_arg=args.compute_dtype,
    )
    stage_max_diff, stage_mean_diff = tensor_diff_stats(runtime_logits, reference_outputs.logits)

    summary = {
        "mode": "prefill",
        "num_frames": len(frame_paths),
        "compute_dtype": str(compute_dtype),
        "input_shape": list(runtime_inputs.inputs_embeds.shape),
        "output_shape": list(runtime_logits.shape),
        "visual_tokens": (
            0 if runtime_inputs.visual_pos_masks is None else int(runtime_inputs.visual_pos_masks.sum().item())
        ),
        "deepstack_layers": sorted(runtime_inputs.deepstack_by_layer),
        "rope_deltas_shape": (
            None if runtime_inputs.rope_deltas is None else list(runtime_inputs.rope_deltas.shape)
        ),
        "stage_max_diff": stage_max_diff,
        "stage_mean_diff": stage_mean_diff,
        "runtime_topk": summarize_last_token_topk(runtime_logits, args.topk),
        "reference_topk": summarize_last_token_topk(reference_outputs.logits, args.topk),
        "runtime_token_id": int(runtime_logits[0, -1].argmax().item()),
        "reference_token_id": int(reference_outputs.logits[0, -1].argmax().item()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_decode(args) -> None:
    model, frame_paths, inputs = load_live_multimodal_case(
        model_path=args.model_path,
        frame_dir=args.frame_dir,
        num_frames=args.num_frames,
    )
    attention_mask_2d = ensure_attention_mask_2d(inputs)

    with torch.inference_mode():
        prefill_outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    runtime_prefill_inputs, _bundle, compute_dtype, runtime_prefill_logits = build_prefill_runtime(
        model=model,
        inputs=inputs,
        compute_dtype_arg=args.compute_dtype,
    )

    default_token_id = int(runtime_prefill_logits[0, -1].argmax().item())
    decode_token_id = default_token_id if args.decode_token_id is None else int(args.decode_token_id)
    decode_input_ids = torch.tensor(
        [[decode_token_id]],
        device=next(model.parameters()).device,
        dtype=inputs["input_ids"].dtype,
    )
    decode_attention_mask_2d = torch.cat(
        [
            attention_mask_2d,
            torch.ones(
                (attention_mask_2d.shape[0], 1),
                device=attention_mask_2d.device,
                dtype=attention_mask_2d.dtype,
            ),
        ],
        dim=-1,
    )

    runtime_inputs = prepare_multimodal_decode_runtime_inputs(
        model,
        decode_input_ids=decode_input_ids,
        attention_mask_2d=decode_attention_mask_2d,
        past_key_values=prefill_outputs.past_key_values,
    )
    # HF DynamicCache 可能在 decode forward 中原地增长，所以 runtime cache 必须先做快照。
    cache_by_layer = build_cache_by_layer_from_past_key_values(
        prefill_outputs.past_key_values,
        device=next(model.parameters()).device,
        compute_dtype=compute_dtype,
    )
    with torch.inference_mode():
        reference_outputs = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask_2d,
            position_ids=runtime_inputs.position_ids,
            past_key_values=prefill_outputs.past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    bundle, _ = build_live_multimodal_stage_bundle(
        model,
        start_idx=0,
        end_idx=len(model.model.language_model.layers) - 1,
        runtime_inputs=runtime_inputs,
        phase="decode",
        compute_dtype_arg=args.compute_dtype,
        cache_by_layer=cache_by_layer,
    )
    runtime_trace = trace_text_decode_logits_with_runtime_cache(
        runtime_inputs.inputs_embeds.to(device=next(model.parameters()).device, dtype=compute_dtype),
        bundle,
        cache_by_layer=cache_by_layer,
    )
    runtime_logits = runtime_trace["logits"]
    stage_max_diff, stage_mean_diff = tensor_diff_stats(runtime_logits, reference_outputs.logits)

    summary = {
        "mode": "decode",
        "num_frames": len(frame_paths),
        "compute_dtype": str(compute_dtype),
        "decode_token_id": decode_token_id,
        "prefill_visual_tokens": (
            0 if runtime_prefill_inputs.visual_pos_masks is None else int(runtime_prefill_inputs.visual_pos_masks.sum().item())
        ),
        "input_shape": list(runtime_inputs.inputs_embeds.shape),
        "output_shape": list(runtime_logits.shape),
        "rope_deltas_shape": (
            None if runtime_inputs.rope_deltas is None else list(runtime_inputs.rope_deltas.shape)
        ),
        "stage_max_diff": stage_max_diff,
        "stage_mean_diff": stage_mean_diff,
        "runtime_topk": summarize_last_token_topk(runtime_logits, args.topk),
        "reference_topk": summarize_last_token_topk(reference_outputs.logits, args.topk),
        "runtime_token_id": int(runtime_logits[0, -1].argmax().item()),
        "reference_token_id": int(reference_outputs.logits[0, -1].argmax().item()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_generate(args) -> None:
    model, frame_paths, inputs = load_live_multimodal_case(
        model_path=args.model_path,
        frame_dir=args.frame_dir,
        num_frames=args.num_frames,
    )
    attention_mask_2d = ensure_attention_mask_2d(inputs)

    with torch.inference_mode():
        prefill_outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    runtime_prefill_inputs, _prefill_bundle, compute_dtype, runtime_prefill_logits = build_prefill_runtime(
        model=model,
        inputs=inputs,
        compute_dtype_arg=args.compute_dtype,
    )

    runtime_generated_token_ids = [int(runtime_prefill_logits[0, -1].argmax().item())]
    reference_generated_token_ids = [int(prefill_outputs.logits[0, -1].argmax().item())]
    cache_by_layer = build_cache_by_layer_from_past_key_values(
        prefill_outputs.past_key_values,
        device=next(model.parameters()).device,
        compute_dtype=compute_dtype,
    )
    reference_past_key_values = prefill_outputs.past_key_values
    current_attention_mask_2d = attention_mask_2d
    step_summaries = []

    for step_idx in range(max(args.max_new_tokens - 1, 0)):
        decode_input_ids = torch.tensor(
            [[runtime_generated_token_ids[-1]]],
            device=next(model.parameters()).device,
            dtype=inputs["input_ids"].dtype,
        )
        current_attention_mask_2d = torch.cat(
            [
                current_attention_mask_2d,
                torch.ones(
                    (current_attention_mask_2d.shape[0], 1),
                    device=current_attention_mask_2d.device,
                    dtype=current_attention_mask_2d.dtype,
                ),
            ],
            dim=-1,
        )

        runtime_inputs = prepare_multimodal_decode_runtime_inputs(
            model,
            decode_input_ids=decode_input_ids,
            attention_mask_2d=current_attention_mask_2d,
            past_key_values=reference_past_key_values,
        )
        with torch.inference_mode():
            reference_outputs = model(
                input_ids=decode_input_ids,
                attention_mask=current_attention_mask_2d,
                position_ids=runtime_inputs.position_ids,
                past_key_values=reference_past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
        bundle, _ = build_live_multimodal_stage_bundle(
            model,
            start_idx=0,
            end_idx=len(model.model.language_model.layers) - 1,
            runtime_inputs=runtime_inputs,
            phase="decode",
            compute_dtype_arg=args.compute_dtype,
            cache_by_layer=cache_by_layer,
        )
        runtime_trace = trace_text_decode_logits_with_runtime_cache(
            runtime_inputs.inputs_embeds.to(device=next(model.parameters()).device, dtype=compute_dtype),
            bundle,
            cache_by_layer=cache_by_layer,
        )
        runtime_logits = runtime_trace["logits"]
        stage_max_diff, stage_mean_diff = tensor_diff_stats(runtime_logits, reference_outputs.logits)

        runtime_next_token_id = int(runtime_logits[0, -1].argmax().item())
        reference_next_token_id = int(reference_outputs.logits[0, -1].argmax().item())
        runtime_generated_token_ids.append(runtime_next_token_id)
        reference_generated_token_ids.append(reference_next_token_id)
        cache_by_layer = runtime_trace["cache_by_layer"]
        reference_past_key_values = reference_outputs.past_key_values

        step_summaries.append(
            {
                "step_idx": step_idx,
                "decode_token_id": int(decode_input_ids.item()),
                "stage_max_diff": stage_max_diff,
                "stage_mean_diff": stage_mean_diff,
                "runtime_topk": summarize_last_token_topk(runtime_logits, args.topk),
                "reference_topk": summarize_last_token_topk(reference_outputs.logits, args.topk),
                "runtime_token_id": runtime_next_token_id,
                "reference_token_id": reference_next_token_id,
            }
        )

    processor = load_processor(args.model_path)
    runtime_generated_text = processor.post_process_image_text_to_text(
        [torch.tensor(runtime_generated_token_ids, dtype=torch.long)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    reference_generated_text = processor.post_process_image_text_to_text(
        [torch.tensor(reference_generated_token_ids, dtype=torch.long)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    summary = {
        "mode": "generate",
        "num_frames": len(frame_paths),
        "compute_dtype": str(compute_dtype),
        "max_new_tokens": args.max_new_tokens,
        "prefill_visual_tokens": (
            0 if runtime_prefill_inputs.visual_pos_masks is None else int(runtime_prefill_inputs.visual_pos_masks.sum().item())
        ),
        "prefill_topk": summarize_last_token_topk(runtime_prefill_logits, args.topk),
        "reference_prefill_topk": summarize_last_token_topk(prefill_outputs.logits, args.topk),
        "runtime_generated_token_ids": runtime_generated_token_ids,
        "generated_token_ids": runtime_generated_token_ids,
        "generated_text": runtime_generated_text,
        "reference_generated_token_ids": reference_generated_token_ids,
        "reference_generated_text": reference_generated_text,
        "token_match": runtime_generated_token_ids == reference_generated_token_ids,
        "steps": step_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-bundle Qwen3-VL multimodal runtime smoke tests.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--frame-dir", type=str, default=FRAME_DIR)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--compute-dtype", type=str, default="auto")
    parser.add_argument("--topk", type=int, default=10)

    subparsers = parser.add_subparsers(dest="command", required=True)

    prefill_parser = subparsers.add_parser("prefill", help="Run live multimodal prefill replay.")
    prefill_parser.set_defaults(func=run_prefill)

    decode_parser = subparsers.add_parser("decode", help="Run live multimodal single-step decode replay.")
    decode_parser.add_argument("--decode-token-id", type=int, default=None)
    decode_parser.set_defaults(func=run_decode)

    generate_parser = subparsers.add_parser("generate", help="Run live multimodal greedy generation replay.")
    generate_parser.add_argument("--max-new-tokens", type=int, default=4)
    generate_parser.set_defaults(func=run_generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
