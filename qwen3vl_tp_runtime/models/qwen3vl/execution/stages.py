"""Stage- and logits-level replay utilities for Qwen3-VL runtime."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen3vl_tp_runtime.models.qwen3vl.execution.common import (
    apply_deepstack,
    compose_layer_bundle,
    get_deepstack_embeds,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution.attention import (
    forward_attention,
    forward_attention_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution.decoder import (
    forward_decoder_layer,
    forward_decoder_layer_cached,
    forward_decoder_layer_cached_tp,
    forward_decoder_layer_tp,
    trace_decoder_layer,
    trace_decoder_layer_cached,
    trace_decoder_layer_cached_tp,
    trace_decoder_layer_tp,
)
from qwen3vl_tp_runtime.models.qwen3vl.execution.mlp import forward_mlp, forward_mlp_tp
from qwen3vl_tp_runtime.models.qwen3vl.functional import rms_norm


def forward_layer_range(hidden_states: torch.Tensor, range_bundle: dict) -> torch.Tensor:
    output = hidden_states
    for layer_bundle in range_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, range_bundle)
        output = forward_decoder_layer(output, runtime_bundle)
    return output


def forward_layer_range_tp(
    hidden_states: torch.Tensor,
    range_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    for layer_bundle in range_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, range_bundle)
        output = forward_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
    return output


def forward_text_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]

        output = forward_decoder_layer(output, runtime_bundle)
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))

    return output


def forward_text_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]

        output = forward_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))

    return output


def forward_text_embeddings(input_ids: torch.Tensor, bundle: dict) -> torch.Tensor:
    return F.embedding(input_ids, bundle["embed_tokens_weight"])


def trace_text_prefill_logits(layer_input: torch.Tensor, bundle: dict) -> dict:
    stage_output = forward_text_stage(layer_input, bundle)
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_prefill_logits(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_prefill_logits(layer_input, bundle)["logits"]


def forward_text_decode_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        output = forward_decoder_layer_cached(output, runtime_bundle)
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))
    return output


def forward_text_decode_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        output = forward_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        output = apply_deepstack(output, visual_pos_masks, get_deepstack_embeds(stage_bundle, layer_idx))
    return output


def trace_text_decode_logits(layer_input: torch.Tensor, bundle: dict) -> dict:
    stage_output = forward_text_decode_stage(layer_input, bundle)
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_decode_logits(layer_input: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_decode_logits(layer_input, bundle)["logits"]


def trace_text_decode_stage_with_runtime_cache(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    current_cache = cache_by_layer or {}
    updated_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = int(layer_bundle["layer_idx"])
        past_key, past_value = current_cache.get(
            layer_idx,
            (runtime_bundle.get("past_key"), runtime_bundle.get("past_value")),
        )
        runtime_bundle["past_key"] = past_key
        runtime_bundle["past_value"] = past_value

        layer_trace = trace_decoder_layer_cached(output, runtime_bundle)
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        updated_cache[layer_idx] = (
            layer_trace["full_key"].detach().clone(),
            layer_trace["full_value"].detach().clone(),
        )
        output = post_deepstack

    return {
        "stage_output": output,
        "layer_traces": traces,
        "cache_by_layer": updated_cache,
    }


def trace_text_decode_logits_with_runtime_cache(
    layer_input: torch.Tensor,
    bundle: dict,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    stage_trace = trace_text_decode_stage_with_runtime_cache(
        layer_input,
        bundle,
        cache_by_layer=cache_by_layer,
    )
    norm_output = rms_norm(
        stage_trace["stage_output"],
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_trace["stage_output"],
        "norm_output": norm_output,
        "logits": logits,
        "cache_by_layer": stage_trace["cache_by_layer"],
        "layer_traces": stage_trace["layer_traces"],
    }


def trace_text_decode_stage_tp_with_runtime_cache(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")
    current_cache = cache_by_layer or {}
    updated_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = int(layer_bundle["layer_idx"])
        past_key, past_value = current_cache.get(
            layer_idx,
            (runtime_bundle.get("past_key"), runtime_bundle.get("past_value")),
        )
        runtime_bundle["past_key"] = past_key
        runtime_bundle["past_value"] = past_value

        layer_trace = trace_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        updated_cache[layer_idx] = (
            layer_trace["full_key"].detach().clone(),
            layer_trace["full_value"].detach().clone(),
        )
        output = post_deepstack

    return {
        "stage_output": output,
        "layer_traces": traces,
        "cache_by_layer": updated_cache,
    }


def trace_text_decode_logits_tp_with_runtime_cache(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
) -> dict:
    stage_trace = trace_text_decode_stage_tp_with_runtime_cache(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
        cache_by_layer=cache_by_layer,
    )
    norm_output = rms_norm(
        stage_trace["stage_output"],
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "layer_input": layer_input,
        "stage_output": stage_trace["stage_output"],
        "norm_output": norm_output,
        "logits": logits,
        "cache_by_layer": stage_trace["cache_by_layer"],
        "layer_traces": stage_trace["layer_traces"],
    }


def forward_text_decode_logits_tp(
    layer_input: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    stage_output = forward_text_decode_stage_tp(
        layer_input,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )
    norm_output = rms_norm(
        stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    return F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )


def trace_text_prefill_stage_logits(hidden_states: torch.Tensor, bundle: dict) -> dict:
    hidden_stage_output = forward_text_stage(hidden_states, bundle)
    norm_output = rms_norm(
        hidden_stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "stage_input": hidden_states,
        "hidden_stage_output": hidden_stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def trace_text_prefill_stage_logits_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> dict:
    hidden_stage_output = forward_text_stage_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )
    norm_output = rms_norm(
        hidden_stage_output,
        bundle["final_norm_weight"],
        bundle["final_norm_eps"],
    )
    logits = F.linear(
        norm_output,
        bundle["lm_head_weight"],
        bundle["lm_head_bias"],
    )
    return {
        "stage_input": hidden_states,
        "hidden_stage_output": hidden_stage_output,
        "norm_output": norm_output,
        "logits": logits,
    }


def forward_text_prefill_stage_logits(hidden_states: torch.Tensor, bundle: dict) -> torch.Tensor:
    return trace_text_prefill_stage_logits(hidden_states, bundle)["logits"]


def forward_text_prefill_stage_logits_tp(
    hidden_states: torch.Tensor,
    bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> torch.Tensor:
    return trace_text_prefill_stage_logits_tp(
        hidden_states,
        bundle,
        rank,
        world_size,
        comm_dtype,
        tp_group=tp_group,
        tp_src_rank=tp_src_rank,
        attn_math_mode=attn_math_mode,
        mlp_math_mode=mlp_math_mode,
    )["logits"]


def trace_text_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer(output, runtime_bundle)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_decode_stage(hidden_states: torch.Tensor, stage_bundle: dict) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_cached(output, runtime_bundle)
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


def trace_text_decode_stage_tp(
    hidden_states: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    attn_math_mode: str = "orig",
    mlp_math_mode: str = "orig",
) -> list[dict]:
    traces = []
    output = hidden_states
    visual_pos_masks = stage_bundle.get("visual_pos_masks")

    for layer_bundle in stage_bundle["layers"]:
        runtime_bundle = compose_layer_bundle(layer_bundle, stage_bundle)
        layer_idx = layer_bundle["layer_idx"]
        deepstack_embeds = get_deepstack_embeds(stage_bundle, layer_idx)

        layer_trace = trace_decoder_layer_cached_tp(
            output,
            runtime_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=attn_math_mode,
            mlp_math_mode=mlp_math_mode,
        )
        post_deepstack = apply_deepstack(layer_trace["layer_output"], visual_pos_masks, deepstack_embeds)
        layer_trace["layer_idx"] = layer_idx
        layer_trace["deepstack_applied"] = deepstack_embeds is not None
        layer_trace["post_deepstack"] = post_deepstack
        traces.append(layer_trace)

        output = post_deepstack

    return traces


build_layer_runtime_bundle = compose_layer_bundle
replay_attn = forward_attention
replay_attn_tp = forward_attention_tp
replay_mlp = forward_mlp
replay_mlp_tp = forward_mlp_tp
replay_layer = forward_decoder_layer
replay_layer_trace = trace_decoder_layer
replay_layer_tp = forward_decoder_layer_tp
replay_layer_tp_trace = trace_decoder_layer_tp
replay_layer_range = forward_layer_range
replay_layer_range_tp = forward_layer_range_tp


__all__ = [
    "forward_layer_range",
    "forward_layer_range_tp",
    "forward_text_stage",
    "forward_text_stage_tp",
    "forward_text_embeddings",
    "trace_text_prefill_logits",
    "forward_text_prefill_logits",
    "forward_text_decode_stage",
    "forward_text_decode_stage_tp",
    "trace_text_decode_logits",
    "forward_text_decode_logits",
    "trace_text_decode_stage_with_runtime_cache",
    "trace_text_decode_logits_with_runtime_cache",
    "trace_text_decode_stage_tp_with_runtime_cache",
    "trace_text_decode_logits_tp_with_runtime_cache",
    "forward_text_decode_logits_tp",
    "trace_text_prefill_stage_logits",
    "trace_text_prefill_stage_logits_tp",
    "forward_text_prefill_stage_logits",
    "forward_text_prefill_stage_logits_tp",
    "trace_text_stage",
    "trace_text_decode_stage",
    "trace_text_stage_tp",
    "trace_text_decode_stage_tp",
    "build_layer_runtime_bundle",
    "replay_attn",
    "replay_attn_tp",
    "replay_mlp",
    "replay_mlp_tp",
    "replay_layer",
    "replay_layer_trace",
    "replay_layer_tp",
    "replay_layer_tp_trace",
    "replay_layer_range",
    "replay_layer_range_tp",
]
