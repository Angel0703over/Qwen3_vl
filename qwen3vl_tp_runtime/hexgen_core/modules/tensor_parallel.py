"""Tensor-parallel runtime entrypoints for direct StageState execution."""

import torch
import torch.distributed as dist

from ..distributed import broadcast_cpu, startup_log, startup_timer
from ..schema import StageState, TensorParallelManifest
from ...debug.tp_debug import (
    TpDebugConfig,
    build_stage_traces,
)
from ..stage import (
    get_stage_input,
    get_stage_output,
    run_stage,
    run_stage_tp,
)
from ...models.qwen3vl.execution import (
    forward_text_embeddings,
    trace_text_decode_logits_tp_with_runtime_cache,
)
from ...models.qwen3vl.capture.common import load_bundle, move_bundle
from ...models.qwen3vl.functional import dtype_from_name, resolve_comm_dtype
from ...models.qwen3vl.runtime_builder import (
    build_direct_stage_state,
    materialize_text_stage_state,
)
from ...models.qwen3vl.runtime_mm_stage import build_mm_decode_state_from_weights
from ...models.qwen3vl.runtime_text_stage import summarize_text_weight_load
from ...models.qwen3vl.weights import (
    build_text_rotary_embedding,
    build_text_runtime_aux_tensors,
    load_text_model_config_spec,
)


def tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    return diff.max().item(), diff.mean().item()


def _all_tp_stages_are_direct(manifest: TensorParallelManifest) -> bool:
    manifest_is_direct = getattr(manifest, "is_direct", None)
    if manifest_is_direct is not None:
        return bool(manifest_is_direct)
    return all(
        getattr(stage, "replay_bundle_path", getattr(stage, "bundle_path", None)) is None
        for stage in getattr(manifest, "stages", [])
    )


def _validate_tp_manifest(manifest: TensorParallelManifest, world_size: int) -> None:
    if manifest.num_stages != 1 or len(manifest.stage_rank_groups) != 1:
        raise ValueError(
            "backend=tp 需要单独 TP manifest：恰好一个 stage、无 PP。"
        )
    if len(manifest.tp_degrees) != 1:
        raise ValueError(f"backend=tp 需要恰好一个 TP degree，当前拿到 {manifest.tp_degrees!r}。")
    if manifest.tp_degrees[0] <= 1:
        raise ValueError("backend=tp 要求 TP degree > 1。")
    if manifest.world_size != world_size or manifest.tp_degrees[0] != world_size:
        raise ValueError(
            "backend=tp manifest world_size 和 torchrun world_size 不一致，"
            f"manifest_world_size={manifest.world_size} tp_degree={manifest.tp_degrees[0]} "
            f"runtime_world_size={world_size}。"
        )


def load_stage_state_for_tp_rank(
    manifest: TensorParallelManifest,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    compute_dtype_arg: str,
) -> tuple[StageState, torch.dtype]:
    _validate_tp_manifest(manifest, world_size)
    stage_meta = manifest.stages[0]
    runtime_modality = str(manifest.runtime_config.get("modality", "multimodal"))

    if _all_tp_stages_are_direct(manifest):
        startup_log(
            "tp-direct-loader",
            f"rank={rank} building shared direct scaffold stage_idx={stage_meta.stage_idx} "
            f"range={stage_meta.start_idx}:{stage_meta.end_idx}",
        )
        scaffold = build_direct_stage_state(
            stage_idx=stage_meta.stage_idx,
            start_idx=stage_meta.start_idx,
            end_idx=stage_meta.end_idx,
            runtime_config=manifest.runtime_config,
            include_text_weights=False,
            mm_activate_frontend=(
                stage_meta.start_idx == 0 if runtime_modality == "multimodal" else None
            ),
        )
        compute_dtype_name = scaffold["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
        compute_dtype = dtype_from_name(compute_dtype_name)
        with startup_timer(
            "tp-direct-loader",
            f"materialize local direct shard rank={rank} stage_idx={stage_meta.stage_idx} "
            f"tp_local_rank={rank}/{world_size}",
        ):
            stage_state = materialize_text_stage_state(
                stage_state_scaffold=scaffold,
                runtime_config=manifest.runtime_config,
                compute_dtype=compute_dtype,
                tp_shard_rank=rank,
                tp_shard_world_size=world_size,
            )
    else:
        replay_bundle_path = getattr(stage_meta, "replay_bundle_path", None) or getattr(
            stage_meta,
            "bundle_path",
            None,
        )
        if replay_bundle_path is None:
            raise RuntimeError("backend=tp replay manifest 缺少单 stage bundle path。")
        stage_state = load_bundle(replay_bundle_path)
        compute_dtype_name = stage_state["save_dtype"] if compute_dtype_arg == "auto" else compute_dtype_arg
        compute_dtype = dtype_from_name(compute_dtype_name)

    return move_bundle(stage_state, device, compute_dtype), compute_dtype


def load_tp_manifest(manifest_path: str) -> TensorParallelManifest:
    payload = torch.load(manifest_path, map_location="cpu")
    if isinstance(payload, TensorParallelManifest):
        return payload

    manifest_dict = payload.to_dict() if hasattr(payload, "to_dict") else payload
    if "tp_degrees" not in manifest_dict:
        raise ValueError("manifest 里没有 tp_degrees，不能按 TP 运行。")
    return TensorParallelManifest.from_dict(manifest_dict)


def build_generate_phase_state(
    stage_state: StageState,
    phase_payload: dict,
    *,
    stage_type: str,
) -> StageState:
    runtime_state = {
        key: value
        for key, value in stage_state.items()
        if key not in {"prefill", "decode_steps", "generated_token_ids"}
    }
    runtime_state.update(phase_payload)
    runtime_state["stage_type"] = stage_type
    if stage_type in {"text_decode", "text_decode_last"}:
        runtime_state["visual_pos_masks"] = phase_payload.get("visual_pos_masks")
        runtime_state["deepstack_by_layer"] = dict(phase_payload.get("deepstack_by_layer", {}))
        runtime_state["deepstack_layer_indices"] = list(phase_payload.get("deepstack_layer_indices", []))
    if "layer_input" not in runtime_state and "stage_input" in runtime_state:
        runtime_state["layer_input"] = runtime_state["stage_input"]
    return runtime_state


def strip_runtime_layer_cache(stage_state: StageState) -> StageState:
    stripped_state = dict(stage_state)
    stripped_state["layers"] = [
        {
            key: value
            for key, value in layer_state.items()
            if key not in {"past_key", "past_value"}
        }
        for layer_state in stage_state["layers"]
    ]
    return stripped_state


def build_generate_cache_map(stage_state: StageState) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    return {
        int(layer_state["layer_idx"]): (
            layer_state.get("past_key"),
            layer_state.get("past_value"),
        )
        for layer_state in stage_state["layers"]
    }


def is_runtime_only_generate_state(stage_state: StageState) -> bool:
    return bool(stage_state.get("runtime_only_generate"))


def infer_runtime_tensor_device(stage_state: StageState) -> torch.device:
    if stage_state.get("embed_tokens_weight") is not None:
        return stage_state["embed_tokens_weight"].device
    if stage_state.get("layers"):
        return stage_state["layers"][0]["q_weight"].device
    if stage_state.get("final_norm_weight") is not None:
        return stage_state["final_norm_weight"].device
    if stage_state.get("input_ids") is not None:
        return stage_state["input_ids"].device
    return stage_state["prefill_attention_mask_2d"].device


def infer_runtime_tensor_dtype(stage_state: StageState) -> torch.dtype:
    if stage_state.get("embed_tokens_weight") is not None:
        return stage_state["embed_tokens_weight"].dtype
    if stage_state.get("layers"):
        return stage_state["layers"][0]["q_weight"].dtype
    if stage_state.get("final_norm_weight") is not None:
        return stage_state["final_norm_weight"].dtype
    return torch.float32


def infer_runtime_token_dtype(stage_state: StageState) -> torch.dtype:
    if stage_state.get("input_ids") is not None:
        return stage_state["input_ids"].dtype
    token_id_dtype = stage_state.get("token_id_dtype")
    if isinstance(token_id_dtype, str):
        return dtype_from_name(token_id_dtype)
    return torch.int64


def build_runtime_only_stage_input_template(stage_state: StageState, *, query_len: int) -> torch.Tensor:
    return torch.empty(
        (int(stage_state["batch_size"]), query_len, int(stage_state["hidden_size"])),
        device=infer_runtime_tensor_device(stage_state),
        dtype=infer_runtime_tensor_dtype(stage_state),
    )


def _build_runtime_only_generate_phase_state(
    stage_state: StageState,
    *,
    phase_kind: str,
    attention_mask_2d: torch.Tensor,
    config_spec,
    rotary_emb,
) -> StageState:
    query_len = int(stage_state["prefill_seq_len"]) if phase_kind == "prefill" else 1
    runtime_state = dict(stage_state)
    if phase_kind == "decode":
        for key in (
            "stage_input",
            "layer_input",
            "stage_output",
            "layer_output",
            "hidden_stage_output",
            "norm_output",
            "output_token_id",
        ):
            runtime_state.pop(key, None)
    runtime_state["stage_type"] = "text_prefill_last" if phase_kind == "prefill" else "text_decode_last"
    runtime_state["attention_mask_2d"] = attention_mask_2d

    if str(stage_state.get("modality", "text")) == "multimodal" and phase_kind == "prefill":
        runtime_state["attention_mask"] = stage_state["prefill_attention_mask"]
        runtime_state["position_ids"] = stage_state.get("prefill_position_ids")
        runtime_state["cos"] = stage_state["prefill_cos"]
        runtime_state["sin"] = stage_state["prefill_sin"]
    elif str(stage_state.get("modality", "text")) == "multimodal":
        decode_input_ids = torch.zeros(
            (int(stage_state["batch_size"]), 1),
            device=infer_runtime_tensor_device(stage_state),
            dtype=infer_runtime_token_dtype(stage_state),
        )
        dummy_embed_tokens_weight = torch.zeros(
            (1, int(config_spec.hidden_size)),
            device=infer_runtime_tensor_device(stage_state),
            dtype=infer_runtime_tensor_dtype(stage_state),
        )
        decode_state = build_mm_decode_state_from_weights(
            decode_input_ids=decode_input_ids,
            attention_mask_2d=attention_mask_2d,
            past_length=int(attention_mask_2d.shape[-1]) - query_len,
            rope_deltas=stage_state["rope_deltas"],
            embed_tokens_weight=dummy_embed_tokens_weight,
            config_spec=config_spec,
            device=infer_runtime_tensor_device(stage_state),
            compute_dtype=infer_runtime_tensor_dtype(stage_state),
            rotary_emb=rotary_emb,
        )
        runtime_state["attention_mask"] = decode_state.attention_mask
        runtime_state["position_ids"] = decode_state.position_ids
        runtime_state["cos"] = decode_state.cos
        runtime_state["sin"] = decode_state.sin
        runtime_state["visual_pos_masks"] = None
        runtime_state["deepstack_by_layer"] = {}
        runtime_state["deepstack_layer_indices"] = []
    else:
        runtime_aux = build_text_runtime_aux_tensors(
            attention_mask_2d=attention_mask_2d,
            batch_size=int(stage_state["batch_size"]),
            seq_len=query_len,
            past_length=int(attention_mask_2d.shape[-1]) - query_len,
            config_spec=config_spec,
            device=infer_runtime_tensor_device(stage_state),
            compute_dtype=infer_runtime_tensor_dtype(stage_state),
            rotary_emb=rotary_emb,
        )
        runtime_state["attention_mask"] = runtime_aux["attention_mask"]
        runtime_state["position_ids"] = runtime_aux["position_ids"]
        runtime_state["cos"] = runtime_aux["cos"]
        runtime_state["sin"] = runtime_aux["sin"]
    if phase_kind == "prefill" and stage_state.get("input_ids") is not None:
        runtime_state["input_ids"] = stage_state["input_ids"]
    if phase_kind == "decode":
        runtime_state["decode_input_ids"] = torch.zeros(
            (int(stage_state["batch_size"]), 1),
            device=infer_runtime_tensor_device(stage_state),
            dtype=infer_runtime_token_dtype(stage_state),
        )
    return runtime_state


def token_tensor_to_list(token_tensor: torch.Tensor) -> list[int]:
    if token_tensor.dim() == 2:
        token_tensor = token_tensor[0]
    return [int(token_id) for token_id in token_tensor.tolist()]


def broadcast_token_id(token_id: int | None, *, src: int) -> int:
    token_tensor = torch.tensor([-1 if token_id is None else token_id], dtype=torch.int64)
    dist.broadcast(token_tensor, src=src)
    return int(token_tensor.item())


def _run_generate_phase_tp(
    *,
    rank: int,
    world_size: int,
    runtime_state: StageState,
    phase_kind: str,
    current_token_id: int | None,
    cache_by_layer: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None,
    comm_dtype: torch.dtype,
    tp_attn_math_mode: str,
    tp_mlp_math_mode: str,
    return_tensor: bool,
) -> tuple[dict, dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] | None]:
    reference_input = runtime_state.get("stage_input")
    if reference_input is None:
        reference_input = runtime_state.get("layer_input")
    query_len = int(runtime_state["prefill_seq_len"]) if phase_kind == "prefill" else 1
    if rank == 0:
        if phase_kind == "prefill":
            if runtime_state.get("embed_tokens_weight") is not None and "input_ids" in runtime_state:
                leader_input = forward_text_embeddings(runtime_state["input_ids"], runtime_state)
            else:
                leader_input = reference_input
        elif phase_kind == "decode":
            if current_token_id is None:
                raise ValueError("decode phase 需要 current_token_id，但当前拿到 None。")
            decode_input_ids = torch.tensor(
                [[current_token_id]],
                device=infer_runtime_tensor_device(runtime_state),
                dtype=runtime_state["decode_input_ids"].dtype,
            )
            leader_input = forward_text_embeddings(decode_input_ids, runtime_state)
            runtime_state["decode_input_ids_runtime"] = decode_input_ids
        else:
            raise ValueError(f"不支持的 phase_kind={phase_kind!r}")
    else:
        leader_input = None

    stage_input = broadcast_cpu(
        reference_tensor=(
            reference_input
            if reference_input is not None
            else build_runtime_only_stage_input_template(runtime_state, query_len=query_len)
        ),
        tensor=leader_input,
        src=0,
        comm_dtype=comm_dtype,
    )
    if reference_input is None:
        embedding_max, embedding_mean = None, None
    else:
        embedding_max, embedding_mean = tensor_diff_stats(stage_input, reference_input)

    if phase_kind == "prefill":
        trace_state = strip_runtime_layer_cache(runtime_state)
        trace = trace_text_decode_logits_tp_with_runtime_cache(
            stage_input,
            trace_state,
            rank,
            world_size,
            comm_dtype,
            tp_src_rank=0,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
            cache_by_layer={},
        )
    elif phase_kind == "decode":
        trace = trace_text_decode_logits_tp_with_runtime_cache(
            stage_input,
            runtime_state,
            rank,
            world_size,
            comm_dtype,
            tp_src_rank=0,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
            cache_by_layer=cache_by_layer,
        )
    else:
        raise ValueError(f"不支持的 phase_kind={phase_kind!r}")

    stage_output = trace["logits"]
    reference_output = runtime_state.get("stage_output")
    if reference_output is None:
        reference_output = runtime_state.get("layer_output")
    stage_max, stage_mean = (None, None) if reference_output is None else tensor_diff_stats(stage_output, reference_output)
    hidden_stage_max, hidden_stage_mean = (None, None)
    if runtime_state.get("hidden_stage_output") is not None:
        hidden_stage_max, hidden_stage_mean = tensor_diff_stats(trace["stage_output"], runtime_state["hidden_stage_output"])
    norm_max, norm_mean = (None, None)
    if runtime_state.get("norm_output") is not None:
        norm_max, norm_mean = tensor_diff_stats(trace["norm_output"], runtime_state["norm_output"])

    predicted_token_id = int(stage_output[0, -1].argmax().item()) if rank == 0 else None
    reference_token_id = None
    if rank == 0 and runtime_state.get("output_token_id") is not None:
        reference_token_id = int(runtime_state["output_token_id"])
    stats = {
        "input_shape": tuple(stage_input.shape),
        "output_shape": tuple(stage_output.shape),
        "boundary_max_diff": None,
        "boundary_mean_diff": None,
        "embedding_max_diff": embedding_max,
        "embedding_mean_diff": embedding_mean,
        "hidden_stage_max_diff": hidden_stage_max,
        "hidden_stage_mean_diff": hidden_stage_mean,
        "norm_max_diff": norm_max,
        "norm_mean_diff": norm_mean,
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
        "sent_shape": None,
        "received_payload_keys": [],
        "sent_payload_keys": [],
        "sent_tensor_shapes": {},
        "predicted_token_id": predicted_token_id,
        "reference_token_id": reference_token_id,
    }
    if return_tensor and rank == 0:
        stats["stage_output_tensor"] = stage_output
    return stats, trace["cache_by_layer"]


class StageRunner:
    """Base worker context for the standalone backend=tp direct StageState path."""

    def __init__(
        self,
        manifest: TensorParallelManifest,
        device: torch.device,
        compute_dtype_arg: str,
        comm_dtype_arg: str,
        tp_attn_math_mode: str = "orig",
        tp_mlp_math_mode: str = "orig",
        debug_config: TpDebugConfig | None = None,
        return_tensors: bool = True,
    ) -> None:
        self.manifest = manifest
        self.device = device
        self.compute_dtype_arg = compute_dtype_arg
        self.comm_dtype_arg = comm_dtype_arg
        self.tp_attn_math_mode = tp_attn_math_mode
        self.tp_mlp_math_mode = tp_mlp_math_mode
        self.debug_config = debug_config or TpDebugConfig()
        self.return_tensors = return_tensors

    def run_rank(self, rank: int, world_size: int) -> dict:
        _validate_tp_manifest(self.manifest, world_size)
        stage_state, compute_dtype = load_stage_state_for_tp_rank(
            self.manifest,
            rank=rank,
            world_size=world_size,
            device=self.device,
            compute_dtype_arg=self.compute_dtype_arg,
        )
        comm_dtype = resolve_comm_dtype(self.comm_dtype_arg, compute_dtype)
        if self.manifest.pipeline_type in {"text_generate", "multimodal_generate"}:
            return self._run_generate_rank(
                rank=rank,
                world_size=world_size,
                stage_state=stage_state,
                comm_dtype=comm_dtype,
            )
        return self._run_stage_rank(
            rank=rank,
            world_size=world_size,
            stage_state=stage_state,
            comm_dtype=comm_dtype,
        )

    def _run_stage_rank(
        self,
        *,
        rank: int,
        world_size: int,
        stage_state: StageState,
        comm_dtype: torch.dtype,
    ) -> dict:
        reference_input = get_stage_input(stage_state)
        stage_input = broadcast_cpu(
            reference_tensor=reference_input,
            tensor=reference_input if rank == 0 else None,
            src=0,
            comm_dtype=comm_dtype,
        )
        reference_output = stage_state.get("stage_output")
        tp_stage_stats = run_stage_state_tp(
            stage_input=stage_input,
            stage_state=stage_state,
            reference_input_override=reference_input,
            local_rank=rank,
            tp_degree=world_size,
            comm_dtype=comm_dtype,
            leader_rank=0,
            tp_attn_math_mode=self.tp_attn_math_mode,
            tp_mlp_math_mode=self.tp_mlp_math_mode,
            debug_config=self.debug_config,
        )
        stage_output = tp_stage_stats.pop("stage_output")
        stats = self._base_stats(
            rank=rank,
            world_size=world_size,
            stage_state=stage_state,
            comm_dtype=comm_dtype,
        )
        stats.update(
            {
                "input_shape": tuple(stage_input.shape),
                "output_shape": tuple(stage_output.shape),
                "sent_shape": None,
                "received_payload_keys": [],
                "sent_payload_keys": [],
                "sent_tensor_shapes": {},
                "boundary_max_diff": tp_stage_stats["boundary_max_diff"],
                "boundary_mean_diff": tp_stage_stats["boundary_mean_diff"],
                "direct_max_diff": tp_stage_stats["direct_max_diff"],
                "direct_mean_diff": tp_stage_stats["direct_mean_diff"],
                "stage_max_diff": tp_stage_stats["stage_max_diff"],
                "stage_mean_diff": tp_stage_stats["stage_mean_diff"],
                "tp_direct_max_diff": tp_stage_stats["tp_direct_max_diff"],
                "tp_direct_mean_diff": tp_stage_stats["tp_direct_mean_diff"],
                "trace_summary": tp_stage_stats["trace_summary"],
                "traces": tp_stage_stats["traces"],
                "outlier_dump": tp_stage_stats["outlier_dump"],
                "next_leader_rank": None,
                "send_list": [],
                "recv_list": [],
                "send_empty_list": [],
                "recv_empty_list": [],
            }
        )
        if self.return_tensors and rank == 0:
            stats["stage_output"] = stage_output
            stats["reference_output"] = reference_output
        return stats

    def _run_generate_rank(
        self,
        *,
        rank: int,
        world_size: int,
        stage_state: StageState,
        comm_dtype: torch.dtype,
    ) -> dict:
        runtime_only_generate = is_runtime_only_generate_state(stage_state)
        runtime_only_context = None
        if runtime_only_generate:
            config_spec = load_text_model_config_spec(self.manifest.runtime_config["model_path"])
            runtime_only_context = {
                "config_spec": config_spec,
                "rotary_emb": build_text_rotary_embedding(config_spec, device=self.device),
            }

        if runtime_only_generate:
            prefill_state = _build_runtime_only_generate_phase_state(
                stage_state,
                phase_kind="prefill",
                attention_mask_2d=stage_state["prefill_attention_mask_2d"],
                config_spec=runtime_only_context["config_spec"],
                rotary_emb=runtime_only_context["rotary_emb"],
            )
        else:
            prefill_state = build_generate_phase_state(
                stage_state,
                stage_state["prefill"],
                stage_type="text_prefill_last",
            )
        prefill_stats, prefill_cache = _run_generate_phase_tp(
            rank=rank,
            world_size=world_size,
            runtime_state=prefill_state,
            phase_kind="prefill",
            current_token_id=None,
            cache_by_layer=None,
            comm_dtype=comm_dtype,
            tp_attn_math_mode=self.tp_attn_math_mode,
            tp_mlp_math_mode=self.tp_mlp_math_mode,
            return_tensor=self.return_tensors,
        )
        current_token_id = broadcast_token_id(
            prefill_stats["predicted_token_id"] if rank == 0 else None,
            src=0,
        )
        generated_token_ids = [current_token_id]
        cache_by_layer = prefill_cache if prefill_cache is not None else build_generate_cache_map(stage_state)
        step_stats = []
        step_output_tensors = []
        current_attention_mask_2d = stage_state["prefill_attention_mask_2d"]
        decode_iterable = (
            range(int(stage_state["max_new_tokens"]) - 1)
            if runtime_only_generate
            else stage_state["decode_steps"]
        )
        for step_payload in decode_iterable:
            if runtime_only_generate:
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
                decode_state = _build_runtime_only_generate_phase_state(
                    stage_state,
                    phase_kind="decode",
                    attention_mask_2d=current_attention_mask_2d,
                    config_spec=runtime_only_context["config_spec"],
                    rotary_emb=runtime_only_context["rotary_emb"],
                )
            else:
                decode_state = build_generate_phase_state(
                    stage_state,
                    step_payload,
                    stage_type="text_decode_last",
                )
            current_step_stats, cache_by_layer = _run_generate_phase_tp(
                rank=rank,
                world_size=world_size,
                runtime_state=decode_state,
                phase_kind="decode",
                current_token_id=current_token_id,
                cache_by_layer=cache_by_layer,
                comm_dtype=comm_dtype,
                tp_attn_math_mode=self.tp_attn_math_mode,
                tp_mlp_math_mode=self.tp_mlp_math_mode,
                return_tensor=self.return_tensors,
            )
            current_token_id = broadcast_token_id(
                current_step_stats["predicted_token_id"] if rank == 0 else None,
                src=0,
            )
            generated_token_ids.append(current_token_id)
            if "stage_output_tensor" in current_step_stats:
                step_output_tensors.append(current_step_stats.pop("stage_output_tensor"))
            step_stats.append(current_step_stats)

        stats = self._base_stats(
            rank=rank,
            world_size=world_size,
            stage_state=stage_state,
            comm_dtype=comm_dtype,
        )
        stats.update(
            {
                "prefill_seq_len": int(stage_state["prefill_seq_len"]),
                "max_new_tokens": int(stage_state["max_new_tokens"]),
                "prefill": prefill_stats,
                "steps": step_stats,
                "generated_token_ids": generated_token_ids,
                "reference_generated_token_ids": (
                    None
                    if runtime_only_generate or stage_state.get("generated_token_ids") is None
                    else token_tensor_to_list(stage_state["generated_token_ids"])
                ),
            }
        )
        if self.return_tensors and rank == 0:
            stats["prefill_output_tensor"] = prefill_stats.pop("stage_output_tensor")
            stats["step_output_tensors"] = step_output_tensors
        return stats

    def _base_stats(
        self,
        *,
        rank: int,
        world_size: int,
        stage_state: StageState,
        comm_dtype: torch.dtype,
    ) -> dict:
        return {
            "rank": rank,
            "stage_idx": 0,
            "stage_ranks": list(range(world_size)),
            "local_rank": rank,
            "tp_degree": world_size,
            "leader_rank": 0,
            "pp_group_idx": 0,
            "current_pp_group": list(range(world_size)),
            "num_stages": 1,
            "start_idx": stage_state["start_idx"],
            "end_idx": stage_state["end_idx"],
            "num_layers": len(stage_state["layers"]),
            "weight_load": summarize_text_weight_load(stage_state),
            "device": str(self.device),
            "comm_dtype": str(comm_dtype),
            "tp_attn_math_mode": self.tp_attn_math_mode,
            "tp_mlp_math_mode": self.tp_mlp_math_mode,
        }


class TensorParallelRunner(StageRunner):
    """Rank runner for standalone backend=tp direct execution."""


def run_tensor_parallel_rank(
    *,
    rank: int,
    world_size: int,
    manifest: TensorParallelManifest,
    device: torch.device,
    compute_dtype_arg: str,
    comm_dtype_arg: str,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    debug_config: TpDebugConfig | None = None,
    return_tensors: bool = True,
) -> dict:
    runner = TensorParallelRunner(
        manifest=manifest,
        device=device,
        compute_dtype_arg=compute_dtype_arg,
        comm_dtype_arg=comm_dtype_arg,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
        debug_config=debug_config,
        return_tensors=return_tensors,
    )
    return runner.run_rank(rank, world_size)


def run_stage_state_tp(
    *,
    stage_input: torch.Tensor,
    stage_state: dict,
    reference_input_override: torch.Tensor | None = None,
    local_rank: int,
    tp_degree: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    leader_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "orig",
    debug_config: TpDebugConfig | None = None,
) -> dict:
    """Execute one StageState under TP and optionally collect direct/trace comparisons."""

    debug_config = debug_config or TpDebugConfig()
    reference_input = (
        reference_input_override if reference_input_override is not None else get_stage_input(stage_state)
    )
    reference_output = get_stage_output(stage_state)
    boundary_max, boundary_mean = tensor_diff_stats(stage_input, reference_input)

    direct_output = None
    direct_max = None
    direct_mean = None
    tp_direct_max = None
    tp_direct_mean = None
    if debug_config.needs_direct_output:
        direct_output = run_stage(stage_input, stage_state)
        direct_max, direct_mean = tensor_diff_stats(direct_output, reference_output)

    stage_output = run_stage_tp(
        stage_input,
        stage_state,
        rank=local_rank,
        world_size=tp_degree,
        comm_dtype=comm_dtype,
        tp_group=tp_group,
        tp_src_rank=leader_rank,
        tp_attn_math_mode=tp_attn_math_mode,
        tp_mlp_math_mode=tp_mlp_math_mode,
    )
    stage_max, stage_mean = tensor_diff_stats(stage_output, reference_output)

    if direct_output is not None:
        tp_direct_max, tp_direct_mean = tensor_diff_stats(stage_output, direct_output)

    traces = None
    outlier_dump = None
    trace_summary = None
    if debug_config.needs_layer_trace:
        traces, outlier_dump, trace_summary = build_stage_traces(
            reference_input=reference_input,
            stage_input=stage_input,
            stage_state=stage_state,
            local_rank=local_rank,
            tp_degree=tp_degree,
            comm_dtype=comm_dtype,
            tp_group=tp_group,
            leader_rank=leader_rank,
            tp_attn_math_mode=tp_attn_math_mode,
            tp_mlp_math_mode=tp_mlp_math_mode,
            dump_layer=debug_config.dump_layer,
            dump_topk=debug_config.dump_topk,
        )

    return {
        "input_shape": tuple(stage_input.shape),
        "output_shape": tuple(stage_output.shape),
        "boundary_max_diff": boundary_max,
        "boundary_mean_diff": boundary_mean,
        "direct_max_diff": direct_max,
        "direct_mean_diff": direct_mean,
        "stage_max_diff": stage_max,
        "stage_mean_diff": stage_mean,
        "tp_direct_max_diff": tp_direct_max,
        "tp_direct_mean_diff": tp_direct_mean,
        "traces": traces,
        "trace_summary": trace_summary,
        "outlier_dump": outlier_dump,
        "stage_output": stage_output,
    }


__all__ = [
    "TensorParallelManifest",
    "StageRunner",
    "TensorParallelRunner",
    "load_tp_manifest",
    "load_stage_state_for_tp_rank",
    "run_tensor_parallel_rank",
    "run_stage_state_tp",
]
