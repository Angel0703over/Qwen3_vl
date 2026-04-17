import torch

from qwen3vl_tp_runtime.core.forward import (
    forward_text_stage,
    forward_text_stage_tp,
    trace_text_stage,
    trace_text_stage_tp,
)


def get_stage_type(stage_bundle: dict) -> str:
    stage_type = stage_bundle.get("stage_type")
    if stage_type is not None:
        return stage_type
    module_name = stage_bundle.get("module_name")
    if module_name == "text_stage":
        return "text"
    raise ValueError(f"无法从 stage bundle 中识别 stage_type，module_name={module_name!r}")


def get_stage_input(stage_bundle: dict) -> torch.Tensor:
    return stage_bundle["stage_input"] if "stage_input" in stage_bundle else stage_bundle["layer_input"]


def get_stage_output(stage_bundle: dict) -> torch.Tensor:
    return stage_bundle["stage_output"] if "stage_output" in stage_bundle else stage_bundle["layer_output"]


def build_stage_bundle(stage_type: str, bundle: dict) -> dict:
    stage_bundle = dict(bundle)
    stage_bundle["stage_type"] = stage_type

    if "stage_input" not in stage_bundle and "layer_input" in stage_bundle:
        stage_bundle["stage_input"] = stage_bundle["layer_input"]
    if "stage_output" not in stage_bundle and "layer_output" in stage_bundle:
        stage_bundle["stage_output"] = stage_bundle["layer_output"]

    return stage_bundle


def run_stage(stage_input: torch.Tensor, stage_bundle: dict) -> torch.Tensor:
    stage_type = get_stage_type(stage_bundle)
    if stage_type == "text":
        return forward_text_stage(stage_input, stage_bundle)
    raise NotImplementedError(f"暂不支持的 stage_type: {stage_type}")


def run_stage_tp(
    stage_input: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "float32",
) -> torch.Tensor:
    stage_type = get_stage_type(stage_bundle)
    if stage_type == "text":
        return forward_text_stage_tp(
            stage_input,
            stage_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {stage_type}")


def trace_stage(stage_input: torch.Tensor, stage_bundle: dict):
    stage_type = get_stage_type(stage_bundle)
    if stage_type == "text":
        return trace_text_stage(stage_input, stage_bundle)
    raise NotImplementedError(f"暂不支持的 stage_type: {stage_type}")


def trace_stage_tp(
    stage_input: torch.Tensor,
    stage_bundle: dict,
    rank: int,
    world_size: int,
    comm_dtype: torch.dtype,
    tp_group=None,
    tp_src_rank: int = 0,
    tp_attn_math_mode: str = "orig",
    tp_mlp_math_mode: str = "float32",
):
    stage_type = get_stage_type(stage_bundle)
    if stage_type == "text":
        return trace_text_stage_tp(
            stage_input,
            stage_bundle,
            rank,
            world_size,
            comm_dtype,
            tp_group=tp_group,
            tp_src_rank=tp_src_rank,
            attn_math_mode=tp_attn_math_mode,
            mlp_math_mode=tp_mlp_math_mode,
        )
    raise NotImplementedError(f"暂不支持的 stage_type: {stage_type}")
