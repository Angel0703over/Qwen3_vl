import importlib

_EXPORTS = {
    "TensorPayload": ("qwen3vl_tp_runtime.hexgen_core.transport", "TensorPayload"),
    "parse_tp_degrees": ("qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups", "parse_tp_degrees"),
    "build_stage_rank_groups": ("qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups", "build_stage_rank_groups"),
    "build_pp_rank_groups": ("qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups", "build_pp_rank_groups"),
    "build_p2p_lists": ("qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups", "build_p2p_lists"),
    "build_hybrid_layout": ("qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups", "build_hybrid_layout"),
    "prepare_text_hybrid": ("qwen3vl_tp_runtime.hexgen_core.heterogeneous_pipeline", "prepare_text_hybrid"),
    "load_hybrid_manifest": ("qwen3vl_tp_runtime.hexgen_core.heterogeneous_pipeline", "load_hybrid_manifest"),
    "init_stage_groups": ("qwen3vl_tp_runtime.hexgen_core.heterogeneous_pipeline", "init_stage_groups"),
    "resolve_rank_stage": ("qwen3vl_tp_runtime.hexgen_core.heterogeneous_pipeline", "resolve_rank_stage"),
    "run_text_hybrid_rank": ("qwen3vl_tp_runtime.hexgen_core.heterogeneous_pipeline", "run_text_hybrid_rank"),
    "compose_layer_bundle": ("qwen3vl_tp_runtime.hexgen_core.generation", "compose_layer_bundle"),
    "apply_deepstack": ("qwen3vl_tp_runtime.hexgen_core.generation", "apply_deepstack"),
    "get_deepstack_embeds": ("qwen3vl_tp_runtime.hexgen_core.generation", "get_deepstack_embeds"),
    "get_stage_type": ("qwen3vl_tp_runtime.hexgen_core.generation", "get_stage_type"),
    "get_stage_input": ("qwen3vl_tp_runtime.hexgen_core.generation", "get_stage_input"),
    "get_stage_output": ("qwen3vl_tp_runtime.hexgen_core.generation", "get_stage_output"),
    "run_stage": ("qwen3vl_tp_runtime.hexgen_core.generation", "run_stage"),
    "run_stage_tp": ("qwen3vl_tp_runtime.hexgen_core.generation", "run_stage_tp"),
    "trace_stage": ("qwen3vl_tp_runtime.hexgen_core.generation", "trace_stage"),
    "trace_stage_tp": ("qwen3vl_tp_runtime.hexgen_core.generation", "trace_stage_tp"),
    "send_payload": ("qwen3vl_tp_runtime.hexgen_core.transport", "send_payload"),
    "recv_payload": ("qwen3vl_tp_runtime.hexgen_core.transport", "recv_payload"),
    "send_tensor": ("qwen3vl_tp_runtime.hexgen_core.transport", "send_tensor"),
    "recv_tensor": ("qwen3vl_tp_runtime.hexgen_core.transport", "recv_tensor"),
    "send_hidden_states": ("qwen3vl_tp_runtime.hexgen_core.transport", "send_hidden_states"),
    "recv_hidden_states": ("qwen3vl_tp_runtime.hexgen_core.transport", "recv_hidden_states"),
    "getenv_int": ("qwen3vl_tp_runtime.hexgen_core.utils", "getenv_int"),
    "init_dist": ("qwen3vl_tp_runtime.hexgen_core.utils", "init_dist"),
    "get_device": ("qwen3vl_tp_runtime.hexgen_core.utils", "get_device"),
    "dtype_from_name": ("qwen3vl_tp_runtime.hexgen_core.utils", "dtype_from_name"),
    "resolve_save_dtype": ("qwen3vl_tp_runtime.hexgen_core.utils", "resolve_save_dtype"),
    "resolve_comm_dtype": ("qwen3vl_tp_runtime.hexgen_core.utils", "resolve_comm_dtype"),
    "tensor_diff_stats": ("qwen3vl_tp_runtime.hexgen_core.utils", "tensor_diff_stats"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
