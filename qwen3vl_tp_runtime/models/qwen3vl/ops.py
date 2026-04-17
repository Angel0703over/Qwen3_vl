"""Low-level tensor ops used by the Qwen3-VL replay and TP kernels."""

import torch
import torch.nn.functional as F


def dtype_from_name(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"不支持的 dtype 名称: {dtype_name}")
    return getattr(torch, dtype_name)


def resolve_save_dtype(save_dtype_arg: str, reference_tensor: torch.Tensor) -> torch.dtype:
    if save_dtype_arg == "auto":
        return reference_tensor.dtype
    return dtype_from_name(save_dtype_arg)


def resolve_comm_dtype(comm_dtype_arg: str, compute_dtype: torch.dtype) -> torch.dtype:
    if comm_dtype_arg == "auto":
        # gloo + CPU 通信默认优先用 float32，避免 bf16 collectives 再引入额外噪声。
        return torch.float32 if compute_dtype == torch.bfloat16 else compute_dtype
    return dtype_from_name(comm_dtype_arg)


def cast_cpu(tensor: torch.Tensor | None, save_dtype: torch.dtype | None = None):
    if tensor is None:
        return None
    out = tensor.detach().clone().to("cpu")
    if save_dtype is not None:
        out = out.to(dtype=save_dtype)
    return out


def build_causal_mask(inputs_embeds: torch.Tensor, attention_mask_2d: torch.Tensor | None) -> torch.Tensor:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    min_value = torch.finfo(dtype).min

    causal_mask = torch.full((batch_size, 1, seq_len, seq_len), min_value, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    if attention_mask_2d is not None:
        attention_mask_2d = attention_mask_2d.to(device=device)
        key_padding_mask = attention_mask_2d[:, None, None, :] == 0
        causal_mask = causal_mask.masked_fill(key_padding_mask, min_value)

    return causal_mask


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + eps)
    return weight * hidden_states_fp32.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def attn_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_key_value_groups: int,
    scaling: float,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
