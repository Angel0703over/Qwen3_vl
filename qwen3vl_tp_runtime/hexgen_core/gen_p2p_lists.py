"""Compatibility shim exposing the HexGen-style point-to-point list builder."""

from qwen3vl_tp_runtime.hexgen_core.gen_hetero_groups import build_p2p_lists

__all__ = ["build_p2p_lists"]
