"""Debug-only manifest replay loaders for the unified runtime CLI."""

from __future__ import annotations


def load_debug_pipeline_manifest(manifest_path: str):
    """Load a prepared PP manifest for debug/regression replay runs."""

    from qwen3vl_tp_runtime.hexgen_core.modules.pipeline_parallel import load_pipeline_manifest

    return load_pipeline_manifest(manifest_path)


def load_debug_hybrid_manifest(manifest_path: str):
    """Load a prepared hybrid/TP manifest for debug/regression replay runs."""

    from qwen3vl_tp_runtime.hexgen_core.modules.hybrid_parallel import load_hybrid_manifest

    return load_hybrid_manifest(manifest_path)


__all__ = [
    "load_debug_pipeline_manifest",
    "load_debug_hybrid_manifest",
]
