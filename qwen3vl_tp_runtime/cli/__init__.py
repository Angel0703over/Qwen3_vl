"""CLI entrypoint exports for the prototype runtime workflows."""

from qwen3vl_tp_runtime.cli.full_layer import main as full_layer_main
from qwen3vl_tp_runtime.cli.layer_range import main as layer_range_main
from qwen3vl_tp_runtime.cli.text_hybrid import main as text_hybrid_main
from qwen3vl_tp_runtime.cli.text_pipeline import main as text_pipeline_main
from qwen3vl_tp_runtime.cli.text_stage import main as text_stage_main
from qwen3vl_tp_runtime.cli.two_stage_text import main as two_stage_text_main

__all__ = [
    "full_layer_main",
    "layer_range_main",
    "text_hybrid_main",
    "text_pipeline_main",
    "text_stage_main",
    "two_stage_text_main",
]
