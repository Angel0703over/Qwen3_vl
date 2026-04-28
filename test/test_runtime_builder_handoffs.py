from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import DirectStageStateBuilder


class RuntimeBuilderHandoffTest(unittest.TestCase):
    def test_build_stage_handoffs_reports_missing_final_boundary(self) -> None:
        builder = object.__new__(DirectStageStateBuilder)
        builder.stage_specs = [
            StageSpec(stage_idx=0, start_idx=0, end_idx=1, num_layers=2, save_dtype="float32")
        ]
        builder.compute_dtype = torch.float32

        with self.assertRaisesRegex(RuntimeError, "hidden_states 缺少 stage handoff 边界"):
            builder._build_stage_handoffs_from_hidden_states([torch.zeros(1, 2, 3), torch.ones(1, 2, 3)])

    def test_build_stage_handoffs_uses_final_boundary(self) -> None:
        builder = object.__new__(DirectStageStateBuilder)
        builder.stage_specs = [
            StageSpec(stage_idx=0, start_idx=0, end_idx=1, num_layers=2, save_dtype="float32")
        ]
        builder.compute_dtype = torch.float32
        hidden_states = [
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, 3),
            torch.full((1, 2, 3), 2.0),
        ]

        handoffs = builder._build_stage_handoffs_from_hidden_states(hidden_states)

        self.assertTrue(torch.equal(handoffs[0]["stage_input"], hidden_states[0]))
        self.assertTrue(torch.equal(handoffs[0]["stage_output"], hidden_states[2]))


if __name__ == "__main__":
    unittest.main()

