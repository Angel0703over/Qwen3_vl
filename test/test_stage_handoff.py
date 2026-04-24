from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import StageHandoffPayload
from qwen3vl_tp_runtime.hexgen_core.stage import build_stage_handoff_target_dtypes


class StageHandoffTest(unittest.TestCase):
    def test_runtime_only_generate_bundle_infers_hidden_dtype_from_layer_weights(self) -> None:
        bundle = {
            "runtime_only_generate": True,
            "save_dtype": "bfloat16",
            "batch_size": 1,
            "hidden_size": 8,
            "prefill_seq_len": 4,
            "prefill_input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "layers": [
                {
                    "layer_idx": 18,
                    "q_weight": torch.empty((8, 8), dtype=torch.bfloat16),
                }
            ],
        }

        target_dtypes = build_stage_handoff_target_dtypes(bundle)

        self.assertEqual(
            target_dtypes[StageHandoffPayload.HIDDEN_STATES_KEY],
            torch.bfloat16,
        )

    def test_runtime_only_generate_bundle_falls_back_to_save_dtype(self) -> None:
        bundle = {
            "runtime_only_generate": True,
            "save_dtype": "float16",
            "batch_size": 1,
            "hidden_size": 8,
            "prefill_seq_len": 4,
            "prefill_input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "layers": [],
        }

        target_dtypes = build_stage_handoff_target_dtypes(bundle)

        self.assertEqual(
            target_dtypes[StageHandoffPayload.HIDDEN_STATES_KEY],
            torch.float16,
        )


if __name__ == "__main__":
    unittest.main()
