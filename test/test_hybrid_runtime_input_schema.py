from __future__ import annotations

import unittest

import torch

import qwen3vl_tp_runtime.hexgen_core as core_pkg
from qwen3vl_tp_runtime.hexgen_core.schema import (
    HYBRID_RUNTIME_INPUT_PROTOCOL,
    HybridRuntimeInputSchema,
)


def _common_payload(modality: str) -> dict[str, object]:
    return {
        "protocol": HYBRID_RUNTIME_INPUT_PROTOCOL,
        "modality": modality,
        "mode": "generate",
        "runtime_only_generate": True,
    }


class HybridRuntimeInputSchemaTest(unittest.TestCase):
    def test_text_runtime_input_schema_allows_prompt_tensors_only(self) -> None:
        payload = {
            **_common_payload("text"),
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask_2d": torch.ones(1, 3, dtype=torch.long),
            "runtime_only_prompt_local_rebuild": True,
        }

        HybridRuntimeInputSchema.validate(payload)

        self.assertIn("input_ids", HybridRuntimeInputSchema.allowed_top_level_keys("text"))
        self.assertIn("prefill_attention_mask_2d", HybridRuntimeInputSchema.local_rebuild_fields("text"))

    def test_multimodal_runtime_input_schema_allows_shared_and_stage_handoff(self) -> None:
        payload = {
            **_common_payload("multimodal"),
            "shared": {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask_2d": torch.ones(1, 3, dtype=torch.long),
                "position_ids": torch.zeros(4, 1, 3, dtype=torch.long),
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            "stage_handoff": {
                "stage_input": torch.zeros(1, 3, 4),
            },
            "stage_visuals": {
                "visual_pos_masks": torch.ones(1, 3, dtype=torch.bool),
                "deepstack_by_layer": {
                    0: torch.zeros(2, 4),
                },
            },
        }

        HybridRuntimeInputSchema.validate(payload)

        self.assertIn("shared", HybridRuntimeInputSchema.allowed_top_level_keys("multimodal"))
        self.assertIn("prefill_cos", HybridRuntimeInputSchema.local_rebuild_fields("multimodal"))

    def test_schema_rejects_stage_state_and_weight_fields(self) -> None:
        payload = {
            **_common_payload("text"),
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "runtime_only_prompt_local_rebuild": True,
            "layers": [],
        }

        with self.assertRaisesRegex(RuntimeError, "禁止广播"):
            HybridRuntimeInputSchema.validate(payload)

        payload = {
            **_common_payload("multimodal"),
            "shared": {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask_2d": torch.ones(1, 1, dtype=torch.long),
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                "q_weight": torch.zeros(1, 1),
            },
            "stage_handoff": {
                "stage_input": torch.zeros(1, 1, 1),
            },
        }

        with self.assertRaisesRegex(RuntimeError, "禁止广播"):
            HybridRuntimeInputSchema.validate(payload)

    def test_schema_rejects_frontend_paths_and_derived_attention_tensors(self) -> None:
        payload = {
            **_common_payload("multimodal"),
            "shared": {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask_2d": torch.ones(1, 1, dtype=torch.long),
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                "attention_mask": torch.zeros(1, 1, 1, 1),
            },
            "stage_handoff": {
                "stage_input": torch.zeros(1, 1, 1),
            },
        }

        with self.assertRaisesRegex(RuntimeError, "禁止广播"):
            HybridRuntimeInputSchema.validate(payload)

        payload = {
            **_common_payload("multimodal"),
            "shared": {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask_2d": torch.ones(1, 1, dtype=torch.long),
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
            },
            "stage_handoff": {
                "stage_input": torch.zeros(1, 1, 1),
            },
            "frame_paths": ["/tmp/frame.png"],
        }

        with self.assertRaisesRegex(RuntimeError, "禁止广播"):
            HybridRuntimeInputSchema.validate(payload)

    def test_schema_rejects_unknown_or_missing_protocol_fields(self) -> None:
        missing_input_ids = {
            **_common_payload("text"),
            "runtime_only_prompt_local_rebuild": True,
        }
        with self.assertRaisesRegex(RuntimeError, "缺少协议必需字段"):
            HybridRuntimeInputSchema.validate(missing_input_ids)

        unknown = {
            **_common_payload("text"),
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "runtime_only_prompt_local_rebuild": True,
            "token_positions": torch.zeros(1, 1, dtype=torch.long),
        }
        with self.assertRaisesRegex(RuntimeError, "字段不在协议内"):
            HybridRuntimeInputSchema.validate(unknown)

    def test_schema_is_exported_from_hexgen_core_root(self) -> None:
        self.assertIs(core_pkg.HybridRuntimeInputSchema, HybridRuntimeInputSchema)
        self.assertEqual(core_pkg.HYBRID_RUNTIME_INPUT_PROTOCOL, HYBRID_RUNTIME_INPUT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
