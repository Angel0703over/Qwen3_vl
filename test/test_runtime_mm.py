from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from qwen3vl_tp_runtime.models.qwen3vl.live.common import MultimodalRuntimeInputs
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm import build_mm_stage_visuals, prepare_mm_session


class _FakeBatch(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.to_calls: list[torch.device] = []

    def to(self, device: torch.device):
        self.to_calls.append(device)
        return self


class RuntimeMmTest(unittest.TestCase):
    def test_prepare_mm_session_builds_live_prefill_inputs(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        fake_processor = object()
        fake_inputs = _FakeBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        fake_runtime_inputs = object()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
            return_value=fake_model,
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_processor",
            return_value=fake_processor,
        ) as load_processor_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.list_frames",
            return_value=["/tmp/f0.png", "/tmp/f1.png"],
        ) as list_frames_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.build_inputs",
            return_value=fake_inputs,
        ) as build_inputs_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_multimodal_prefill_runtime_inputs",
            return_value=fake_runtime_inputs,
        ) as prepare_inputs_mock:
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                }
            )

        load_model_mock.assert_called_once_with("/tmp/fake-model", attn_implementation="eager")
        load_processor_mock.assert_called_once_with("/tmp/fake-model")
        list_frames_mock.assert_called_once_with(2, "/tmp/frames")
        build_inputs_mock.assert_called_once_with(fake_processor, ["/tmp/f0.png", "/tmp/f1.png"])
        prepare_inputs_mock.assert_called_once_with(fake_model, fake_inputs)
        self.assertEqual(fake_inputs.to_calls, [torch.device("cpu")])
        self.assertIs(session.model, fake_model)
        self.assertIs(session.raw_inputs, fake_inputs)
        self.assertIs(session.runtime_inputs, fake_runtime_inputs)
        self.assertEqual(
            session.extra,
            {
                "num_frames": 2,
                "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            },
        )

    def test_build_mm_stage_visuals_filters_stage_layers(self) -> None:
        runtime_inputs = MultimodalRuntimeInputs(
            input_ids=None,
            attention_mask_2d=None,
            position_ids=None,
            inputs_embeds=torch.zeros(1, 1, 4),
            attention_mask=torch.zeros(1, 1, 1, 1),
            cos=torch.zeros(1, 1, 4),
            sin=torch.zeros(1, 1, 4),
            visual_pos_masks=torch.ones(1, 3, dtype=torch.bool),
            deepstack_by_layer={
                1: torch.ones(1, 2, dtype=torch.float32),
                4: torch.full((1, 2), 4.0, dtype=torch.float32),
            },
        )

        visual_pos_masks, deepstack_by_layer = build_mm_stage_visuals(
            runtime_inputs,
            start_idx=0,
            end_idx=2,
            device=torch.device("cpu"),
            compute_dtype=torch.float16,
        )

        self.assertIsNotNone(visual_pos_masks)
        self.assertTrue(torch.equal(visual_pos_masks, runtime_inputs.visual_pos_masks))
        self.assertEqual(sorted(deepstack_by_layer), [1])
        self.assertEqual(deepstack_by_layer[1].dtype, torch.float16)
        self.assertTrue(torch.equal(deepstack_by_layer[1], torch.ones(1, 2, dtype=torch.float16)))

    def test_build_mm_stage_visuals_clears_masks_without_local_layers(self) -> None:
        runtime_inputs = MultimodalRuntimeInputs(
            input_ids=None,
            attention_mask_2d=None,
            position_ids=None,
            inputs_embeds=torch.zeros(1, 1, 4),
            attention_mask=torch.zeros(1, 1, 1, 1),
            cos=torch.zeros(1, 1, 4),
            sin=torch.zeros(1, 1, 4),
            visual_pos_masks=torch.ones(1, 3, dtype=torch.bool),
            deepstack_by_layer={5: torch.ones(1, 2)},
        )

        visual_pos_masks, deepstack_by_layer = build_mm_stage_visuals(
            runtime_inputs,
            start_idx=0,
            end_idx=2,
            device=torch.device("cpu"),
            compute_dtype=torch.float16,
        )

        self.assertIsNone(visual_pos_masks)
        self.assertEqual(deepstack_by_layer, {})


if __name__ == "__main__":
    unittest.main()
