from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from qwen3vl_tp_runtime.models.qwen3vl.live.common import MultimodalRuntimeInputs
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm import (
    build_mm_stage_visuals,
    prepare_mm_frontend_seed,
    prepare_mm_session,
    restore_mm_frontend_seed,
)
from qwen3vl_tp_runtime.models.qwen3vl.processing import DEFAULT_VIDEO_PROMPT
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmRuntimeState,
    MmVisualState,
    compact_mm_frontend_meta,
    compact_mm_frontend_tensors,
    compact_mm_runtime_shared,
    build_mm_stage_state,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import TextModelConfigSpec
from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import MmFrontendBatch
from qwen3vl_tp_runtime.models.qwen3vl.vision.state import (
    MmFrontendAttnPlan,
    MmFrontendPlan,
    MmFrontendPosPlan,
    MmFrontendVisualPlan,
    mm_state_from_frontend_plan,
)


class _FakeBatch(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.to_calls: list[torch.device] = []

    def to(self, device: torch.device):
        self.to_calls.append(device)
        return self


class _DummyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.bias = nn.Parameter(torch.zeros(2))


class _ForwardVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.spatial_merge_size = 1
        self.calls: list[dict[str, object]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(self, pixel_values, **kwargs):
        self.calls.append({"pixel_values": pixel_values, **kwargs})
        return type(
            "VisionOutput",
            (),
            {
                "pooler_output": torch.ones(1, 4, dtype=self.weight.dtype, device=pixel_values.device),
                "deepstack_features": [],
            },
        )()


def _fake_mm_frontend_plan() -> MmFrontendPlan:
    return MmFrontendPlan(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask_2d=torch.ones(1, 3, dtype=torch.long),
        visual=MmFrontendVisualPlan(
            inputs_embeds=torch.zeros(1, 3, 4),
            visual=MmVisualState(
                visual_pos_masks=torch.ones(1, 3, dtype=torch.bool),
                deepstack_by_layer={},
            ),
        ),
        pos=MmFrontendPosPlan(
            position_ids=torch.tensor(
                [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]],
                dtype=torch.long,
            ),
            text_position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            vision_position_ids=torch.tensor(
                [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]],
                dtype=torch.long,
            ),
        ),
        attn=MmFrontendAttnPlan(
            attention_mask=torch.zeros(1, 1, 3, 3),
            cos=torch.zeros(1, 3, 4),
            sin=torch.zeros(1, 3, 4),
        ),
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 1, 0]], dtype=torch.int),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )


def _fake_mm_runtime_state() -> MmRuntimeState:
    return mm_state_from_frontend_plan(_fake_mm_frontend_plan())


def _fake_text_config_spec() -> TextModelConfigSpec:
    return TextModelConfigSpec(
        model_path="/tmp/fake-model",
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
        vocab_size=8,
        head_dim=2,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "mrope_interleaved": True,
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000.0,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=True,
        pad_token_id=None,
        scaling=1.0 / (2 ** 0.5),
        attn_implementation="eager",
    )


class RuntimeMmTest(unittest.TestCase):
    def test_build_mm_stage_state_rebuilds_missing_derived_shared_tensors(self) -> None:
        fake_state = _fake_mm_runtime_state()
        compact_shared = compact_mm_runtime_shared(
            fake_state,
            include_derived=False,
        )
        compact_shared.pop("attention_mask_2d")
        compact_shared.pop("position_ids")

        class _FakeRotary(torch.nn.Module):
            def forward(self, inputs_embeds, vision_position_ids):
                return (
                    torch.full_like(inputs_embeds, 2.0),
                    torch.full_like(inputs_embeds, 3.0),
                )

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage.create_causal_mask",
            return_value=None,
        ):
            restored = build_mm_stage_state(
                compact_shared,
                stage_input=fake_state.inputs_embeds,
                start_idx=0,
                end_idx=0,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
                config_spec=_fake_text_config_spec(),
                mm_config=type("FakeMmConfig", (), {
                    "vision_config": type("FakeVisionConfig", (), {"spatial_merge_size": 1})(),
                })(),
                rotary_emb=_FakeRotary(),
            )

        self.assertNotIn("attention_mask_2d", compact_shared)
        self.assertNotIn("position_ids", compact_shared)
        self.assertNotIn("attention_mask", compact_shared)
        self.assertNotIn("cos", compact_shared)
        self.assertNotIn("sin", compact_shared)
        self.assertTrue(torch.equal(restored.attention_mask_2d, fake_state.attention_mask_2d))
        self.assertEqual(tuple(restored.attention_mask.shape), (1, 1, 3, 3))
        self.assertTrue(torch.equal(restored.cos, torch.full_like(fake_state.inputs_embeds, 2.0)))
        self.assertTrue(torch.equal(restored.sin, torch.full_like(fake_state.inputs_embeds, 3.0)))
        self.assertTrue(torch.equal(restored.position_ids, fake_state.position_ids))

    def test_prepare_mm_frontend_seed_uses_frontend_helper_not_session_wrapper(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        fake_inputs = _FakeBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        fake_frontend_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_session",
            side_effect=AssertionError("frontend seed 不应再依赖 prepare_mm_session"),
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend_seed_parts",
            return_value=(
                fake_model,
                MmFrontendBatch(
                    raw_inputs=fake_inputs,
                    frame_paths=["/tmp/f0.png", "/tmp/f1.png"],
                ),
                fake_frontend_state,
            ),
        ) as prepare_frontend_mock:
            payload = prepare_mm_frontend_seed(
                {
                    "model_path": "/tmp/fake-model",
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                }
            )

        prepare_frontend_mock.assert_called_once_with(
            {
                "model_path": "/tmp/fake-model",
                "num_frames": 2,
                "frame_dir": "/tmp/frames",
            },
            device=torch.device("cpu"),
        )
        self.assertEqual(payload["frontend_meta"]["num_frames"], 2)
        self.assertEqual(payload["frontend_meta"]["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertIsInstance(payload["frontend_tensors"], dict)
        self.assertTrue(torch.equal(payload["frontend_tensors"]["inputs_embeds"], fake_frontend_state.inputs_embeds))
        self.assertNotIn("input_ids", payload["frontend_tensors"])
        self.assertNotIn("attention_mask_2d", payload["frontend_tensors"])
        self.assertNotIn("mm_token_type_ids", payload["frontend_tensors"])
        self.assertNotIn("image_grid_thw", payload["frontend_tensors"])
        self.assertNotIn("attention_mask", payload["frontend_tensors"])
        self.assertNotIn("cos", payload["frontend_tensors"])
        self.assertNotIn("sin", payload["frontend_tensors"])
        self.assertNotIn("position_ids", payload["frontend_tensors"])
        self.assertNotIn("rope_deltas", payload["frontend_tensors"])
        self.assertIn("runtime", payload["frontend_meta"])
        self.assertTrue(torch.equal(payload["frontend_meta"]["runtime"]["input_ids"], fake_frontend_state.input_ids))
        self.assertTrue(
            torch.equal(
                payload["frontend_meta"]["runtime"]["attention_mask_2d"],
                fake_frontend_state.attention_mask_2d,
            )
        )
        self.assertNotIn("frontend_plan", payload)
        self.assertNotIn("frontend_state", payload)

    def test_prepare_mm_session_builds_live_prefill_inputs(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        fake_frontend_plan = _fake_mm_frontend_plan()
        fake_frontend_seed = {
            "frontend_tensors": compact_mm_frontend_tensors(
                mm_state_from_frontend_plan(fake_frontend_plan),
            ),
            "frontend_meta": {
                "runtime": compact_mm_frontend_meta(
                    mm_state_from_frontend_plan(fake_frontend_plan),
                ),
                "num_frames": 2,
                "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            },
        }

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
            return_value=fake_model,
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend_seed",
            return_value=fake_frontend_seed,
        ) as prepare_frontend_seed_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_text_model_config_spec",
            return_value=_fake_text_config_spec(),
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_mm_frontend_config",
            return_value=type(
                "FakeMmConfig",
                (),
                {"vision_config": type("FakeVisionConfig", (), {"spatial_merge_size": 1})()},
            )(),
        ):
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                }
            )

        load_model_mock.assert_called_once_with("/tmp/fake-model", attn_implementation="eager")
        prepare_frontend_seed_mock.assert_called_once_with(
            {
                "model_path": "/tmp/fake-model",
                "num_frames": 2,
                "frame_dir": "/tmp/frames",
            },
        )
        self.assertIs(session.model, fake_model)
        self.assertTrue(torch.equal(session.raw_inputs["input_ids"], fake_frontend_plan.input_ids))
        self.assertTrue(torch.equal(session.raw_inputs["attention_mask"], fake_frontend_plan.attention_mask_2d))
        self.assertIsNotNone(session.frontend_state)
        self.assertTrue(torch.equal(session.runtime_inputs.input_ids, fake_frontend_plan.input_ids))
        self.assertTrue(torch.equal(session.runtime_inputs.position_ids, fake_frontend_plan.position_ids))
        self.assertTrue(torch.equal(session.runtime_inputs.rope_deltas, fake_frontend_plan.rope_deltas))
        self.assertEqual(session.extra["num_frames"], 2)
        self.assertEqual(session.extra["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertEqual(session.extra["frontend_activation"], "active")
        self.assertEqual(session.extra.get("video_input_metadata", {}), {})

    def test_prepare_mm_session_active_path_runs_frontend_before_decoder_load(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        fake_frontend_plan = _fake_mm_frontend_plan()
        fake_frontend_seed = {
            "frontend_tensors": compact_mm_frontend_tensors(
                mm_state_from_frontend_plan(fake_frontend_plan),
                include_derived=True,
            ),
            "frontend_meta": {
                "runtime": compact_mm_frontend_meta(
                    mm_state_from_frontend_plan(fake_frontend_plan),
                    include_derived=True,
                ),
                "num_frames": 2,
                "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            },
        }
        call_order: list[str] = []

        def _fake_prepare_frontend_seed(runtime_config):
            call_order.append("frontend")
            return fake_frontend_seed

        def _fake_load_model(*args, **kwargs):
            call_order.append("decoder")
            return fake_model

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend_seed",
            side_effect=_fake_prepare_frontend_seed,
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
            side_effect=_fake_load_model,
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_text_model_config_spec",
            return_value=_fake_text_config_spec(),
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_mm_frontend_config",
            return_value=type(
                "FakeMmConfig",
                (),
                {"vision_config": type("FakeVisionConfig", (), {"spatial_merge_size": 1})()},
            )(),
        ):
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                }
            )

        self.assertEqual(call_order, ["frontend", "decoder"])
        self.assertIs(session.model, fake_model)

    def test_prepare_mm_session_reuses_seeded_frontend_state(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        seeded_frontend_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
            return_value=fake_model,
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend",
        ) as prepare_frontend_mock:
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "_mm_frontend_state": seeded_frontend_state,
                    "_mm_num_frames": 2,
                    "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                }
            )

        load_model_mock.assert_called_once_with("/tmp/fake-model", attn_implementation="eager")
        prepare_frontend_mock.assert_not_called()
        self.assertTrue(torch.equal(session.raw_inputs["input_ids"], seeded_frontend_state.input_ids))
        self.assertIsNot(session.frontend_state, seeded_frontend_state)
        self.assertEqual(session.extra["num_frames"], 2)
        self.assertEqual(session.extra["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertEqual(session.extra["frontend_activation"], "seeded")

    def test_prepare_mm_session_reuses_seeded_frontend_tensors(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        seeded_frontend_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
            return_value=fake_model,
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend",
        ) as prepare_frontend_mock:
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "_mm_frontend_seed": seeded_frontend_state,
                    "_mm_num_frames": 2,
                    "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                }
            )

        load_model_mock.assert_called_once_with("/tmp/fake-model", attn_implementation="eager")
        prepare_frontend_mock.assert_not_called()
        self.assertTrue(torch.equal(session.raw_inputs["input_ids"], seeded_frontend_state.input_ids))
        self.assertIsNotNone(session.frontend_state)
        self.assertEqual(session.extra["num_frames"], 2)
        self.assertEqual(session.extra["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertEqual(session.extra["frontend_activation"], "seeded")

    def test_prepare_mm_session_rejects_legacy_frontend_plan(self) -> None:
        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend",
        ) as prepare_frontend_mock:
            with self.assertRaisesRegex(RuntimeError, "legacy `_mm_frontend_plan` 已下线"):
                prepare_mm_session(
                    {
                        "model_path": "/tmp/fake-model",
                        "_mm_frontend_plan": _fake_mm_frontend_plan(),
                        "_mm_num_frames": 2,
                        "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                    },
                    activate_frontend=False,
                )

        load_model_mock.assert_not_called()
        prepare_frontend_mock.assert_not_called()

    def test_restore_mm_frontend_seed_rejects_legacy_frontend_plan_payload(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "legacy frontend_plan payload 已下线"):
            restore_mm_frontend_seed(
                {
                    "frontend_plan": _fake_mm_frontend_plan(),
                    "num_frames": 2,
                    "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                }
            )

    def test_prepare_mm_session_consume_only_requires_seeded_frontend_state(self) -> None:
        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend",
        ) as prepare_frontend_mock:
            with self.assertRaisesRegex(RuntimeError, "consume-only stage"):
                prepare_mm_session(
                    {
                        "model_path": "/tmp/fake-model",
                        "num_frames": 2,
                    },
                    activate_frontend=False,
                )

        load_model_mock.assert_not_called()
        prepare_frontend_mock.assert_not_called()

    def test_prepare_mm_session_can_skip_decoder_model_load(self) -> None:
        seeded_frontend_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
        ) as load_model_mock:
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "_mm_frontend_seed": seeded_frontend_state,
                    "_mm_num_frames": 2,
                    "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                },
                load_decoder_model=False,
            )

        load_model_mock.assert_not_called()
        self.assertIsNone(session.model)
        self.assertTrue(torch.equal(session.raw_inputs["input_ids"], seeded_frontend_state.input_ids))
        self.assertIsNotNone(session.frontend_state)
        self.assertEqual(session.frontend_state.inputs_embeds.device, seeded_frontend_state.inputs_embeds.device)
        self.assertEqual(session.extra["frontend_activation"], "seeded")

    def test_prepare_mm_session_reuses_provided_decoder_model(self) -> None:
        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        seeded_frontend_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.load_model",
        ) as load_model_mock:
            session = prepare_mm_session(
                {
                    "model_path": "/tmp/fake-model",
                    "_mm_frontend_state": seeded_frontend_state,
                    "_mm_num_frames": 2,
                    "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                },
                decoder_model=fake_model,
            )

        load_model_mock.assert_not_called()
        self.assertIs(session.model, fake_model)
        self.assertTrue(torch.equal(session.raw_inputs["input_ids"], seeded_frontend_state.input_ids))
        self.assertEqual(session.extra["frontend_activation"], "seeded")

    def test_build_mm_frontend_batch_uses_processor_and_frame_listing(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import build_mm_frontend_batch

        fake_processor = object()
        fake_inputs = _FakeBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.load_processor",
            return_value=fake_processor,
        ) as load_processor_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.list_frames",
            return_value=["/tmp/f0.png", "/tmp/f1.png"],
        ) as list_frames_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.build_inputs_with_metadata",
            return_value=(fake_inputs, {"frame_count": 2}),
        ) as build_inputs_mock:
            batch = build_mm_frontend_batch(
                {
                    "model_path": "/tmp/fake-model",
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                    "prompt": "custom video prompt",
                }
            )

        load_processor_mock.assert_called_once_with("/tmp/fake-model")
        list_frames_mock.assert_called_once_with(2, "/tmp/frames")
        build_inputs_mock.assert_called_once_with(
            fake_processor,
            ["/tmp/f0.png", "/tmp/f1.png"],
            video_path=None,
            video_url=None,
            prompt="custom video prompt",
            sample_fps=1,
            video_fps=None,
            video_nframes=None,
            video_start=None,
            video_end=None,
            video_min_frames=None,
            video_max_frames=None,
        )
        self.assertIs(batch.raw_inputs, fake_inputs)
        self.assertEqual(batch.frame_paths, ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertEqual(batch.num_frames, 2)

    def test_build_mm_frontend_batch_uses_full_video_without_listing_frames(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import build_mm_frontend_batch

        fake_processor = object()
        fake_inputs = _FakeBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.load_processor",
            return_value=fake_processor,
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.list_frames",
        ) as list_frames_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.build_inputs_with_metadata",
            return_value=(
                fake_inputs,
                {
                    "source": "video_path",
                    "frame_count": 6,
                    "video_path_basename": "sample.mp4",
                },
            ),
        ) as build_inputs_mock:
            batch = build_mm_frontend_batch(
                {
                    "model_path": "/tmp/fake-model",
                    "video_path": "/tmp/sample.mp4",
                    "video_nframes": 6,
                }
            )

        list_frames_mock.assert_not_called()
        build_inputs_mock.assert_called_once_with(
            fake_processor,
            None,
            video_path="/tmp/sample.mp4",
            video_url=None,
            prompt=DEFAULT_VIDEO_PROMPT,
            sample_fps=1,
            video_fps=None,
            video_nframes=6,
            video_start=None,
            video_end=None,
            video_min_frames=None,
            video_max_frames=None,
        )
        self.assertIs(batch.raw_inputs, fake_inputs)
        self.assertEqual(batch.frame_paths, [])
        self.assertEqual(batch.num_frames, 6)
        self.assertEqual(batch.video_input_metadata["video_path_basename"], "sample.mp4")

    def test_prepare_mm_frontend_plan_parts_builds_plan_from_runtime_batch(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import prepare_mm_frontend_plan_parts

        fake_model = type("FakeModel", (), {"device": torch.device("cpu")})()
        fake_batch = MmFrontendBatch(
            raw_inputs=_FakeBatch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long)),
            frame_paths=["/tmp/f0.png", "/tmp/f1.png"],
        )
        moved_inputs = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
        fake_plan = _fake_mm_frontend_plan()
        fake_weight_index = object()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.load_mm_frontend_model",
            return_value=fake_model,
        ) as load_model_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.build_mm_frontend_batch",
            return_value=fake_batch,
        ) as build_batch_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.move_frontend_inputs",
            return_value=moved_inputs,
        ) as move_inputs_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.build_mm_frontend_plan",
            return_value=fake_plan,
        ) as build_plan_mock:
            frontend_model, frontend_batch, frontend_plan = prepare_mm_frontend_plan_parts(
                {
                    "model_path": "/tmp/fake-model",
                    "_mm_weight_index": fake_weight_index,
                    "num_frames": 2,
                    "frame_dir": "/tmp/frames",
                }
            )

        load_model_mock.assert_called_once_with("/tmp/fake-model", weight_index=fake_weight_index)
        build_batch_mock.assert_called_once_with(
            {
                "model_path": "/tmp/fake-model",
                "_mm_weight_index": fake_weight_index,
                "num_frames": 2,
                "frame_dir": "/tmp/frames",
            }
        )
        move_inputs_mock.assert_called_once_with(fake_batch.raw_inputs, device=torch.device("cpu"))
        build_plan_mock.assert_called_once_with(
            fake_model,
            moved_inputs,
            inputs_on_device=True,
        )
        self.assertIs(frontend_model, fake_model)
        self.assertIsNot(frontend_batch, fake_batch)
        self.assertIs(frontend_batch.raw_inputs, moved_inputs)
        self.assertEqual(frontend_batch.frame_paths, ["/tmp/f0.png", "/tmp/f1.png"])
        self.assertIs(frontend_plan, fake_plan)

    def test_prepare_mm_frontend_parts_converts_plan_to_state(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.runtime import prepare_mm_frontend_parts

        fake_model = object()
        fake_batch = MmFrontendBatch(
            raw_inputs={"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
            frame_paths=["/tmp/f0.png"],
        )
        fake_state = _fake_mm_runtime_state()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime._prepare_mm_frontend_batch_on_device",
            return_value=(fake_model, fake_batch),
        ) as prepare_batch_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.runtime.prepare_mm_frontend_state",
            return_value=fake_state,
        ) as prepare_state_mock:
            frontend_model, frontend_batch, frontend_state = prepare_mm_frontend_parts(
                {
                    "model_path": "/tmp/fake-model",
                }
            )

        prepare_batch_mock.assert_called_once_with(
            {
                "model_path": "/tmp/fake-model",
            },
            model=None,
        )
        prepare_state_mock.assert_called_once_with(
            fake_model,
            fake_batch.raw_inputs,
            inputs_on_device=True,
        )
        self.assertIs(frontend_model, fake_model)
        self.assertIs(frontend_batch, fake_batch)
        self.assertIs(frontend_state, fake_state)

    def test_build_mm_frontend_plan_exposes_visual_pos_and_attn_parts(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.state import build_mm_frontend_plan

        class _FakeFrontendLanguageModel:
            def __init__(self) -> None:
                self.config = object()

            def get_input_embeddings(self):
                return lambda input_ids: torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

            def rotary_emb(self, inputs_embeds, vision_position_ids):
                return torch.zeros_like(inputs_embeds), torch.zeros_like(inputs_embeds)

        class _FakeFrontendCore:
            def __init__(self) -> None:
                self.language_model = _FakeFrontendLanguageModel()
                self.rope_deltas = torch.tensor([[3]], dtype=torch.long)

            def compute_3d_position_ids(self, **kwargs):
                return torch.zeros(4, 1, 3, dtype=torch.long)

            def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
                image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
                image_mask[:, 0, :] = True
                video_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
                return image_mask, video_mask

        fake_model = type("FakeModel", (), {"device": torch.device("cpu"), "model": _FakeFrontendCore()})()
        fake_inputs = _FakeBatch(
            input_ids=torch.tensor([[9, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            pixel_values=torch.ones(1, 2),
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
        )

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.encode_image_features",
            return_value=(torch.full((1, 4), 5.0), [torch.full((1, 4), 7.0)]),
        ) as encode_image_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.create_causal_mask",
            return_value=torch.zeros(1, 1, 3, 3),
        ):
            plan = build_mm_frontend_plan(fake_model, fake_inputs)

        encode_image_mock.assert_called_once()
        self.assertEqual(fake_inputs.to_calls, [torch.device("cpu")])
        self.assertTrue(torch.equal(plan.inputs_embeds[0, 0], torch.full((4,), 5.0)))
        self.assertTrue(torch.equal(plan.visual.visual_pos_masks, torch.tensor([[True, False, False]])))
        self.assertEqual(sorted(plan.visual.deepstack_by_layer), [0])
        self.assertTrue(torch.equal(plan.visual.deepstack_by_layer[0], torch.full((1, 4), 7.0)))
        self.assertEqual(tuple(plan.pos.position_ids.shape), (4, 1, 3))
        self.assertEqual(tuple(plan.attn.attention_mask.shape), (1, 1, 3, 3))
        self.assertEqual(tuple(plan.attn.cos.shape), (1, 3, 4))
        self.assertEqual(tuple(plan.attn.sin.shape), (1, 3, 4))
        self.assertTrue(torch.equal(plan.rope_deltas, torch.tensor([[3]], dtype=torch.long)))

    def test_prepare_mm_frontend_state_moves_batch_once(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.state import (
            prepare_mm_frontend_state,
        )

        fake_language_model = type(
            "FakeLanguageModel",
            (),
            {
                "config": object(),
                "get_input_embeddings": lambda self: (lambda input_ids: torch.zeros(1, 3, 4)),
                "rotary_emb": lambda self, inputs_embeds, vision_position_ids: (
                    torch.zeros(1, 3, 4),
                    torch.zeros(1, 3, 4),
                ),
            },
        )()
        fake_model_core = type(
            "FakeModelCore",
            (),
            {
                "language_model": fake_language_model,
                "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
                "compute_3d_position_ids": lambda self, **kwargs: torch.zeros(4, 1, 3, dtype=torch.long),
            },
        )()
        fake_model = type("FakeModel", (), {"device": torch.device("cpu"), "model": fake_model_core})()
        fake_inputs = _FakeBatch(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.create_causal_mask",
            return_value=torch.zeros(1, 1, 3, 3),
        ) as mask_mock:
            state = prepare_mm_frontend_state(fake_model, fake_inputs)

        self.assertEqual(fake_inputs.to_calls, [torch.device("cpu")])
        mask_mock.assert_called_once()
        self.assertTrue(torch.equal(state.input_ids, torch.tensor([[1, 2, 3]], dtype=torch.long)))
        self.assertIsNone(state.visual.visual_pos_masks)
        self.assertEqual(state.visual.deepstack_by_layer, {})

    def test_prepare_mm_frontend_state_merges_image_features_without_bridge(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.state import (
            prepare_mm_frontend_state,
        )

        class _FakeFrontendLanguageModel:
            def __init__(self) -> None:
                self.config = object()

            def get_input_embeddings(self):
                return lambda input_ids: torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

            def rotary_emb(self, inputs_embeds, vision_position_ids):
                return torch.zeros_like(inputs_embeds), torch.zeros_like(inputs_embeds)

        class _FakeFrontendCore:
            def __init__(self) -> None:
                self.language_model = _FakeFrontendLanguageModel()
                self.rope_deltas = torch.zeros(1, 1, dtype=torch.long)

            def compute_3d_position_ids(self, **kwargs):
                return torch.zeros(4, 1, 3, dtype=torch.long)

            def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
                image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
                image_mask[:, 0, :] = True
                video_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
                return image_mask, video_mask

        fake_model = type("FakeModel", (), {"device": torch.device("cpu"), "model": _FakeFrontendCore()})()
        fake_inputs = _FakeBatch(
            input_ids=torch.tensor([[9, 2, 3]], dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            pixel_values=torch.ones(1, 2),
            image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
        )

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.encode_image_features",
            return_value=(torch.full((1, 4), 5.0), [torch.full((1, 4), 7.0)]),
        ) as encode_image_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.create_causal_mask",
            return_value=torch.zeros(1, 1, 3, 3),
        ):
            state = prepare_mm_frontend_state(fake_model, fake_inputs)

        encode_image_mock.assert_called_once()
        self.assertTrue(torch.equal(state.inputs_embeds[0, 0], torch.full((4,), 5.0)))
        self.assertTrue(torch.equal(state.visual.visual_pos_masks, torch.tensor([[True, False, False]])))
        self.assertEqual(sorted(state.visual.deepstack_by_layer), [0])
        self.assertTrue(torch.equal(state.visual.deepstack_by_layer[0], torch.full((1, 4), 7.0)))

    def test_load_mm_frontend_weight_bundle_selectively_loads_frontend_weights(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.weights.vision import (
            load_mm_frontend_weight_bundle,
        )

        fake_index = type(
            "FakeIndex",
            (),
            {"has_tensor": lambda self, name: name == "model.language_model.embed_tokens.weight"},
        )()
        sample_weight = torch.ones(4, 3, dtype=torch.float16)
        loaded_visual_weight = torch.full((2, 2), 2.0, dtype=torch.float16)
        loaded_visual_bias = torch.full((2,), 3.0, dtype=torch.float16)

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.weights.vision.load_tensors_from_index",
            side_effect=[
                {"model.language_model.embed_tokens.weight": sample_weight},
                {
                    "model.language_model.embed_tokens.weight": sample_weight,
                    "model.visual.weight": loaded_visual_weight,
                    "model.visual.bias": loaded_visual_bias,
                },
            ],
        ) as load_tensors_mock:
            bundle = load_mm_frontend_weight_bundle(
                model_path="/tmp/fake-model",
                visual_parameter_names=["weight", "bias"],
                device="cpu",
                weight_index=fake_index,
            )

        self.assertEqual(load_tensors_mock.call_count, 2)
        self.assertEqual(
            load_tensors_mock.call_args_list[0].args[1],
            ["model.language_model.embed_tokens.weight"],
        )
        self.assertEqual(
            set(load_tensors_mock.call_args_list[1].args[1]),
            {
                "model.language_model.embed_tokens.weight",
                "model.visual.weight",
                "model.visual.bias",
            },
        )
        self.assertEqual(load_tensors_mock.call_args_list[1].kwargs["compute_dtype"], torch.float16)
        self.assertEqual(bundle.compute_dtype, torch.float16)
        self.assertTrue(torch.equal(bundle.embed_tokens_weight, sample_weight))
        self.assertEqual(sorted(bundle.visual_state), ["bias", "weight"])
        self.assertTrue(torch.equal(bundle.visual_state["weight"], loaded_visual_weight))
        self.assertTrue(torch.equal(bundle.visual_state["bias"], loaded_visual_bias))

    def test_load_mm_frontend_model_applies_frontend_weight_bundle(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import load_mm_frontend_model

        config = Qwen3VLConfig(
            text_config={"vocab_size": 4, "hidden_size": 3},
            vision_config={"hidden_size": 2, "out_hidden_size": 2},
        )
        sample_weight = torch.ones(4, 3, dtype=torch.float16)
        loaded_visual_weight = torch.full((2, 2), 2.0, dtype=torch.float16)
        loaded_visual_bias = torch.full((2,), 3.0, dtype=torch.float16)
        fake_bundle = type(
            "FakeBundle",
            (),
            {
                "compute_dtype": torch.float16,
                "visual_state": {
                    "weight": loaded_visual_weight,
                    "bias": loaded_visual_bias,
                },
                "embed_tokens_weight": sample_weight,
            },
        )()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.load_mm_frontend_config",
            return_value=config,
        ) as config_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.Qwen3VLVisionModel._from_config",
            return_value=_DummyVision(),
        ) as vision_mock, patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.load_mm_frontend_weight_bundle",
            return_value=fake_bundle,
        ) as load_bundle_mock:
            model = load_mm_frontend_model("/tmp/fake-model", device="cpu")

        config_mock.assert_called_once_with("/tmp/fake-model")
        vision_mock.assert_called_once()
        load_bundle_mock.assert_called_once()
        self.assertEqual(
            set(load_bundle_mock.call_args.kwargs["visual_parameter_names"]),
            {"weight", "bias"},
        )
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(model.model.language_model.embed_tokens.weight.dtype, torch.float16)
        self.assertTrue(torch.equal(model.model.language_model.embed_tokens.weight.detach(), sample_weight))
        self.assertTrue(torch.equal(model.model.visual.weight.detach(), loaded_visual_weight))
        self.assertTrue(torch.equal(model.model.visual.bias.detach(), loaded_visual_bias))

    def test_mm_frontend_video_features_do_not_duplicate_return_dict(self) -> None:
        from qwen3vl_tp_runtime.models.qwen3vl.vision.frontend import MmFrontendModel

        config = Qwen3VLConfig(
            text_config={"vocab_size": 4, "hidden_size": 3},
            vision_config={"hidden_size": 2, "out_hidden_size": 2},
        )
        fake_visual = _ForwardVision()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.vision.frontend.Qwen3VLVisionModel._from_config",
            return_value=fake_visual,
        ):
            model = MmFrontendModel(config)

        outputs = model.model.get_video_features(
            pixel_values_videos=torch.ones(1, 3, 2, 2, 2, dtype=torch.float32),
            video_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
            return_dict=True,
        )

        self.assertEqual(len(fake_visual.calls), 1)
        self.assertIs(outputs.pooler_output.__class__, tuple)
        self.assertEqual(len(outputs.pooler_output), 1)
        self.assertTrue(torch.equal(outputs.pooler_output[0], torch.ones(1, 4)))
        self.assertIs(fake_visual.calls[0]["return_dict"], True)

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
