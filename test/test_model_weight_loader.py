from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec
from qwen3vl_tp_runtime.models.qwen3vl.runtime_builder import (
    DirectStageBundleBuilder,
    materialize_text_stage,
    pack_text_scaffold_transport,
    prepare_text_prompt_meta,
    restore_text_scaffold_transport,
    seed_mm_startup_runtime_config,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_text_stage import (
    assert_text_tp_shard_shapes,
    assert_text_weight_scope,
    summarize_text_weight_load,
)
from qwen3vl_tp_runtime.models.qwen3vl.runtime_mm_stage import (
    MmRuntimeState,
    MmVisualState,
    build_mm_decode_state_from_weights,
    compact_mm_frontend_meta,
    compact_mm_frontend_tensors,
    compact_mm_runtime_shared,
    mm_state_from_decode_state,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    ModelWeightIndex,
    TensorSliceSpec,
    TextTensorParallelShardPlan,
    build_text_causal_mask,
    build_text_decoder_stage_weight_plan,
    build_text_decoder_stage_tp_sharded_parameter_names,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
    load_model_weight_index,
    load_tensors_by_name,
    prepare_text_decode_runtime_inputs_from_weights,
    prepare_text_prefill_runtime_inputs_from_weights,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights.loader import (
    load_tensors_from_index as _real_load_tensors_from_index,
)


class ModelWeightLoaderTest(unittest.TestCase):
    def _write_text_config(self, model_dir: Path, **overrides) -> None:
        payload = {
            "tie_word_embeddings": overrides.pop("tie_word_embeddings", True),
            "pad_token_id": overrides.pop("pad_token_id", None),
            "text_config": {
                "hidden_size": overrides.pop("hidden_size", 8),
                "intermediate_size": overrides.pop("intermediate_size", 16),
                "num_hidden_layers": overrides.pop("num_hidden_layers", 2),
                "num_attention_heads": overrides.pop("num_attention_heads", 2),
                "num_key_value_heads": overrides.pop("num_key_value_heads", 1),
                "head_dim": overrides.pop("head_dim", 4),
                "rms_norm_eps": overrides.pop("rms_norm_eps", 1e-6),
                "hidden_act": overrides.pop("hidden_act", "silu"),
                "vocab_size": overrides.pop("vocab_size", 6),
                "max_position_embeddings": overrides.pop("max_position_embeddings", 64),
                "rope_scaling": overrides.pop(
                    "rope_scaling",
                    {
                        "rope_type": "default",
                        "mrope_interleaved": True,
                        "mrope_section": [1, 0, 0],
                    },
                ),
                "rope_theta": overrides.pop("rope_theta", 10000.0),
                "attention_bias": overrides.pop("attention_bias", False),
                "attention_dropout": overrides.pop("attention_dropout", 0.0),
                "use_cache": overrides.pop("use_cache", True),
            },
        }
        if overrides:
            self.fail(f"unexpected config overrides: {sorted(overrides)}")
        (model_dir / "config.json").write_text(json.dumps(payload))

    def _write_two_layer_text_checkpoint(self, model_dir: Path) -> None:
        shard = model_dir / "pytorch_model.bin"
        torch.save(
            {
                "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(2, 4),
                "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(2, 4),
                "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                "model.language_model.norm.weight": torch.ones(4),
            },
            shard,
        )

    def _build_small_mm_frontend_state(
        self,
        model_dir: Path,
    ) -> tuple[TextModelConfigSpec, torch.Tensor, MmRuntimeState]:
        config = load_text_model_config_spec(str(model_dir))
        embed_tokens_weight = load_tensors_by_name(
            str(model_dir),
            ["model.language_model.embed_tokens.weight"],
        )["model.language_model.embed_tokens.weight"]
        prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
            input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
            embed_tokens_weight=embed_tokens_weight,
            config_spec=config,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )
        frontend_state = MmRuntimeState(
            input_ids=prefill_inputs.input_ids,
            attention_mask_2d=prefill_inputs.attention_mask_2d,
            position_ids=prefill_inputs.position_ids,
            inputs_embeds=prefill_inputs.inputs_embeds,
            attention_mask=prefill_inputs.attention_mask,
            cos=prefill_inputs.cos,
            sin=prefill_inputs.sin,
            visual=MmVisualState(
                visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                deepstack_by_layer={},
            ),
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        )
        return config, embed_tokens_weight, frontend_state

    def _seed_small_mm_startup_contract(
        self,
        runtime_config: dict[str, object],
        frontend_state: MmRuntimeState,
        *,
        stage_idx: int = 1,
        stage_input: torch.Tensor | None = None,
        stage_output: torch.Tensor | None = None,
        num_frames: int = 2,
        frame_paths: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage_input is None:
            stage_input = torch.full_like(frontend_state.inputs_embeds, 3.0 + float(stage_idx))
        if stage_output is None:
            stage_output = torch.full_like(frontend_state.inputs_embeds, 7.0 + float(stage_idx))
        seed_mm_startup_runtime_config(
            runtime_config,
            {
                "shared": compact_mm_runtime_shared(frontend_state),
                "stage_handoffs": {
                    stage_idx: {
                        "stage_input": stage_input.detach().clone(),
                        "stage_output": stage_output.detach().clone(),
                    },
                },
                "stage_visuals": {
                    stage_idx: {
                        "visual_pos_masks": None,
                        "deepstack_by_layer": {},
                    },
                },
                "num_frames": num_frames,
                "frame_paths": list(frame_paths or ["/tmp/f0.png", "/tmp/f1.png"]),
            },
            local_stage_indices=[stage_idx],
        )
        return stage_input, stage_output

    def test_assert_text_weight_scope_rejects_unrelated_decoder_layer(self) -> None:
        bundle = {
            "start_idx": 18,
            "end_idx": 35,
            "layers": [
                {
                    "layer_idx": 17,
                    "q_weight": torch.ones(4, 4),
                },
                {
                    "layer_idx": 18,
                    "q_weight": torch.ones(4, 4),
                },
            ],
        }

        weight_load = summarize_text_weight_load(bundle)

        self.assertFalse(weight_load["stage_weight_scope_ok"])
        self.assertEqual(weight_load["unexpected_layer_indices"], [17])
        with self.assertRaisesRegex(RuntimeError, "非本 stage"):
            assert_text_weight_scope(bundle)

    def test_assert_text_tp_shard_shapes_rejects_full_projection_tensor(self) -> None:
        bundle = {
            "tp_weight_sharded": True,
            "tp_shard_rank": 0,
            "tp_shard_world_size": 2,
            "layers": [
                {
                    "layer_idx": 0,
                    "head_dim": 2,
                    "tp_local_num_attention_heads": 2,
                    "tp_local_num_key_value_heads": 1,
                    "tp_local_intermediate_size": 8,
                    "input_ln_weight": torch.ones(8),
                    "q_weight": torch.zeros(8, 8),
                },
            ],
        }

        weight_load = summarize_text_weight_load(bundle)

        self.assertFalse(weight_load["tp_shard_shape_ok"])
        self.assertEqual(weight_load["tp_sharded_projection_check_count"], 1)
        self.assertEqual(
            weight_load["tp_shard_shape_mismatches"][0]["expected_shape"],
            [4, 8],
        )
        self.assertEqual(
            weight_load["tp_shard_shape_mismatches"][0]["actual_shape"],
            [8, 8],
        )
        with self.assertRaisesRegex(RuntimeError, "投影权重形状"):
            assert_text_tp_shard_shapes(bundle)

    def test_load_tensors_from_torch_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            shard1 = model_dir / "pytorch_model-00001-of-00002.bin"
            shard2 = model_dir / "pytorch_model-00002-of-00002.bin"

            embed = torch.arange(12, dtype=torch.float32).view(3, 4)
            q_proj = torch.arange(16, dtype=torch.float32).view(4, 4)
            norm = torch.ones(4, dtype=torch.float32)

            torch.save(
                {
                    "model.language_model.embed_tokens.weight": embed,
                    "model.language_model.layers.0.self_attn.q_proj.weight": q_proj,
                },
                shard1,
            )
            torch.save(
                {
                    "model.language_model.norm.weight": norm,
                },
                shard2,
            )

            index_payload = {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": shard1.name,
                    "model.language_model.layers.0.self_attn.q_proj.weight": shard1.name,
                    "model.language_model.norm.weight": shard2.name,
                },
            }
            (model_dir / "pytorch_model.bin.index.json").write_text(json.dumps(index_payload))

            index = load_model_weight_index(str(model_dir))
            self.assertEqual(index.format, "torch_bin")
            self.assertEqual(
                set(index.files_for_tensors(index.tensor_names)),
                {str(shard1), str(shard2)},
            )

            loaded = load_tensors_by_name(
                str(model_dir),
                [
                    "model.language_model.embed_tokens.weight",
                    "model.language_model.norm.weight",
                ],
            )
            self.assertTrue(torch.equal(loaded["model.language_model.embed_tokens.weight"], embed))
            self.assertTrue(torch.equal(loaded["model.language_model.norm.weight"], norm))

    def test_load_tensors_from_index_uses_safetensors_slice_reader_for_sharded_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sharded = torch.arange(16, dtype=torch.float32).view(4, 4)
            replicated = torch.arange(4, dtype=torch.float32)
            index = ModelWeightIndex(
                model_path=tmpdir,
                format="safetensors",
                index_file=None,
                weight_map={
                    "sharded.weight": "model.safetensors",
                    "replicated.weight": "model.safetensors",
                },
                metadata={},
            )
            calls = []

            class _FakeSlice:
                def __init__(self, name: str, tensor: torch.Tensor) -> None:
                    self.name = name
                    self.tensor = tensor

                def __getitem__(self, item):
                    calls.append(("slice_getitem", self.name, item))
                    return self.tensor[item]

            class _FakeHandle:
                def __enter__(self):
                    return self

                def __exit__(self, _exc_type, _exc, _tb) -> None:
                    return None

                def get_slice(self, name: str):
                    calls.append(("get_slice", name))
                    return _FakeSlice(name, {"sharded.weight": sharded}[name])

                def get_tensor(self, name: str):
                    calls.append(("get_tensor", name))
                    return {"replicated.weight": replicated}[name]

            def _fake_safe_open(_path: str, *, framework: str, device: str):
                self.assertEqual(framework, "pt")
                self.assertEqual(device, "cpu")
                return _FakeHandle()

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.weights.loader._resolve_safe_open",
                return_value=_fake_safe_open,
            ):
                loaded = _real_load_tensors_from_index(
                    index,
                    ["sharded.weight", "replicated.weight"],
                    tensor_slices={
                        "sharded.weight": (TensorSliceSpec(dim=0, start=2, end=4),),
                    },
                )

            self.assertTrue(torch.equal(loaded["sharded.weight"], sharded[2:4]))
            self.assertTrue(torch.equal(loaded["replicated.weight"], replicated))
            self.assertIn(("get_slice", "sharded.weight"), calls)
            self.assertIn(("get_tensor", "replicated.weight"), calls)
            self.assertNotIn(("get_tensor", "sharded.weight"), calls)

    def test_text_stage_weight_plan_uses_tied_lm_head_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(2),
                    "model.language_model.norm.weight": torch.ones(2),
                },
                shard,
            )

            index = load_model_weight_index(str(model_dir))
            plan = build_text_decoder_stage_weight_plan(
                index,
                start_idx=0,
                end_idx=0,
                is_first_stage=True,
                is_last_stage=True,
            )

            self.assertIn("model.language_model.embed_tokens.weight", plan.resolved_parameter_names)
            self.assertEqual(
                plan.shared_parameter_aliases["lm_head.weight"],
                "model.language_model.embed_tokens.weight",
            )

    def test_text_stage_weight_bundle_loads_tied_lm_head_from_embed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            shard = model_dir / "pytorch_model.bin"
            embed = torch.arange(12, dtype=torch.float32).view(3, 4)
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": embed,
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(4),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=3,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )

            index = load_model_weight_index(str(model_dir))
            config = load_text_model_config_spec(str(model_dir))
            stage_weights = load_text_decoder_stage_weight_bundle(
                model_path=str(model_dir),
                start_idx=1,
                end_idx=1,
                is_first_stage=False,
                is_last_stage=True,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
                weight_index=index,
                config_spec=config,
            )

            self.assertEqual(len(stage_weights.layer_bundles), 1)
            self.assertTrue(torch.equal(stage_weights.lm_head_weight, embed))
            self.assertIsNone(stage_weights.lm_head_bias)
            self.assertTrue(torch.equal(stage_weights.final_norm_weight, torch.ones(4)))

    def test_load_text_model_config_spec_keeps_rope_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                rope_theta=5000000.0,
                attention_bias=True,
                attention_dropout=0.125,
                pad_token_id=42,
            )

            config = load_text_model_config_spec(str(model_dir))

            self.assertEqual(config.max_position_embeddings, 64)
            self.assertEqual(config.rope_parameters["rope_theta"], 5000000.0)
            self.assertEqual(config.rope_parameters["mrope_section"], [1, 0, 0])
            self.assertTrue(config.attention_bias)
            self.assertEqual(config.attention_dropout, 0.125)
            self.assertEqual(config.pad_token_id, 42)

    def test_prepare_text_prompt_meta_prefers_tokenizer_backend(self) -> None:
        runtime_config = {
            "model_path": "/tmp/fake-model",
            "prompt": "hello",
        }
        class _FakeEncoding:
            def __init__(self) -> None:
                self.ids = [1, 2, 3]

        class _FakeTokenizerBackend:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def encode(self, text: str):
                self.calls.append(text)
                return _FakeEncoding()

        tokenizer_backend = _FakeTokenizerBackend()

        with patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_text_tokenizer_backend",
            return_value=tokenizer_backend,
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_text_tokenizer",
            side_effect=AssertionError("backend 路径命中后不应回退到 AutoTokenizer"),
        ), patch(
            "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_processor",
            side_effect=AssertionError("backend 路径命中后不应回退到 processor"),
        ):
            metadata = prepare_text_prompt_meta(runtime_config)

        self.assertTrue(torch.equal(metadata["input_ids"], torch.tensor([[1, 2, 3]], dtype=torch.long)))
        self.assertIsNone(metadata["attention_mask"])
        self.assertEqual(
            tokenizer_backend.calls,
            ["<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"],
        )

    def test_load_text_stage_weight_bundle_tp_shards_tensor_parallel_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            q_weight = torch.arange(64, dtype=torch.float32).view(8, 8)
            q_bias = torch.arange(8, dtype=torch.float32)
            k_weight = torch.arange(32, dtype=torch.float32).view(4, 8) + 100
            k_bias = torch.arange(4, dtype=torch.float32) + 200
            v_weight = torch.arange(32, dtype=torch.float32).view(4, 8) + 300
            v_bias = torch.arange(4, dtype=torch.float32) + 400
            o_weight = torch.arange(64, dtype=torch.float32).view(8, 8) + 500
            o_bias = torch.arange(8, dtype=torch.float32) + 600
            gate_weight = torch.arange(128, dtype=torch.float32).view(16, 8) + 700
            gate_bias = torch.arange(16, dtype=torch.float32) + 800
            up_weight = torch.arange(128, dtype=torch.float32).view(16, 8) + 900
            up_bias = torch.arange(16, dtype=torch.float32) + 1000
            down_weight = torch.arange(128, dtype=torch.float32).view(8, 16) + 1100
            down_bias = torch.arange(8, dtype=torch.float32) + 1200
            lm_head_weight = torch.arange(32, dtype=torch.float32).view(4, 8) + 1300
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": q_weight,
                    "model.language_model.layers.0.self_attn.q_proj.bias": q_bias,
                    "model.language_model.layers.0.self_attn.k_proj.weight": k_weight,
                    "model.language_model.layers.0.self_attn.k_proj.bias": k_bias,
                    "model.language_model.layers.0.self_attn.v_proj.weight": v_weight,
                    "model.language_model.layers.0.self_attn.v_proj.bias": v_bias,
                    "model.language_model.layers.0.self_attn.o_proj.weight": o_weight,
                    "model.language_model.layers.0.self_attn.o_proj.bias": o_bias,
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": gate_weight,
                    "model.language_model.layers.0.mlp.gate_proj.bias": gate_bias,
                    "model.language_model.layers.0.mlp.up_proj.weight": up_weight,
                    "model.language_model.layers.0.mlp.up_proj.bias": up_bias,
                    "model.language_model.layers.0.mlp.down_proj.weight": down_weight,
                    "model.language_model.layers.0.mlp.down_proj.bias": down_bias,
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                    "lm_head.weight": lm_head_weight,
                },
                shard,
            )

            captured_tensor_slices = {}

            def _recording_loader(index, tensor_names, **kwargs):
                tensor_slices = kwargs.get("tensor_slices")
                self.assertIsNotNone(tensor_slices)
                captured_tensor_slices.update(tensor_slices)
                return _real_load_tensors_from_index(index, tensor_names, **kwargs)

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.weights.text.load_tensors_from_index",
                side_effect=_recording_loader,
            ):
                stage_weights = load_text_decoder_stage_weight_bundle(
                    model_path=str(model_dir),
                    start_idx=0,
                    end_idx=0,
                    is_first_stage=True,
                    is_last_stage=True,
                    device=torch.device("cpu"),
                    compute_dtype=torch.float32,
                    config_spec=load_text_model_config_spec(str(model_dir)),
                    tp_shard_rank=1,
                    tp_shard_world_size=2,
                )

            layer_bundle = stage_weights.layer_bundles[0]
            expected_sharded_names = set(
                build_text_decoder_stage_tp_sharded_parameter_names(start_idx=0, end_idx=0)
            )
            self.assertTrue(stage_weights.tp_weight_sharded)
            self.assertEqual(stage_weights.tp_shard_rank, 1)
            self.assertEqual(stage_weights.tp_shard_world_size, 2)
            self.assertEqual(set(captured_tensor_slices), expected_sharded_names)
            self.assertEqual(set(stage_weights.tp_sharded_parameter_names), expected_sharded_names)
            self.assertIn("model.language_model.embed_tokens.weight", stage_weights.tp_replicated_parameter_names)
            self.assertIn("model.language_model.layers.0.self_attn.o_proj.bias", stage_weights.tp_replicated_parameter_names)
            self.assertIn("model.language_model.layers.0.mlp.down_proj.bias", stage_weights.tp_replicated_parameter_names)
            self.assertIn("model.language_model.norm.weight", stage_weights.tp_replicated_parameter_names)
            self.assertIn("lm_head.weight", stage_weights.tp_replicated_parameter_names)
            self.assertNotIn("model.language_model.embed_tokens.weight", captured_tensor_slices)
            self.assertNotIn("model.language_model.layers.0.self_attn.o_proj.bias", captured_tensor_slices)
            self.assertNotIn("model.language_model.layers.0.mlp.down_proj.bias", captured_tensor_slices)
            self.assertNotIn("lm_head.weight", captured_tensor_slices)
            self.assertEqual(layer_bundle["tp_local_num_attention_heads"], 2)
            self.assertEqual(layer_bundle["tp_local_num_key_value_heads"], 1)
            self.assertEqual(layer_bundle["tp_local_intermediate_size"], 8)
            self.assertTrue(torch.equal(layer_bundle["q_weight"], q_weight[4:8]))
            self.assertTrue(torch.equal(layer_bundle["q_bias"], q_bias[4:8]))
            self.assertTrue(torch.equal(layer_bundle["k_weight"], k_weight[2:4]))
            self.assertTrue(torch.equal(layer_bundle["v_weight"], v_weight[2:4]))
            self.assertTrue(torch.equal(layer_bundle["o_weight"], o_weight[:, 4:8]))
            self.assertTrue(torch.equal(layer_bundle["o_bias"], o_bias))
            self.assertTrue(torch.equal(layer_bundle["gate_weight"], gate_weight[8:16]))
            self.assertTrue(torch.equal(layer_bundle["gate_bias"], gate_bias[8:16]))
            self.assertTrue(torch.equal(layer_bundle["up_weight"], up_weight[8:16]))
            self.assertTrue(torch.equal(layer_bundle["down_weight"], down_weight[:, 8:16]))
            self.assertTrue(torch.equal(layer_bundle["down_bias"], down_bias))
            self.assertTrue(torch.equal(stage_weights.lm_head_weight, lm_head_weight))

    def test_load_text_stage_weight_bundle_tp_requires_slice_plan_for_shardable_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            torch.save(
                {
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8),
                },
                model_dir / "pytorch_model.bin",
            )
            incomplete_plan = TextTensorParallelShardPlan(
                rank=0,
                world_size=2,
                local_num_attention_heads=2,
                local_num_key_value_heads=1,
                local_intermediate_size=8,
                tensor_slices={},
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.weights.text.build_text_decoder_stage_tp_shard_plan",
                return_value=incomplete_plan,
            ):
                with self.assertRaisesRegex(RuntimeError, "缺少必须的 tensor slice"):
                    load_text_decoder_stage_weight_bundle(
                        model_path=str(model_dir),
                        start_idx=0,
                        end_idx=0,
                        is_first_stage=False,
                        is_last_stage=False,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                        config_spec=load_text_model_config_spec(str(model_dir)),
                        tp_shard_rank=0,
                        tp_shard_world_size=2,
                    )

    def test_prepare_text_prefill_runtime_inputs_from_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(model_dir)
            config = load_text_model_config_spec(str(model_dir))

            embed = torch.arange(48, dtype=torch.float32).view(6, 8)
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            attention_mask_2d = torch.tensor([[1, 1, 1]], dtype=torch.long)

            runtime_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=input_ids,
                attention_mask_2d=attention_mask_2d,
                embed_tokens_weight=embed,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )

            self.assertEqual(tuple(runtime_inputs.input_ids.shape), (1, 3))
            self.assertEqual(tuple(runtime_inputs.inputs_embeds.shape), (1, 3, 8))
            self.assertEqual(tuple(runtime_inputs.position_ids.shape), (4, 1, 3))
            self.assertEqual(tuple(runtime_inputs.attention_mask.shape), (1, 1, 3, 3))
            self.assertEqual(tuple(runtime_inputs.cos.shape), (1, 3, 4))
            self.assertEqual(tuple(runtime_inputs.sin.shape), (1, 3, 4))
            self.assertTrue(torch.all(runtime_inputs.attention_mask[0, 0, 0, 1:] < 0))
            self.assertTrue(torch.all(runtime_inputs.attention_mask[0, 0, 1, 2:] < 0))
            self.assertEqual(float(runtime_inputs.attention_mask[0, 0, 2, 2].item()), 0.0)

    def test_prepare_text_decode_runtime_inputs_from_weights_uses_past_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(model_dir)
            config = load_text_model_config_spec(str(model_dir))

            embed = torch.arange(48, dtype=torch.float32).view(6, 8)
            decode_input_ids = torch.tensor([[4]], dtype=torch.long)
            attention_mask_2d = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

            runtime_inputs = prepare_text_decode_runtime_inputs_from_weights(
                decode_input_ids=decode_input_ids,
                attention_mask_2d=attention_mask_2d,
                past_length=3,
                embed_tokens_weight=embed,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )

            self.assertEqual(tuple(runtime_inputs.inputs_embeds.shape), (1, 1, 8))
            self.assertEqual(tuple(runtime_inputs.position_ids.shape), (4, 1, 1))
            self.assertEqual(tuple(runtime_inputs.attention_mask.shape), (1, 1, 1, 4))
            self.assertEqual(tuple(runtime_inputs.cos.shape), (1, 1, 4))
            self.assertEqual(tuple(runtime_inputs.sin.shape), (1, 1, 4))
            self.assertTrue(torch.equal(runtime_inputs.attention_mask, torch.zeros_like(runtime_inputs.attention_mask)))

    def test_build_text_causal_mask_validates_total_sequence_length(self) -> None:
        inputs_embeds = torch.zeros((1, 1, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "attention_mask_2d 的长度和 past/query 长度不匹配"):
            build_text_causal_mask(
                inputs_embeds,
                attention_mask_2d=torch.ones((1, 3), dtype=torch.long),
                past_length=3,
            )

    def test_direct_stage_bundle_builder_prefill_works_without_live_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(16, dtype=torch.float32).view(4, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            fake_inputs = {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_processor", return_value=object()):
                with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.build_text_inputs", return_value=fake_inputs):
                    with DirectStageBundleBuilder(stage_specs=[stage_spec], runtime_config=runtime_config) as builder:
                        self.assertFalse(hasattr(builder, "model"))
                        bundle = builder.build_stage_bundle(0)

            self.assertEqual(bundle["module_name"], "text_prefill_stage")
            self.assertEqual(tuple(bundle["stage_input"].shape), (1, 3, 4))
            self.assertEqual(tuple(bundle["stage_output"].shape), (1, 3, 4))
            self.assertEqual(tuple(bundle["embed_tokens_weight"].shape), (4, 4))
            self.assertIn("final_norm_weight", bundle)
            self.assertIn("lm_head_weight", bundle)

    def test_direct_stage_bundle_builder_prefill_supports_tp_sharded_text_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            fake_inputs = {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_processor", return_value=object()):
                with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.build_text_inputs", return_value=fake_inputs):
                    with DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                        tp_shard_rank=1,
                        tp_shard_world_size=2,
                    ) as builder:
                        bundle = builder.build_stage_bundle(0)

            self.assertTrue(bundle["tp_weight_sharded"])
            self.assertEqual(bundle["tp_shard_rank"], 1)
            self.assertEqual(bundle["tp_shard_world_size"], 2)
            self.assertEqual(tuple(bundle["layers"][0]["q_weight"].shape), (4, 8))
            self.assertEqual(tuple(bundle["layers"][0]["k_weight"].shape), (2, 8))
            self.assertEqual(tuple(bundle["layers"][0]["o_weight"].shape), (8, 4))
            self.assertEqual(tuple(bundle["layers"][0]["gate_weight"].shape), (8, 8))
            self.assertEqual(tuple(bundle["layers"][0]["down_weight"].shape), (8, 8))

    def test_direct_stage_bundle_builder_prefill_can_build_text_scaffold_without_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )
            fake_inputs = {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

            with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_processor", return_value=object()):
                with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.build_text_inputs", return_value=fake_inputs):
                    with DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                        include_text_weights=False,
                    ) as builder:
                        scaffold = builder.build_stage_bundle(0)

            self.assertEqual(scaffold["layers"], [])
            self.assertTrue(scaffold["runtime_inputs_local_rebuild"])
            self.assertEqual(scaffold["runtime_prefill_cache_policy"], "recompute")
            self.assertNotIn("cache_by_layer", scaffold)
            self.assertNotIn("embed_tokens_weight", scaffold)
            self.assertNotIn("final_norm_weight", scaffold)
            self.assertNotIn("lm_head_weight", scaffold)
            self.assertNotIn("attention_mask", scaffold["prefill"])
            self.assertNotIn("cos", scaffold["prefill"])
            self.assertNotIn("sin", scaffold["prefill"])
            self.assertNotIn("position_ids", scaffold["decode_steps"][0])
            self.assertNotIn("attention_mask", scaffold["decode_steps"][0])
            self.assertNotIn("cos", scaffold["decode_steps"][0])
            self.assertNotIn("sin", scaffold["decode_steps"][0])

    def test_text_scaffold_transport_roundtrip_restores_tensors(self) -> None:
        scaffold = {
            "save_dtype": "float32",
            "start_idx": 0,
            "end_idx": 0,
            "layers": [],
            "runtime_inputs_local_rebuild": True,
            "prefill": {
                "stage_input": torch.ones(1, 3, 4, dtype=torch.float32),
                "attention_mask_2d": torch.ones(1, 3, dtype=torch.int64),
            },
            "decode_steps": [
                {
                    "stage_input": torch.full((1, 1, 4), 2.0, dtype=torch.float32),
                    "attention_mask_2d": torch.ones(1, 4, dtype=torch.int64),
                }
            ],
        }

        meta, tensor_payload = pack_text_scaffold_transport(scaffold)
        restored = restore_text_scaffold_transport(meta, tensor_payload)

        self.assertEqual(restored["save_dtype"], "float32")
        self.assertTrue(torch.equal(restored["prefill"]["stage_input"], scaffold["prefill"]["stage_input"]))
        self.assertTrue(
            torch.equal(
                restored["decode_steps"][0]["attention_mask_2d"],
                scaffold["decode_steps"][0]["attention_mask_2d"],
            )
        )

    def test_materialize_text_stage_restores_tp_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )
            fake_inputs = {
                "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

            with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_processor", return_value=object()):
                with patch("qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.build_text_inputs", return_value=fake_inputs):
                    with DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                        include_text_weights=False,
                    ) as builder:
                        scaffold = builder.build_stage_bundle(0)
                    with DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                        tp_shard_rank=1,
                        tp_shard_world_size=2,
                    ) as builder:
                        reference_bundle = builder.build_stage_bundle(0)

            materialized = materialize_text_stage(
                stage_bundle_scaffold=scaffold,
                runtime_config=runtime_config,
                compute_dtype=torch.float32,
                tp_shard_rank=1,
                tp_shard_world_size=2,
            )

            self.assertTrue(materialized["tp_weight_sharded"])
            self.assertEqual(materialized["tp_shard_rank"], 1)
            self.assertEqual(materialized["tp_shard_world_size"], 2)
            self.assertEqual(tuple(materialized["layers"][0]["q_weight"].shape), (4, 8))
            self.assertEqual(tuple(materialized["layers"][0]["k_weight"].shape), (2, 8))
            self.assertEqual(tuple(materialized["layers"][0]["o_weight"].shape), (8, 4))
            self.assertEqual(tuple(materialized["layers"][0]["gate_weight"].shape), (8, 8))
            self.assertEqual(tuple(materialized["layers"][0]["down_weight"].shape), (8, 8))
            self.assertIn("embed_tokens_weight", materialized)
            self.assertIn("final_norm_weight", materialized)
            self.assertIn("lm_head_weight", materialized)
            self.assertNotIn("cache_by_layer", materialized)
            self.assertIn(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                materialized["tp_sharded_parameter_names"],
            )
            self.assertIn(
                "model.language_model.layers.0.input_layernorm.weight",
                materialized["tp_replicated_parameter_names"],
            )
            weight_load = summarize_text_weight_load(materialized)
            self.assertTrue(weight_load["tp_weight_sharded"])
            self.assertEqual(weight_load["tp_shard_rank"], 1)
            self.assertEqual(weight_load["tp_shard_world_size"], 2)
            self.assertTrue(weight_load["tp_shard_shape_ok"])
            self.assertEqual(weight_load["tp_shard_shape_mismatches"], [])
            self.assertEqual(weight_load["tp_sharded_projection_check_count"], 7)
            projection_shapes = {
                example["name"]: example["actual_shape"]
                for example in weight_load["tp_sharded_projection_examples"]
            }
            self.assertEqual(projection_shapes["layers.0.q_weight"], [4, 8])
            self.assertEqual(projection_shapes["layers.0.k_weight"], [2, 8])
            self.assertEqual(projection_shapes["layers.0.v_weight"], [2, 8])
            self.assertEqual(projection_shapes["layers.0.o_weight"], [8, 4])
            self.assertEqual(projection_shapes["layers.0.gate_weight"], [8, 8])
            self.assertEqual(projection_shapes["layers.0.up_weight"], [8, 8])
            self.assertEqual(projection_shapes["layers.0.down_weight"], [8, 8])
            self.assertEqual(weight_load["stage_start_idx"], 0)
            self.assertEqual(weight_load["stage_end_idx"], 0)
            self.assertEqual(weight_load["loaded_layer_indices"], [0])
            self.assertTrue(weight_load["stage_weight_scope_ok"])
            self.assertEqual(weight_load["unexpected_layer_indices"], [])
            self.assertGreater(weight_load["loaded_weight_tensor_count"], 0)
            self.assertGreater(weight_load["loaded_weight_tensor_bytes"], 0)
            self.assertEqual(
                weight_load["tp_sharded_parameter_count"],
                len(materialized["tp_sharded_parameter_names"]),
            )
            self.assertIn("attention_mask", materialized["prefill"])
            self.assertIn("cos", materialized["prefill"])
            self.assertIn("sin", materialized["prefill"])
            self.assertIn("position_ids", materialized["decode_steps"][0])
            self.assertIn("attention_mask", materialized["decode_steps"][0])
            self.assertIn("cos", materialized["decode_steps"][0])
            self.assertIn("sin", materialized["decode_steps"][0])
            self.assertTrue(
                torch.equal(
                    materialized["prefill"]["attention_mask"],
                    reference_bundle["prefill"]["attention_mask"],
                )
            )
            self.assertEqual(tuple(materialized["decode_steps"][0]["position_ids"].shape), (4, 1, 1))
            self.assertTrue(
                torch.allclose(
                    materialized["prefill"]["cos"],
                    reference_bundle["prefill"]["cos"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    materialized["decode_steps"][0]["sin"],
                    reference_bundle["decode_steps"][0]["sin"],
                )
            )

    def test_direct_stage_bundle_builder_generate_runtime_only_skips_reference_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
                "include_runtime_reference": False,
                "_runtime_only_input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "_runtime_only_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch.object(
                DirectStageBundleBuilder,
                "_capture_prefill_stage_boundaries",
                side_effect=AssertionError("runtime-only generate 不应触发 prefill boundary capture"),
            ), patch.object(
                DirectStageBundleBuilder,
                "_ensure_generate_state",
                side_effect=AssertionError("runtime-only generate 不应触发 generate reference state"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    bundle = builder.build_stage_bundle(0)

            self.assertTrue(bundle["runtime_only_generate"])
            self.assertEqual(tuple(bundle["prefill_attention_mask_2d"].shape), (1, 3))
            self.assertEqual(bundle["prefill_seq_len"], 3)
            self.assertEqual(bundle["batch_size"], 1)
            self.assertEqual(bundle["token_id_dtype"], "int64")
            self.assertNotIn("prefill", bundle)
            self.assertNotIn("decode_steps", bundle)
            self.assertNotIn("generated_token_ids", bundle)
            self.assertNotIn("cache_by_layer", bundle)
            self.assertNotIn("prompt", bundle)
            self.assertNotIn("prefill_input_ids", bundle)
            self.assertIn("input_ids", bundle)
            self.assertIn("embed_tokens_weight", bundle)
            self.assertIn("final_norm_weight", bundle)
            self.assertIn("lm_head_weight", bundle)

    def test_runtime_only_text_generate_scaffold_uses_lightweight_session_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "auto",
                "prompt": "hello",
                "max_new_tokens": 3,
                "include_runtime_reference": False,
                "_runtime_only_input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "_runtime_only_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="auto",
                bundle_path=None,
            )
            load_calls: list[tuple[str, ...]] = []

            def _fake_load_tensors_from_index(index, tensor_names, **kwargs):
                del index, kwargs
                names = tuple(str(name) for name in tensor_names)
                load_calls.append(names)
                self.assertEqual(names, ("model.language_model.layers.0.input_layernorm.weight",))
                return {
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                }

            with patch.object(
                DirectStageBundleBuilder,
                "_capture_prefill_stage_boundaries",
                side_effect=AssertionError("runtime-only generate 不应触发 prefill boundary capture"),
            ), patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.prepare_text_prefill_runtime_inputs_from_weights",
                side_effect=AssertionError("runtime-only generate 不应构造 prefill runtime inputs"),
            ), patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.load_tensors_from_index",
                side_effect=_fake_load_tensors_from_index,
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                    include_text_weights=False,
                ) as builder:
                    bundle = builder.build_stage_bundle(0)

            self.assertEqual(load_calls, [("model.language_model.layers.0.input_layernorm.weight",)])
            self.assertTrue(bundle["runtime_only_generate"])
            self.assertEqual(bundle["save_dtype"], "bfloat16")
            self.assertNotIn("prefill_attention_mask_2d", bundle)
            self.assertNotIn("prefill_seq_len", bundle)
            self.assertNotIn("batch_size", bundle)
            self.assertNotIn("token_id_dtype", bundle)
            self.assertNotIn("input_ids", bundle)
            self.assertNotIn("prefill_input_ids", bundle)
            self.assertNotIn("prompt", bundle)
            self.assertNotIn("embed_tokens_weight", bundle)
            self.assertEqual(bundle["layers"], [])

    def test_runtime_only_non_first_stage_scaffold_omits_token_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
                    "model.language_model.norm.weight": torch.ones(8, dtype=torch.float32),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
                "include_runtime_reference": False,
                "_runtime_only_input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "_runtime_only_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with DirectStageBundleBuilder(
                stage_specs=[stage_spec],
                runtime_config=runtime_config,
                include_text_weights=False,
            ) as builder:
                bundle = builder.build_stage_bundle(1)

            self.assertTrue(bundle["runtime_only_generate"])
            self.assertNotIn("input_ids", bundle)
            self.assertNotIn("prefill_attention_mask_2d", bundle)
            self.assertNotIn("prefill_seq_len", bundle)
            self.assertNotIn("batch_size", bundle)
            self.assertNotIn("token_id_dtype", bundle)
            self.assertNotIn("prefill_input_ids", bundle)
            self.assertNotIn("prompt", bundle)

    def test_runtime_only_text_generate_session_uses_preseeded_prompt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float32),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
                "include_runtime_reference": False,
                "_runtime_only_input_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "_runtime_only_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.prepare_text_prompt_meta",
                side_effect=AssertionError("预广播 metadata 后不应再重建 prompt metadata"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                    include_text_weights=False,
                ) as builder:
                    bundle = builder.build_stage_bundle(0)

            self.assertTrue(bundle["runtime_only_generate"])
            self.assertNotIn("input_ids", bundle)
            self.assertNotIn("prefill_attention_mask_2d", bundle)
            self.assertNotIn("prefill_seq_len", bundle)
            self.assertNotIn("batch_size", bundle)
            self.assertNotIn("token_id_dtype", bundle)

    def test_runtime_only_text_scaffold_transport_roundtrip_restores_tensors(self) -> None:
        scaffold = {
            "save_dtype": "float32",
            "runtime_only_generate": True,
            "start_idx": 0,
            "end_idx": 0,
            "max_new_tokens": 0,
            "layers": [],
            "stage_input": torch.ones(1, 3, 4, dtype=torch.float32),
            "prefill_attention_mask_2d": torch.ones(1, 3, dtype=torch.int64),
        }

        meta, tensor_payload = pack_text_scaffold_transport(scaffold)
        restored = restore_text_scaffold_transport(meta, tensor_payload)

        self.assertTrue(restored["runtime_only_generate"])
        self.assertTrue(torch.equal(restored["stage_input"], scaffold["stage_input"]))
        self.assertTrue(
            torch.equal(
                restored["prefill_attention_mask_2d"],
                scaffold["prefill_attention_mask_2d"],
            )
        )

    def test_materialize_runtime_only_text_scaffold_restores_tp_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )

            runtime_config = {
                "modality": "text",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "prompt": "hello",
                "max_new_tokens": 3,
                "include_runtime_reference": False,
                "_runtime_only_input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
                "_runtime_only_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with DirectStageBundleBuilder(
                stage_specs=[stage_spec],
                runtime_config=runtime_config,
                include_text_weights=False,
            ) as builder:
                scaffold = builder.build_stage_bundle(0)

            self.assertNotIn("input_ids", scaffold)
            self.assertNotIn("prefill_attention_mask_2d", scaffold)
            self.assertNotIn("prefill_seq_len", scaffold)
            self.assertNotIn("batch_size", scaffold)
            self.assertNotIn("token_id_dtype", scaffold)

            materialized = materialize_text_stage(
                stage_bundle_scaffold=scaffold,
                runtime_config=runtime_config,
                compute_dtype=torch.float32,
                tp_shard_rank=1,
                tp_shard_world_size=2,
            )

            self.assertTrue(materialized["runtime_only_generate"])
            self.assertTrue(materialized["tp_weight_sharded"])
            self.assertEqual(tuple(materialized["layers"][0]["q_weight"].shape), (4, 8))
            self.assertEqual(tuple(materialized["layers"][0]["k_weight"].shape), (2, 8))
            self.assertEqual(tuple(materialized["layers"][0]["o_weight"].shape), (8, 4))
            self.assertIn("embed_tokens_weight", materialized)
            self.assertIn("final_norm_weight", materialized)
            self.assertIn("lm_head_weight", materialized)
            self.assertNotIn("prefill", materialized)
            self.assertNotIn("decode_steps", materialized)
            self.assertEqual(tuple(materialized["input_ids"].shape), (1, 3))
            self.assertEqual(tuple(materialized["prefill_attention_mask_2d"].shape), (1, 3))
            self.assertEqual(materialized["prefill_seq_len"], 3)
            self.assertEqual(materialized["batch_size"], 1)
            self.assertEqual(materialized["token_id_dtype"], "int64")

    def test_multimodal_generate_scaffold_omits_weights_and_materializes_local_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                vocab_size=4,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(8),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.arange(128, dtype=torch.float32).view(16, 8),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.arange(128, dtype=torch.float32).view(8, 16),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(8),
                    "model.language_model.norm.weight": torch.ones(8),
                },
                shard,
            )
            _config, _embed_tokens_weight, frontend_state = self._build_small_mm_frontend_state(model_dir)

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
                "_mm_frontend_state": frontend_state,
                "_mm_num_frames": 2,
                "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with DirectStageBundleBuilder(
                stage_specs=[stage_spec],
                runtime_config=runtime_config,
                include_text_weights=False,
            ) as builder:
                scaffold = builder.build_stage_bundle(0)

            with DirectStageBundleBuilder(
                stage_specs=[stage_spec],
                runtime_config=runtime_config,
                tp_shard_rank=1,
                tp_shard_world_size=2,
            ) as builder:
                reference_bundle = builder.build_stage_bundle(0)

            self.assertEqual(scaffold["layers"], [])
            self.assertTrue(scaffold["runtime_inputs_local_rebuild"])
            self.assertEqual(scaffold["runtime_prefill_cache_policy"], "recompute")
            self.assertNotIn("embed_tokens_weight", scaffold)
            self.assertNotIn("final_norm_weight", scaffold)
            self.assertNotIn("lm_head_weight", scaffold)
            self.assertEqual(scaffold["num_frames"], 2)
            self.assertEqual(scaffold["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
            self.assertNotIn("attention_mask", scaffold["prefill"])
            self.assertNotIn("cos", scaffold["prefill"])
            self.assertNotIn("sin", scaffold["prefill"])
            self.assertNotIn("position_ids", scaffold["decode_steps"][0])
            self.assertNotIn("attention_mask", scaffold["decode_steps"][0])
            self.assertNotIn("cos", scaffold["decode_steps"][0])
            self.assertNotIn("sin", scaffold["decode_steps"][0])

            materialized = materialize_text_stage(
                stage_bundle_scaffold=scaffold,
                runtime_config=runtime_config,
                compute_dtype=torch.float32,
                tp_shard_rank=1,
                tp_shard_world_size=2,
            )

            self.assertTrue(materialized["tp_weight_sharded"])
            self.assertEqual(materialized["tp_shard_rank"], 1)
            self.assertEqual(materialized["tp_shard_world_size"], 2)
            self.assertEqual(tuple(materialized["layers"][0]["q_weight"].shape), (4, 8))
            self.assertEqual(tuple(materialized["layers"][0]["k_weight"].shape), (2, 8))
            self.assertEqual(tuple(materialized["layers"][0]["o_weight"].shape), (8, 4))
            self.assertEqual(tuple(materialized["layers"][0]["gate_weight"].shape), (8, 8))
            self.assertEqual(tuple(materialized["layers"][0]["down_weight"].shape), (8, 8))
            self.assertIn("embed_tokens_weight", materialized)
            self.assertNotIn("final_norm_weight", materialized)
            self.assertNotIn("lm_head_weight", materialized)
            self.assertIn("attention_mask", materialized["prefill"])
            self.assertIn("cos", materialized["prefill"])
            self.assertIn("sin", materialized["prefill"])
            self.assertIn("position_ids", materialized["decode_steps"][0])
            self.assertIn("attention_mask", materialized["decode_steps"][0])
            self.assertIn("cos", materialized["decode_steps"][0])
            self.assertIn("sin", materialized["decode_steps"][0])
            self.assertEqual(materialized["num_frames"], 2)
            self.assertEqual(materialized["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
            self.assertTrue(
                torch.equal(
                    materialized["prefill"]["attention_mask"],
                    reference_bundle["prefill"]["attention_mask"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    materialized["prefill"]["cos"],
                    reference_bundle["prefill"]["cos"],
                )
            )
            self.assertTrue(
                torch.equal(
                    materialized["decode_steps"][0]["position_ids"],
                    reference_bundle["decode_steps"][0]["position_ids"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    materialized["decode_steps"][0]["sin"],
                    reference_bundle["decode_steps"][0]["sin"],
                )
            )

    def test_direct_stage_bundle_builder_multimodal_non_stage0_generate_rejects_seeded_frontend_without_startup_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            seeded_frontend_state = MmRuntimeState(
                input_ids=prefill_inputs.input_ids,
                attention_mask_2d=prefill_inputs.attention_mask_2d,
                position_ids=prefill_inputs.position_ids,
                inputs_embeds=prefill_inputs.inputs_embeds,
                attention_mask=prefill_inputs.attention_mask,
                cos=prefill_inputs.cos,
                sin=prefill_inputs.sin,
                visual=MmVisualState(
                    visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                    deepstack_by_layer={},
                ),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            )

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
                "_mm_frontend_state": seeded_frontend_state,
                "_mm_num_frames": 2,
                "_mm_frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
            }
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("non-stage0 不应消费 legacy frontend state"),
            ) as resolve_frontend_mock:
                with self.assertRaisesRegex(RuntimeError, "startup contract"):
                    DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                    )

            resolve_frontend_mock.assert_not_called()

    def test_direct_stage_bundle_builder_multimodal_non_stage0_decode_prefers_stage_handoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            frontend_state = MmRuntimeState(
                input_ids=prefill_inputs.input_ids,
                attention_mask_2d=prefill_inputs.attention_mask_2d,
                position_ids=prefill_inputs.position_ids,
                inputs_embeds=prefill_inputs.inputs_embeds,
                attention_mask=prefill_inputs.attention_mask,
                cos=prefill_inputs.cos,
                sin=prefill_inputs.sin,
                visual=MmVisualState(
                    visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                    deepstack_by_layer={},
                ),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            )

            runtime_config = {
                "modality": "multimodal",
                "mode": "decode",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "decode_token_id": 4,
            }
            self._seed_small_mm_startup_contract(runtime_config, frontend_state)
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    decode_runtime_state = build_mm_decode_state_from_weights(
                        decode_input_ids=torch.tensor([[4]], dtype=torch.long),
                        attention_mask_2d=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                        past_length=3,
                        rope_deltas=frontend_state.rope_deltas,
                        embed_tokens_weight=embed_tokens_weight,
                        config_spec=config,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                    )
                    stage_input = torch.full((1, 1, 4), 9.0, dtype=torch.float32)
                    hidden_stage_output = torch.full((1, 1, 4), 11.0, dtype=torch.float32)
                    norm_output = torch.full((1, 1, 4), 13.0, dtype=torch.float32)
                    logits = torch.full((1, 1, 6), 17.0, dtype=torch.float32)
                    builder._decode_state = {
                        "decode_source": "provided",
                        "decode_token_id": 4,
                        "decode_input_ids": decode_runtime_state.input_ids,
                        "attention_mask_2d": decode_runtime_state.attention_mask_2d,
                        "attention_mask": decode_runtime_state.attention_mask,
                        "cos": decode_runtime_state.cos,
                        "sin": decode_runtime_state.sin,
                        "position_ids": decode_runtime_state.position_ids,
                        "mm_runtime_state": decode_runtime_state,
                        "stage_handoffs": {
                            1: {
                                "stage_input": stage_input,
                                "stage_output": hidden_stage_output,
                            }
                        },
                        "hidden_stage_output": hidden_stage_output,
                        "norm_output": norm_output,
                        "logits": logits,
                        "cache_by_layer": {},
                    }
                    bundle = builder.build_stage_bundle(1)

            self.assertTrue(torch.equal(bundle["stage_input"], stage_input))
            self.assertTrue(torch.equal(bundle["hidden_stage_output"], hidden_stage_output))
            self.assertTrue(torch.equal(bundle["norm_output"], norm_output))
            self.assertTrue(torch.equal(bundle["logits"], logits))

    def test_direct_stage_bundle_builder_multimodal_non_stage0_generate_prefers_step_handoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            frontend_state = MmRuntimeState(
                input_ids=prefill_inputs.input_ids,
                attention_mask_2d=prefill_inputs.attention_mask_2d,
                position_ids=prefill_inputs.position_ids,
                inputs_embeds=prefill_inputs.inputs_embeds,
                attention_mask=prefill_inputs.attention_mask,
                cos=prefill_inputs.cos,
                sin=prefill_inputs.sin,
                visual=MmVisualState(
                    visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                    deepstack_by_layer={},
                ),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            )

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
            }
            self._seed_small_mm_startup_contract(runtime_config, frontend_state)
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    decode_runtime_state = build_mm_decode_state_from_weights(
                        decode_input_ids=torch.tensor([[4]], dtype=torch.long),
                        attention_mask_2d=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                        past_length=3,
                        rope_deltas=frontend_state.rope_deltas,
                        embed_tokens_weight=embed_tokens_weight,
                        config_spec=config,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                    )
                    step_stage_input = torch.full((1, 1, 4), 19.0, dtype=torch.float32)
                    step_hidden_stage_output = torch.full((1, 1, 4), 23.0, dtype=torch.float32)
                    step_norm_output = torch.full((1, 1, 4), 29.0, dtype=torch.float32)
                    step_logits = torch.full((1, 1, 6), 31.0, dtype=torch.float32)
                    builder._generate_state = {
                        "max_new_tokens": 2,
                        "generated_token_ids": [4, 5],
                        "prefill_norm_output": torch.zeros((1, 3, 4), dtype=torch.float32),
                        "prefill_logits": torch.zeros((1, 3, 6), dtype=torch.float32),
                        "cache_by_layer": {},
                        "step_results": [
                            {
                                "step_idx": 0,
                                "decode_input_ids": decode_runtime_state.input_ids,
                                "attention_mask_2d": decode_runtime_state.attention_mask_2d,
                                "attention_mask": decode_runtime_state.attention_mask,
                                "cos": decode_runtime_state.cos,
                                "sin": decode_runtime_state.sin,
                                "position_ids": decode_runtime_state.position_ids,
                                "mm_runtime_state": decode_runtime_state,
                                "stage_handoffs": {
                                    1: {
                                        "stage_input": step_stage_input,
                                        "stage_output": step_hidden_stage_output,
                                    }
                                },
                                "hidden_stage_output": step_hidden_stage_output,
                                "norm_output": step_norm_output,
                                "logits": step_logits,
                                "output_token_id": 5,
                            }
                        ],
                    }
                    bundle = builder.build_stage_bundle(1)

            self.assertEqual(len(bundle["decode_steps"]), 1)
            self.assertTrue(torch.equal(bundle["decode_steps"][0]["stage_input"], step_stage_input))
            self.assertTrue(torch.equal(bundle["decode_steps"][0]["hidden_stage_output"], step_hidden_stage_output))
            self.assertTrue(torch.equal(bundle["decode_steps"][0]["norm_output"], step_norm_output))
            self.assertTrue(torch.equal(bundle["decode_steps"][0]["logits"], step_logits))

    def test_direct_stage_bundle_builder_multimodal_runtime_only_generate_uses_local_startup_handoff(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            self._write_two_layer_text_checkpoint(model_dir)
            _config, _embed_tokens_weight, frontend_state = self._build_small_mm_frontend_state(model_dir)

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
                "include_runtime_reference": False,
            }
            expected_stage_input, _expected_stage_output = self._seed_small_mm_startup_contract(
                runtime_config,
                frontend_state,
            )
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch.object(
                DirectStageBundleBuilder,
                "_ensure_generate_state",
                side_effect=AssertionError("runtime-only multimodal generate 不应构建 full generate reference"),
            ), patch.object(
                DirectStageBundleBuilder,
                "_ensure_prefill_full_state",
                side_effect=AssertionError("runtime-only multimodal generate 不应回退到 full prefill reference"),
            ), patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    bundle = builder.build_stage_bundle(1)

            self.assertTrue(bundle["runtime_only_generate"])
            self.assertEqual(bundle["modality"], "multimodal")
            self.assertEqual(bundle["max_new_tokens"], 2)
            self.assertTrue(torch.equal(bundle["stage_input"], expected_stage_input))
            self.assertIn("prefill_attention_mask", bundle)
            self.assertIn("prefill_cos", bundle)
            self.assertIn("prefill_sin", bundle)
            self.assertIn("rope_deltas", bundle)
            self.assertNotIn("prefill", bundle)
            self.assertNotIn("decode_steps", bundle)
            self.assertNotIn("generated_token_ids", bundle)

    def test_mm_state_from_decode_state_accepts_stage_handoffs_without_hidden_states(self) -> None:
        stage_input = torch.full((1, 1, 4), 5.0, dtype=torch.float32)
        state = {
            "decode_input_ids": torch.tensor([[4]], dtype=torch.long),
            "attention_mask_2d": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.zeros(4, 1, 1, dtype=torch.long),
            "attention_mask": torch.zeros(1, 1, 1, 4, dtype=torch.float32),
            "cos": torch.zeros(1, 1, 4, dtype=torch.float32),
            "sin": torch.zeros(1, 1, 4, dtype=torch.float32),
            "stage_handoffs": {
                1: {
                    "stage_input": stage_input,
                    "stage_output": torch.full((1, 1, 4), 7.0, dtype=torch.float32),
                }
            },
        }

        runtime_state = mm_state_from_decode_state(state)

        self.assertTrue(torch.equal(runtime_state.inputs_embeds, stage_input))

    def test_direct_stage_bundle_builder_multimodal_file_backed_prefill_is_handoff_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            self._write_two_layer_text_checkpoint(model_dir)
            _config, _embed_tokens_weight, frontend_state = self._build_small_mm_frontend_state(model_dir)

            runtime_config = {
                "modality": "multimodal",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
            }
            expected_stage_input, _expected_stage_output = self._seed_small_mm_startup_contract(
                runtime_config,
                frontend_state,
            )
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    self.assertFalse(hasattr(builder, "model"))
                    self.assertIsNone(builder._prefill_full_state)
                    self.assertIn(1, builder._prefill_stage_inputs_by_stage)
                    self.assertTrue(
                        torch.equal(
                            builder._prefill_stage_inputs_by_stage[1],
                            expected_stage_input,
                        )
                    )
                    bundle = builder.build_stage_bundle(1)

            self.assertTrue(torch.equal(bundle["stage_input"], expected_stage_input))

    def test_direct_stage_bundle_builder_multimodal_file_backed_decode_is_handoff_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            self._write_two_layer_text_checkpoint(model_dir)
            config, embed_tokens_weight, frontend_state = self._build_small_mm_frontend_state(model_dir)

            runtime_config = {
                "modality": "multimodal",
                "mode": "decode",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "decode_token_id": 4,
            }
            self._seed_small_mm_startup_contract(runtime_config, frontend_state)
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    decode_runtime_state = build_mm_decode_state_from_weights(
                        decode_input_ids=torch.tensor([[4]], dtype=torch.long),
                        attention_mask_2d=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                        past_length=3,
                        rope_deltas=frontend_state.rope_deltas,
                        embed_tokens_weight=embed_tokens_weight,
                        config_spec=config,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                    )
                    stage_input = torch.full((1, 1, 4), 9.0, dtype=torch.float32)
                    hidden_stage_output = torch.full((1, 1, 4), 11.0, dtype=torch.float32)
                    state = {
                        "decode_source": "provided",
                        "decode_token_id": 4,
                        "decode_input_ids": decode_runtime_state.input_ids,
                        "attention_mask_2d": decode_runtime_state.attention_mask_2d,
                        "attention_mask": decode_runtime_state.attention_mask,
                        "cos": decode_runtime_state.cos,
                        "sin": decode_runtime_state.sin,
                        "position_ids": decode_runtime_state.position_ids,
                        "mm_runtime_state": decode_runtime_state,
                        "stage_handoffs": {
                            1: {
                                "stage_input": stage_input,
                                "stage_output": hidden_stage_output,
                            }
                        },
                        "hidden_stage_output": hidden_stage_output,
                        "norm_output": torch.full((1, 1, 4), 13.0, dtype=torch.float32),
                        "logits": torch.full((1, 1, 6), 17.0, dtype=torch.float32),
                        "cache_by_layer": {},
                    }
                    builder._decode_state = state
                    bundle = builder.build_stage_bundle(1)

            self.assertTrue(
                torch.equal(
                    bundle["stage_input"],
                    state["stage_handoffs"][1]["stage_input"],
                )
            )

    def test_direct_stage_bundle_builder_multimodal_file_backed_generate_is_handoff_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            self._write_two_layer_text_checkpoint(model_dir)
            config, embed_tokens_weight, frontend_state = self._build_small_mm_frontend_state(model_dir)

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
            }
            self._seed_small_mm_startup_contract(runtime_config, frontend_state)
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    decode_runtime_state = build_mm_decode_state_from_weights(
                        decode_input_ids=torch.tensor([[4]], dtype=torch.long),
                        attention_mask_2d=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                        past_length=3,
                        rope_deltas=frontend_state.rope_deltas,
                        embed_tokens_weight=embed_tokens_weight,
                        config_spec=config,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                    )
                    step_stage_input = torch.full((1, 1, 4), 19.0, dtype=torch.float32)
                    step_hidden_stage_output = torch.full((1, 1, 4), 23.0, dtype=torch.float32)
                    state = {
                        "max_new_tokens": 2,
                        "generated_token_ids": [4, 5],
                        "prefill_norm_output": torch.zeros((1, 3, 4), dtype=torch.float32),
                        "prefill_logits": torch.zeros((1, 3, 6), dtype=torch.float32),
                        "cache_by_layer": {},
                        "step_results": [
                            {
                                "step_idx": 0,
                                "decode_input_ids": decode_runtime_state.input_ids,
                                "attention_mask_2d": decode_runtime_state.attention_mask_2d,
                                "attention_mask": decode_runtime_state.attention_mask,
                                "cos": decode_runtime_state.cos,
                                "sin": decode_runtime_state.sin,
                                "position_ids": decode_runtime_state.position_ids,
                                "mm_runtime_state": decode_runtime_state,
                                "stage_handoffs": {
                                    1: {
                                        "stage_input": step_stage_input,
                                        "stage_output": step_hidden_stage_output,
                                    }
                                },
                                "hidden_stage_output": step_hidden_stage_output,
                                "norm_output": torch.full((1, 1, 4), 29.0, dtype=torch.float32),
                                "logits": torch.full((1, 1, 6), 31.0, dtype=torch.float32),
                                "output_token_id": 5,
                            }
                        ],
                    }
                    builder._generate_state = state
                    bundle = builder.build_stage_bundle(1)

            self.assertTrue(
                torch.equal(
                    bundle["decode_steps"][0]["stage_input"],
                    state["step_results"][0]["stage_handoffs"][1]["stage_input"],
                )
            )

    def test_direct_stage_bundle_builder_multimodal_stage0_generate_activates_frontend_once_then_uses_file_backed_decoder(self) -> None:
        import qwen3vl_tp_runtime.models.qwen3vl.runtime_builder as runtime_builder

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            frontend_state = MmRuntimeState(
                    input_ids=prefill_inputs.input_ids,
                    attention_mask_2d=prefill_inputs.attention_mask_2d,
                    position_ids=prefill_inputs.position_ids,
                    inputs_embeds=prefill_inputs.inputs_embeds,
                    attention_mask=prefill_inputs.attention_mask,
                    cos=prefill_inputs.cos,
                    sin=prefill_inputs.sin,
                    visual=MmVisualState(
                        visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                        deepstack_by_layer={},
                    ),
                    rope_deltas=torch.zeros(1, 1, dtype=torch.long),
                    mm_token_type_ids=torch.tensor([[0, 1, 0]], dtype=torch.int),
                    image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
                )
            frontend_seed = {
                "frontend_tensors": compact_mm_frontend_tensors(frontend_state),
                "frontend_meta": {
                    "runtime": compact_mm_frontend_meta(frontend_state),
                    "num_frames": 2,
                    "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                },
            }

            runtime_config = {
                "modality": "multimodal",
                "mode": "generate",
                "model_path": str(model_dir),
                "save_dtype": "float32",
                "max_new_tokens": 2,
            }
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_mm.prepare_mm_frontend_seed",
                return_value=frontend_seed,
            ) as prepare_seed_mock:
                with patch.object(
                    DirectStageBundleBuilder,
                    "_run_text_file_backed_prefill",
                    side_effect=AssertionError("single-stage multimodal generate 不应再回退到 full file-backed prefill"),
                ):
                    with DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                    ) as builder:
                        self.assertTrue(builder.weight_backed_multimodal)
                        self.assertTrue(builder.mm_activate_frontend)
                        self.assertEqual(builder.extra["frontend_activation"], "active")
                        self.assertFalse(hasattr(builder, "model"))
                        self.assertIsNone(builder._mm_prefill_state)
                        self.assertIsNone(builder.prefill_runtime_inputs)
                        self.assertIsInstance(builder._mm_prefill_shared, dict)
                        bundle = builder.build_stage_bundle(0)
                        self.assertIn("stage_handoffs", builder._prefill_full_state)
                        self.assertNotIn("hidden_states", builder._prefill_full_state)
                        self.assertIn("mm_runtime_shared", builder._prefill_full_state)
                        self.assertIn("mm_runtime_shared", builder._generate_state)

            prepare_seed_mock.assert_called_once()
            self.assertEqual(
                prepare_seed_mock.call_args.args[0],
                {
                    "modality": "multimodal",
                    "mode": "generate",
                    "model_path": str(model_dir),
                    "save_dtype": "float32",
                    "max_new_tokens": 2,
                    "_mm_weight_index": runtime_config["_mm_weight_index"],
                },
            )
            self.assertFalse(hasattr(runtime_builder, "load_model"))
            self.assertFalse(hasattr(runtime_builder, "prepare_mm_session"))
            self.assertTrue(runtime_config["_mm_frontend_state_ready"])
            self.assertIn("_mm_frontend_seed", runtime_config)
            self.assertNotIn("input_ids", runtime_config["_mm_frontend_seed"])
            self.assertNotIn("attention_mask_2d", runtime_config["_mm_frontend_seed"])
            self.assertNotIn("position_ids", runtime_config["_mm_frontend_seed"])
            self.assertNotIn("rope_deltas", runtime_config["_mm_frontend_seed"])
            self.assertIn("_mm_frontend_meta", runtime_config)
            self.assertNotIn("_mm_frontend_plan", runtime_config)
            self.assertNotIn("_mm_frontend_state", runtime_config)
            self.assertEqual(bundle["module_name"], "multimodal_generate_stage")
            self.assertEqual(bundle["stage_type"], "multimodal_generate_last")
            self.assertEqual(bundle["num_frames"], 2)
            self.assertEqual(bundle["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
            self.assertEqual(tuple(bundle["prefill"]["stage_input"].shape), (1, 3, 4))
            self.assertEqual(tuple(bundle["prefill"]["stage_output"].shape), (1, 3, 6))
            self.assertEqual(len(bundle["decode_steps"]), 1)
            self.assertIn("embed_tokens_weight", bundle)
            self.assertIn("final_norm_weight", bundle)
            self.assertIn("lm_head_weight", bundle)
            self.assertEqual(tuple(bundle["embed_tokens_weight"].shape), (6, 4))
            self.assertEqual(tuple(bundle["final_norm_weight"].shape), (4,))
            self.assertEqual(tuple(bundle["lm_head_weight"].shape), (6, 4))

    def test_direct_stage_bundle_builder_multimodal_stage0_prefill_uses_startup_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.0.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            frontend_state = MmRuntimeState(
                input_ids=prefill_inputs.input_ids,
                attention_mask_2d=prefill_inputs.attention_mask_2d,
                position_ids=prefill_inputs.position_ids,
                inputs_embeds=prefill_inputs.inputs_embeds,
                attention_mask=prefill_inputs.attention_mask,
                cos=prefill_inputs.cos,
                sin=prefill_inputs.sin,
                visual=MmVisualState(
                    visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                    deepstack_by_layer={},
                ),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            )
            runtime_config = {
                "modality": "multimodal",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
            }
            seed_mm_startup_runtime_config(
                runtime_config,
                {
                    "shared": compact_mm_runtime_shared(frontend_state),
                    "stage_handoffs": {
                        0: {
                            "stage_input": frontend_state.inputs_embeds.detach().clone(),
                            "stage_output": torch.full_like(frontend_state.inputs_embeds, 7.0),
                        },
                    },
                    "stage_visuals": {
                        0: {
                            "visual_pos_masks": None,
                            "deepstack_by_layer": {},
                        },
                    },
                    "num_frames": 2,
                    "frame_paths": ["/tmp/f0.png", "/tmp/f1.png"],
                },
            )
            stage_spec = StageSpec(
                stage_idx=0,
                start_idx=0,
                end_idx=0,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("startup contract 就绪后不应再回退到 resolve_mm_frontend"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    self.assertTrue(builder.mm_activate_frontend)
                    self.assertEqual(builder.extra["frontend_activation"], "startup-contract")
                    self.assertFalse(hasattr(builder, "model"))
                    self.assertIn(0, builder._prefill_stage_inputs_by_stage)
                    bundle = builder.build_stage_bundle(0)

            self.assertEqual(bundle["module_name"], "multimodal_prefill_stage")
            self.assertEqual(bundle["num_frames"], 2)
            self.assertEqual(bundle["frame_paths"], ["/tmp/f0.png", "/tmp/f1.png"])
            self.assertTrue(
                torch.equal(
                    bundle["stage_input"],
                    frontend_state.inputs_embeds,
                )
            )
            self.assertTrue(
                torch.equal(
                    bundle["stage_output"],
                    torch.full_like(frontend_state.inputs_embeds, 7.0),
                )
            )
            self.assertNotIn("_mm_frontend_seed", runtime_config)
            self.assertTrue(runtime_config["_mm_startup_contract_ready"])
            self.assertTrue(runtime_config["_mm_frontend_state_ready"])

    def test_direct_stage_bundle_builder_multimodal_non_frontend_requires_startup_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            torch.save(
                {
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                },
                model_dir / "pytorch_model.bin",
            )

            runtime_config = {
                "modality": "multimodal",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
            }
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch(
                "qwen3vl_tp_runtime.models.qwen3vl.runtime_builder.resolve_mm_frontend",
                side_effect=AssertionError("non-stage0 不应本地解析/构建 multimodal frontend"),
            ) as resolve_frontend_mock:
                with self.assertRaisesRegex(RuntimeError, "startup contract"):
                    DirectStageBundleBuilder(
                        stage_specs=[stage_spec],
                        runtime_config=runtime_config,
                    )

            resolve_frontend_mock.assert_not_called()

    def test_direct_stage_bundle_builder_multimodal_last_stage_prefill_uses_local_handoff_without_root_input(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_text_config(
                model_dir,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                vocab_size=6,
                rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [1, 0, 0]},
            )
            shard = model_dir / "pytorch_model.bin"
            torch.save(
                {
                    "model.language_model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).view(6, 4),
                    "model.language_model.layers.1.input_layernorm.weight": torch.ones(4),
                    "model.language_model.layers.1.self_attn.q_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.k_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.v_proj.weight": torch.ones(2, 4),
                    "model.language_model.layers.1.self_attn.o_proj.weight": torch.ones(4, 4),
                    "model.language_model.layers.1.self_attn.q_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.self_attn.k_norm.weight": torch.ones(2),
                    "model.language_model.layers.1.mlp.gate_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.up_proj.weight": torch.ones(8, 4),
                    "model.language_model.layers.1.mlp.down_proj.weight": torch.ones(4, 8),
                    "model.language_model.layers.1.post_attention_layernorm.weight": torch.ones(4),
                    "model.language_model.norm.weight": torch.ones(4),
                },
                shard,
            )

            config = load_text_model_config_spec(str(model_dir))
            embed_tokens_weight = load_tensors_by_name(
                str(model_dir),
                ["model.language_model.embed_tokens.weight"],
            )["model.language_model.embed_tokens.weight"]
            prefill_inputs = prepare_text_prefill_runtime_inputs_from_weights(
                input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
                attention_mask_2d=torch.tensor([[1, 1, 1]], dtype=torch.long),
                embed_tokens_weight=embed_tokens_weight,
                config_spec=config,
                device=torch.device("cpu"),
                compute_dtype=torch.float32,
            )
            frontend_state = MmRuntimeState(
                input_ids=prefill_inputs.input_ids,
                attention_mask_2d=prefill_inputs.attention_mask_2d,
                position_ids=prefill_inputs.position_ids,
                inputs_embeds=prefill_inputs.inputs_embeds,
                attention_mask=prefill_inputs.attention_mask,
                cos=prefill_inputs.cos,
                sin=prefill_inputs.sin,
                visual=MmVisualState(
                    visual_pos_masks=torch.zeros(1, 3, dtype=torch.bool),
                    deepstack_by_layer={},
                ),
                rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            )
            stage_input = torch.full_like(frontend_state.inputs_embeds, 3.0)
            stage_output = torch.full_like(frontend_state.inputs_embeds, 7.0)
            runtime_config = {
                "modality": "multimodal",
                "mode": "prefill",
                "model_path": str(model_dir),
                "save_dtype": "float32",
            }
            seed_mm_startup_runtime_config(
                runtime_config,
                {
                    "shared": compact_mm_runtime_shared(frontend_state),
                    "stage_handoffs": {
                        1: {
                            "stage_input": stage_input.detach().clone(),
                            "stage_output": stage_output.detach().clone(),
                        },
                    },
                    "stage_visuals": {
                        1: {
                            "visual_pos_masks": None,
                            "deepstack_by_layer": {},
                        },
                    },
                    "num_frames": 1,
                    "frame_paths": ["/tmp/f0.png"],
                },
            )
            stage_spec = StageSpec(
                stage_idx=1,
                start_idx=1,
                end_idx=1,
                num_layers=1,
                save_dtype="float32",
                bundle_path=None,
            )

            with patch.object(
                DirectStageBundleBuilder,
                "_ensure_mm_full_prefill_runtime",
                side_effect=AssertionError("last-stage prefill 不应再依赖 root_input/full multimodal runtime"),
            ):
                with DirectStageBundleBuilder(
                    stage_specs=[stage_spec],
                    runtime_config=runtime_config,
                ) as builder:
                    self.assertFalse(hasattr(builder, "model"))
                    self.assertIsNone(builder._mm_prefill_root_input)
                    bundle = builder.build_stage_bundle(1)

            self.assertEqual(bundle["module_name"], "multimodal_prefill_stage")
            self.assertEqual(bundle["stage_type"], "text_prefill_last")
            self.assertEqual(bundle["num_frames"], 1)
            self.assertEqual(bundle["frame_paths"], ["/tmp/f0.png"])
            self.assertTrue(torch.equal(bundle["stage_input"], stage_input))
            self.assertTrue(torch.equal(bundle["hidden_stage_output"], stage_output))
            self.assertEqual(tuple(bundle["norm_output"].shape), (1, 3, 4))
            self.assertEqual(tuple(bundle["logits"].shape), (1, 3, 6))
            self.assertIn("final_norm_weight", bundle)
            self.assertIn("lm_head_weight", bundle)
            weight_load = summarize_text_weight_load(bundle)
            self.assertEqual(weight_load["stage_start_idx"], 1)
            self.assertEqual(weight_load["stage_end_idx"], 1)
            self.assertEqual(weight_load["loaded_layer_indices"], [1])
            self.assertTrue(weight_load["stage_weight_scope_ok"])
            self.assertEqual(weight_load["unexpected_layer_indices"], [])
            self.assertIn("final_norm_weight", weight_load["loaded_top_level_weight_names"])
            self.assertIn("lm_head_weight", weight_load["loaded_top_level_weight_names"])
            self.assertNotIn("embed_tokens_weight", weight_load["loaded_top_level_weight_names"])
            self.assertTrue(runtime_config["_mm_startup_contract_ready"])
            self.assertTrue(runtime_config["_mm_frontend_state_ready"])


if __name__ == "__main__":
    unittest.main()
