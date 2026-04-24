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
    prepare_text_prompt_meta,
)
from qwen3vl_tp_runtime.models.qwen3vl.weights import (
    build_text_causal_mask,
    build_text_decoder_stage_weight_plan,
    load_text_decoder_stage_weight_bundle,
    load_text_model_config_spec,
    load_model_weight_index,
    load_tensors_by_name,
    prepare_text_decode_runtime_inputs_from_weights,
    prepare_text_prefill_runtime_inputs_from_weights,
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
                },
                shard,
            )

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
            self.assertTrue(stage_weights.tp_weight_sharded)
            self.assertEqual(stage_weights.tp_shard_rank, 1)
            self.assertEqual(stage_weights.tp_shard_world_size, 2)
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


if __name__ == "__main__":
    unittest.main()
