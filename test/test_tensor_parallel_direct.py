from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel import (
    TensorParallelManifest,
    TensorParallelRunner,
)
from qwen3vl_tp_runtime.hexgen_core.schema import StageSpec


def _build_tp_manifest(
    *,
    stage_ranges: list[tuple[int, int]],
    tp_degrees: list[int],
) -> TensorParallelManifest:
    stages = [
        StageSpec(
            stage_idx=stage_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            num_layers=end_idx - start_idx + 1,
            save_dtype="float32",
            bundle_path=None,
        )
        for stage_idx, (start_idx, end_idx) in enumerate(stage_ranges)
    ]
    return TensorParallelManifest(
        runtime="text_generate_tp",
        tp_degrees=tp_degrees,
        stage_ranges=stage_ranges,
        bundle_dir=None,
        stages=stages,
        boundaries=[],
        num_frames=0,
        save_dtype="float32",
        pipeline_type="text_generate",
        runtime_config={
            "modality": "text",
            "mode": "generate",
            "model_path": "/tmp/fake-model",
            "save_dtype": "float32",
        },
    )


class TensorParallelDirectRunnerTest(unittest.TestCase):
    def test_direct_tp_runner_uses_tensor_parallel_loader(self) -> None:
        manifest = _build_tp_manifest(stage_ranges=[(0, 35)], tp_degrees=[2])
        runner = TensorParallelRunner(
            manifest=manifest,
            device=torch.device("cpu"),
            compute_dtype_arg="float32",
            comm_dtype_arg="float32",
            return_tensors=True,
        )

        with patch(
            "qwen3vl_tp_runtime.hexgen_core.modules.tensor_parallel.load_stage_state_for_tp_rank",
            return_value=({"start_idx": 0, "end_idx": 35, "layers": []}, torch.float32),
        ) as load_mock, patch.object(
            TensorParallelRunner,
            "_run_generate_rank",
            return_value={"rank": 0, "backend": "tp"},
        ) as run_generate_mock:
            stats = runner.run_rank(rank=0, world_size=2)

        self.assertEqual(stats["backend"], "tp")
        load_mock.assert_called_once()
        self.assertIs(load_mock.call_args.args[0], manifest)
        run_generate_mock.assert_called_once()

    def test_direct_tp_runner_rejects_multi_stage_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "单 stage"):
            _build_tp_manifest(stage_ranges=[(0, 17), (18, 35)], tp_degrees=[2])


if __name__ == "__main__":
    unittest.main()
