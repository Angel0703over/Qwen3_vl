from __future__ import annotations

import unittest
from unittest import mock

import torch

from qwen3vl_tp_runtime.hexgen_core.distributed import (
    all_gather_cpu,
    all_reduce_cpu,
    broadcast_cpu,
    get_transport_profile_events,
    reset_transport_profile_events,
    set_transport_pin_memory_enabled,
)


class DistributedSingleRankBypassTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_transport_profile_events()
        set_transport_pin_memory_enabled(False)

    def tearDown(self) -> None:
        set_transport_pin_memory_enabled(False)

    def test_all_reduce_single_rank_returns_local_tensor_without_event(self) -> None:
        local = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=1),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.all_reduce") as all_reduce_mock,
        ):
            result = all_reduce_cpu(
                local,
                target_device=torch.device("cpu"),
                target_dtype=torch.float64,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "mlp"},
            )

        all_reduce_mock.assert_not_called()
        self.assertEqual(result.dtype, torch.float64)
        self.assertTrue(torch.equal(result, local.to(dtype=torch.float64)))
        self.assertEqual(get_transport_profile_events(), [])

    def test_all_gather_single_rank_returns_single_local_tensor_without_event(self) -> None:
        local = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=1),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.all_gather") as all_gather_mock,
        ):
            result = all_gather_cpu(
                local,
                target_device=torch.device("cpu"),
                target_dtype=torch.float64,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "mlp"},
            )

        all_gather_mock.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dtype, torch.float64)
        self.assertTrue(torch.equal(result[0], local.to(dtype=torch.float64)))
        self.assertEqual(get_transport_profile_events(), [])

    def test_broadcast_single_rank_returns_local_tensor_without_event(self) -> None:
        reference = torch.zeros((1, 2), dtype=torch.float32)
        local = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=1),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.broadcast") as broadcast_mock,
        ):
            result = broadcast_cpu(
                reference_tensor=reference,
                tensor=local,
                src=7,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "attention"},
            )

        broadcast_mock.assert_not_called()
        self.assertEqual(result.dtype, reference.dtype)
        self.assertTrue(torch.equal(result, local.to(dtype=reference.dtype)))
        self.assertEqual(get_transport_profile_events(), [])

    def test_broadcast_single_rank_requires_local_tensor(self) -> None:
        reference = torch.zeros((1, 2), dtype=torch.float32)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=1),
        ):
            with self.assertRaises(ValueError):
                broadcast_cpu(
                    reference_tensor=reference,
                    tensor=None,
                    src=7,
                    comm_dtype=torch.float32,
                    group=object(),
                )

    def test_all_reduce_records_substage_profile_for_multi_rank(self) -> None:
        local = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=2),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.all_reduce") as all_reduce_mock,
        ):
            result = all_reduce_cpu(
                local,
                target_device=torch.device("cpu"),
                target_dtype=torch.float64,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "mlp"},
            )

        all_reduce_mock.assert_called_once()
        self.assertTrue(torch.equal(result, local.to(dtype=torch.float64)))
        events = get_transport_profile_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["operation"], "all_reduce")
        self.assertEqual(event["world_size"], 2)
        self.assertEqual(event["source_dtype"], "torch.float32")
        self.assertEqual(event["comm_dtype"], "torch.float32")
        self.assertEqual(event["target_dtype"], "torch.float64")
        self.assertIn("device_to_cpu_seconds", event)
        self.assertIn("gloo_collective_seconds", event)
        self.assertIn("cpu_to_device_seconds", event)
        self.assertFalse(event["transport_pin_memory_requested"])

    def test_broadcast_records_substage_profile_for_multi_rank(self) -> None:
        reference = torch.zeros((1, 2), dtype=torch.float32)
        local = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=2),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.broadcast") as broadcast_mock,
        ):
            result = broadcast_cpu(
                reference_tensor=reference,
                tensor=local,
                src=0,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "runtime_input"},
            )

        broadcast_mock.assert_called_once()
        self.assertTrue(torch.equal(result, local.to(dtype=reference.dtype)))
        events = get_transport_profile_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["operation"], "broadcast")
        self.assertEqual(event["world_size"], 2)
        self.assertEqual(event["source_dtype"], "torch.float64")
        self.assertEqual(event["reference_dtype"], "torch.float32")
        self.assertEqual(event["comm_dtype"], "torch.float32")
        self.assertEqual(event["target_dtype"], "torch.float32")
        self.assertIn("payload_prepare_seconds", event)
        self.assertIn("device_to_cpu_seconds", event)
        self.assertIn("gloo_collective_seconds", event)
        self.assertIn("cpu_to_device_seconds", event)
        self.assertFalse(event["transport_pin_memory_requested"])

    def test_pin_memory_opt_in_is_profiled_without_changing_result(self) -> None:
        set_transport_pin_memory_enabled(True)
        local = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        with (
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_available", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.is_initialized", return_value=True),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.get_world_size", return_value=2),
            mock.patch("qwen3vl_tp_runtime.hexgen_core.distributed.dist.all_reduce") as all_reduce_mock,
            mock.patch(
                "qwen3vl_tp_runtime.hexgen_core.distributed.copy_tensor_to_cpu_transport",
                return_value=(local.clone(), False),
            ),
        ):
            result = all_reduce_cpu(
                local,
                target_device=torch.device("cpu"),
                target_dtype=torch.float32,
                comm_dtype=torch.float32,
                group=object(),
                profile_context={"phase": "prefill", "module": "mlp"},
            )

        all_reduce_mock.assert_called_once()
        self.assertTrue(torch.equal(result, local))
        event = get_transport_profile_events()[0]
        self.assertTrue(event["transport_pin_memory_requested"])
        self.assertIn("transport_pin_memory_used", event)


if __name__ == "__main__":
    unittest.main()
