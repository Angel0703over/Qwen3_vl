import io
import unittest

import torch

from qwen3vl_tp_runtime.hexgen_core.distributed import (
    _deserialize_object_from_uint8,
    _serialize_object_to_uint8,
)


class DistributedSerializationTest(unittest.TestCase):
    def test_plain_python_payload_uses_compact_wire_format(self) -> None:
        payload = {
            "input_ids_list": [151644, 872, 198, 104455],
            "runtime_only_prompt_local_rebuild": True,
            "max_new_tokens": 4,
        }

        serialized = _serialize_object_to_uint8(payload)
        restored = _deserialize_object_from_uint8(serialized)

        legacy_buffer = io.BytesIO()
        torch.save(payload, legacy_buffer)

        self.assertEqual(restored, payload)
        self.assertLess(serialized.numel(), len(legacy_buffer.getbuffer()))

    def test_tensor_payload_round_trips_with_torch_wire_format(self) -> None:
        payload = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "max_new_tokens": 4,
        }

        serialized = _serialize_object_to_uint8(payload)
        restored = _deserialize_object_from_uint8(serialized)

        self.assertEqual(restored["max_new_tokens"], 4)
        self.assertTrue(torch.equal(restored["input_ids"], payload["input_ids"]))

    def test_deserialize_accepts_legacy_torchsave_payload(self) -> None:
        payload = {"stage_idx": 0, "hidden_size": 2560}
        legacy_buffer = io.BytesIO()
        torch.save(payload, legacy_buffer)
        raw = legacy_buffer.getvalue()
        serialized = torch.frombuffer(memoryview(bytearray(raw)), dtype=torch.uint8).clone()

        restored = _deserialize_object_from_uint8(serialized)

        self.assertEqual(restored, payload)


if __name__ == "__main__":
    unittest.main()
