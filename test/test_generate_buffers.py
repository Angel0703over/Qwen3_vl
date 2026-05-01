from __future__ import annotations

import unittest

import torch

from qwen3vl_tp_runtime.hexgen_core.generate_buffers import (
    build_decode_attention_mask_buffer,
    decode_attention_mask_view,
    fill_decode_input_ids,
)


class GenerateBuffersTest(unittest.TestCase):
    def test_decode_attention_mask_buffer_reuses_single_storage(self) -> None:
        prefill_mask = torch.ones((1, 5), dtype=torch.long)

        buffer = build_decode_attention_mask_buffer(prefill_mask, max_new_tokens=4)
        step0 = decode_attention_mask_view(buffer, prefill_seq_len=5, step_idx=0)
        step2 = decode_attention_mask_view(buffer, prefill_seq_len=5, step_idx=2)

        self.assertEqual(tuple(buffer.shape), (1, 8))
        self.assertEqual(tuple(step0.shape), (1, 6))
        self.assertEqual(tuple(step2.shape), (1, 8))
        self.assertEqual(step0.untyped_storage().data_ptr(), buffer.untyped_storage().data_ptr())
        self.assertEqual(step2.untyped_storage().data_ptr(), buffer.untyped_storage().data_ptr())

    def test_fill_decode_input_ids_reuses_buffer(self) -> None:
        token_buffer = torch.empty((1, 1), dtype=torch.long)

        result = fill_decode_input_ids(token_buffer, 123)

        self.assertEqual(result.data_ptr(), token_buffer.data_ptr())
        self.assertEqual(int(token_buffer.item()), 123)


if __name__ == "__main__":
    unittest.main()
