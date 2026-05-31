from __future__ import annotations

import unittest

import torch

from scripts import run_rnn_full_history_simclr_aligned as aligned


class RnnFullHistorySimclrAlignedTest(unittest.TestCase):
    def test_strip_padding_removes_only_pad_tokens(self) -> None:
        self.assertEqual(aligned.strip_padding([0, 0, 12, 4, 0, 8]), [12, 4, 8])

    def test_tensor_to_sequences_strips_padding_per_row(self) -> None:
        tensor = torch.tensor([[0, 3, 4], [0, 0, 9], [7, 0, 0]])

        sequences = aligned.tensor_to_sequences(tensor)

        self.assertEqual(sequences, [[3, 4], [9], [7]])

    def test_build_split_frame_preserves_split_membership(self) -> None:
        split_user_ids = {
            "train": ["u1", "u2"],
            "val": ["u3"],
            "test": ["u4"],
        }

        frame = aligned.build_split_frame(split_user_ids)

        self.assertEqual(frame.height, 4)
        self.assertEqual(
            frame.sort("appmetrica_device_id")["split"].to_list(),
            ["train", "train", "val", "test"],
        )


if __name__ == "__main__":
    unittest.main()
