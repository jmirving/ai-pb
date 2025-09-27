"""Unit tests for :mod:`pbai.data.processors.draft_processor`."""

import unittest
import pandas as pd

from pbai.data.processors.draft_processor import DraftProcessor


class DraftProcessorTest(unittest.TestCase):
    """Tests for :class:`DraftProcessor` behaviour on preprocessed rows."""

    def setUp(self):
        # Each test can reuse a processor instance because it has no state.
        self.processor = DraftProcessor()

    def test_process_returns_tuple_of_draft_sequence_target_and_mask(self):
        """Ensure ``process`` exposes the draft sequence, target, and mask."""
        # Arrange: build a minimal row with the 20-slot sequence the processor expects.
        row = pd.Series({
            "draft_sequence": list(range(1, 21)),
            "target": 123,
            "already_picked_or_banned": {5, 6, 7},
        })

        # Act: run the processor on the prepared row.
        draft_sequence, target, already_taken = self.processor.process(row)

        # Assert: the tuple contains every column that downstream code consumes.
        self.assertEqual(list(range(1, 21)), draft_sequence)
        self.assertEqual(123, target)
        self.assertEqual({5, 6, 7}, already_taken)

    def test_already_picked_or_banned_is_always_returned_as_a_set(self):
        """``process`` should coerce list inputs into a set for fast membership tests."""
        # Arrange: duplicate entries mimic a fearless history carried through from the dataset.
        row = pd.Series({
            "draft_sequence": [0] * 20,
            "target": 42,
            "already_picked_or_banned": [1, 1, 2, 3],
        })

        # Act: process the row and capture the mask output.
        _, _, already_taken = self.processor.process(row)

        # Assert: duplicates are removed and membership lookups remain O(1).
        self.assertIsInstance(already_taken, set)
        self.assertEqual({1, 2, 3}, already_taken)


if __name__ == "__main__":
    unittest.main()
