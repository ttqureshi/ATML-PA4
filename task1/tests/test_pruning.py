"""Hand-checked magnitude, Wanda, and 2:4 masks."""

from __future__ import annotations

import unittest

import numpy as np

from task1.src.pruning import (
    column_l2,
    magnitude_scores,
    two_four_keep_mask,
    uniform_keep_mask,
    wanda_scores,
    zeros_per_row_uniform,
)


WEIGHT = np.array(
    [
        [0.5, 1.5, 0.2, 3.0, 2.0, 0.1],
        [4.0, 0.3, 0.3, 1.0, 2.0, 5.0],
        [1.0, 1.0, 1.0, 1.0, 0.4, 0.2],
        [0.0, -2.0, 0.5, -0.5, 3.0, 0.1],
    ],
    dtype=np.float64,
)

# Two calibration tokens. Column L2 norms are 1, 0, 4, 0, 3, 0.
INPUTS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _zero_columns(mask_row: np.ndarray) -> list[int]:
    return [index for index, keep in enumerate(mask_row.tolist()) if not keep]


class MaskTests(unittest.TestCase):
    def test_magnitude_4x6_matches_hand_ranking(self) -> None:
        self.assertEqual(zeros_per_row_uniform(6, 0.5), 3)
        mask = uniform_keep_mask(magnitude_scores(WEIGHT), 0.5)
        # Smallest |W| per row, ties broken by the lower column index.
        self.assertEqual(_zero_columns(mask[0]), [0, 2, 5])
        self.assertEqual(_zero_columns(mask[1]), [1, 2, 3])
        self.assertEqual(_zero_columns(mask[2]), [0, 4, 5])
        self.assertEqual(_zero_columns(mask[3]), [0, 2, 5])
        self.assertEqual(int((~mask).sum()), 12)

    def test_wanda_4x6_matches_hand_ranking(self) -> None:
        norms = column_l2(INPUTS)
        np.testing.assert_allclose(norms, [1.0, 0.0, 4.0, 0.0, 3.0, 0.0])
        mask = uniform_keep_mask(wanda_scores(WEIGHT, norms), 0.5)
        self.assertEqual(_zero_columns(mask[0]), [1, 3, 5])
        self.assertEqual(_zero_columns(mask[1]), [1, 3, 5])
        self.assertEqual(_zero_columns(mask[2]), [1, 3, 5])
        self.assertEqual(_zero_columns(mask[3]), [0, 1, 3])
        self.assertFalse(np.array_equal(mask, uniform_keep_mask(magnitude_scores(WEIGHT), 0.5)))

    def test_two_four_keeps_two_weights_in_every_group(self) -> None:
        scores = np.array(
            [
                [0.1, 0.4, 0.3, 0.2, 5.0, 1.0, 4.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.2, 0.2, 0.2, 9.0],
            ],
            dtype=np.float64,
        )
        mask = two_four_keep_mask(scores)
        self.assertTrue(np.all(mask.reshape(2, 2, 4).sum(axis=-1) == 2))
        # Group 0 keeps 0.4 and 0.3. The tied 1s in row 1 keep the lower indices.
        self.assertEqual(_zero_columns(mask[0])[:2], [0, 3])
        self.assertEqual(_zero_columns(~mask[1][:4]), [0, 1])
        rng = np.random.default_rng(6304)
        random_scores = rng.normal(size=(5, 12))
        random_mask = two_four_keep_mask(random_scores)
        kept = random_mask.reshape(5, 3, 4).sum(axis=-1)
        self.assertTrue(np.all(kept == 2))


if __name__ == "__main__":
    unittest.main()
