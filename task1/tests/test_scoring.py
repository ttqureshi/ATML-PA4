"""Perplexity excludes the first token. HellaSwag acc_norm uses characters."""

from __future__ import annotations

import math
import unittest

import numpy as np

from task1.src.hellaswag import aggregate_accuracy, hellaswag_prompt, score_endings
from task1.src.perplexity import aggregate_perplexity, nll_from_logits, nll_from_token_logprobs


class ScoringTests(unittest.TestCase):
    def test_perplexity_excludes_the_first_token(self) -> None:
        log_probs = np.array([-10.0, -0.2, -0.3, -0.5])
        nll_sum, count = nll_from_token_logprobs(log_probs, score_start=0, score_end=4)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(nll_sum, 1.0)
        second_sum, second_count = nll_from_token_logprobs(np.array([0.0, -2.0]), score_start=0)
        self.assertEqual(second_count, 1)
        pooled = aggregate_perplexity(np.array([nll_sum, second_sum]), np.array([count, second_count]))
        self.assertEqual(pooled["n_tokens"], 4)
        self.assertAlmostEqual(pooled["mean_nll"], 0.75)
        self.assertAlmostEqual(pooled["perplexity"], math.exp(0.75))

        logits = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [4.0, -4.0],
            ]
        )
        tokens = np.array([0, 1, 0])
        total, scored = nll_from_logits(logits, tokens, score_start=0, score_end=3)
        self.assertEqual(scored, 2)
        self.assertAlmostEqual(total, 2.0 * math.log(2.0))
        self.assertAlmostEqual(math.exp(total / scored), 2.0)

    def test_acc_norm_uses_character_length(self) -> None:
        choices = ["a" * 10, "b" * 2, "c" * 10, "d" * 10]
        # Raw argmax is ending 1. Length normalization prefers ending 0.
        scored = score_endings(np.array([-5.0, -2.0, -6.0, -6.0]), choices, gold=0)
        self.assertEqual(scored["prediction"], 1)
        self.assertEqual(scored["prediction_norm"], 0)
        self.assertEqual(scored["correct"], 0)
        self.assertEqual(scored["correct_norm"], 1)
        summary = aggregate_accuracy([scored, {**scored, "correct": 1, "correct_norm": 1}])
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["acc_norm"], 1.0)

    def test_hellaswag_prompt_matches_the_harness_cleanup(self) -> None:
        example = {
            "activity_label": "Cooking",
            "ctx_a": "A cook [title]",
            "ctx_b": "stirs",
            "endings": ["the soup.", "a car.", "a book.", "a shoe."],
            "label": "2",
        }
        query, choices, gold = hellaswag_prompt(example)
        self.assertEqual(gold, 2)
        self.assertNotIn("[title]", query)
        self.assertTrue(query.startswith("Cooking:"))
        self.assertEqual(len(choices), 4)


if __name__ == "__main__":
    unittest.main()
