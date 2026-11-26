"""CPU correctness gate for exact speculative sampling.

The length-3 Markov experiment draws 100,000 sequences at seed 6304. The null
is the target's ancestral joint, which is also the target-only reference. A
sampler that draws the rejection token from p instead of (p - q)_+ is the
negative control.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

from task3.src.speculative import (
    CategoricalTable,
    acceptance_probability,
    markov_sequence_probabilities,
    positive_residual,
    sample_categorical,
    sequence_counts,
    speculative_generate,
    strip_padding,
)

SEED = 6304
N_SEQUENCES = 100_000
GAMMA = 2
LENGTH = 3
TARGET_INITIAL = np.array([0.34, 0.33, 0.33], dtype=np.float64)
TARGET_TRANSITIONS = np.array(
    [
        [0.50, 0.30, 0.20],
        [0.25, 0.45, 0.30],
        [0.30, 0.20, 0.50],
    ],
    dtype=np.float64,
)
DRAFT_INITIAL = np.array([0.15, 0.20, 0.65], dtype=np.float64)
DRAFT_TRANSITIONS = np.array(
    [
        [0.10, 0.15, 0.75],
        [0.70, 0.15, 0.15],
        [0.15, 0.70, 0.15],
    ],
    dtype=np.float64,
)
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "results_export" / "toy_sampler_chisquare.json"

# Hand distribution used by the scripted paths. p = (0.5, 0.3, 0.2), q = (0.2, 0.2, 0.6).
# a(0) = 1, a(2) = 0.2/0.6 = 1/3, (p-q)_+ = (0.3, 0.1, 0) -> (0.75, 0.25, 0).
HAND_P = np.array([0.5, 0.3, 0.2])
HAND_Q = np.array([0.2, 0.2, 0.6])


class ScriptedRandom:
    def __init__(self, draws: list[float]):
        self._draws = [float(value) for value in draws]
        self.calls = 0

    def random(self) -> float:
        if self.calls >= len(self._draws):
            raise AssertionError("scripted RNG was asked for more draws than were provided")
        value = self._draws[self.calls]
        self.calls += 1
        return value


def _draw_markov(residual: str, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((N_SEQUENCES, LENGTH), dtype=np.int64)
    for index in range(N_SEQUENCES):
        target = CategoricalTable(TARGET_INITIAL, TARGET_TRANSITIONS)
        draft = CategoricalTable(DRAFT_INITIAL, DRAFT_TRANSITIONS)
        generated = speculative_generate(
            target,
            draft,
            gamma=GAMMA,
            max_new_tokens=LENGTH,
            rng=rng,
            residual=residual,
            trace=False,
        )
        if len(generated.tokens) != LENGTH:
            raise AssertionError(f"expected length {LENGTH}, got {generated.tokens}")
        samples[index] = generated.tokens
    return samples


def _pearson(samples: np.ndarray, joint: np.ndarray) -> dict:
    counts = sequence_counts(samples, joint.shape[0])
    expected = joint.reshape(-1) * samples.shape[0]
    observed = counts.reshape(-1).astype(np.float64)
    result = chisquare(observed, expected)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": int(observed.size - 1),
        "min_expected": float(expected.min()),
        "observed": observed,
        "expected": expected,
    }


class ToySamplerTests(unittest.TestCase):
    def test_inverse_cdf_boundaries(self) -> None:
        probs = np.array([0.2, 0.3, 0.5])
        self.assertEqual(sample_categorical(probs, ScriptedRandom([0.0])), 0)
        self.assertEqual(sample_categorical(probs, ScriptedRandom([0.19])), 0)
        self.assertEqual(sample_categorical(probs, ScriptedRandom([0.2])), 1)
        self.assertEqual(sample_categorical(probs, ScriptedRandom([0.49])), 1)
        self.assertEqual(sample_categorical(probs, ScriptedRandom([0.5])), 2)
        self.assertEqual(sample_categorical(np.array([0.0, 0.0, 1.0]), ScriptedRandom([0.0])), 2)

    def test_rejection_residual_hand_probabilities(self) -> None:
        self.assertAlmostEqual(acceptance_probability(0.5, 0.2), 1.0)
        self.assertAlmostEqual(acceptance_probability(0.2, 0.6), 1.0 / 3.0)
        residual = positive_residual(HAND_P, HAND_Q)
        self.assertTrue(np.allclose(residual, [0.75, 0.25, 0.0]))
        # Path: draft 0, draft 2, draft 1, accept 0, reject 2, residual 0.
        # Proposal probabilities: q(0)=0.2, q(2)=0.6, q(1)=0.2.
        # Reject probability of token 2 is 1 - 1/3 = 2/3. Residual puts mass 0.75 on token 0.
        path_probability = 0.2 * 0.6 * 0.2 * 1.0 * (2.0 / 3.0) * 0.75
        self.assertAlmostEqual(path_probability, 0.012)
        generated = speculative_generate(
            CategoricalTable(HAND_P),
            CategoricalTable(HAND_Q),
            gamma=3,
            max_new_tokens=3,
            rng=ScriptedRandom([0.10, 0.50, 0.30, 0.20, 0.90, 0.10, 0.05, 0.10]),
        )
        self.assertEqual(generated.rounds[0].new_tokens, (0, 0))
        self.assertEqual(generated.tokens, (0, 0, 0))
        self.assertEqual(generated.rounds[0].drafted, (0, 2, 1))
        self.assertTrue(generated.rounds[0].rejected)
        self.assertEqual(generated.rounds[0].residual_token, 0)
        self.assertIsNone(generated.rounds[0].bonus_token)
        self.assertEqual(generated.rounds[0].n_accepted, 1)
        self.assertEqual(generated.rounds[0].n_examined, 2)
        self.assertEqual(generated.cache_tokens, generated.tokens)
        self.assertEqual(generated.draft_cache_tokens, generated.tokens)

    def test_bonus_token_comes_from_the_target(self) -> None:
        # Both drafted tokens are 0, which is always accepted. The bonus uniform
        # 0.90 falls in p's last bin, token 2, not in q's mass on token 2 alone.
        generated = speculative_generate(
            CategoricalTable(HAND_P),
            CategoricalTable(HAND_Q),
            gamma=2,
            max_new_tokens=3,
            rng=ScriptedRandom([0.05, 0.05, 0.10, 0.10, 0.90]),
        )
        self.assertEqual(generated.tokens, (0, 0, 2))
        self.assertEqual(generated.rounds[0].bonus_token, 2)
        self.assertFalse(generated.rounds[0].rejected)
        self.assertEqual(generated.rounds[0].n_accepted, 2)
        self.assertEqual(len(generated.cache_tokens), 3)

    def test_cache_rollback_length_matches_accepted_prefix(self) -> None:
        prefix = (0, 1, 2, 0)
        generated = speculative_generate(
            CategoricalTable(HAND_P),
            CategoricalTable(HAND_Q),
            gamma=3,
            max_new_tokens=3,
            rng=ScriptedRandom([0.10, 0.50, 0.30, 0.20, 0.90, 0.10, 0.05, 0.10]),
            prefix=prefix,
        )
        self.assertEqual(generated.rounds[0].new_tokens, (0, 0))
        self.assertEqual(generated.rounds[0].drafted, (0, 2, 1))
        self.assertEqual(generated.tokens, (0, 0, 0))
        self.assertEqual(generated.cache_tokens, prefix + generated.tokens)
        self.assertEqual(generated.draft_cache_tokens, generated.cache_tokens)
        self.assertEqual(len(generated.cache_tokens), len(prefix) + len(generated.tokens))
        self.assertNotEqual(generated.cache_tokens, prefix + (0, 2, 1))

    def test_eos_stops_proposal_without_a_bonus(self) -> None:
        eos = 2
        generated = speculative_generate(
            CategoricalTable([0.2, 0.3, 0.5]),
            CategoricalTable([0.1, 0.1, 0.8]),
            gamma=4,
            max_new_tokens=10,
            rng=ScriptedRandom([0.50, 0.10]),
            eos_id=eos,
        )
        self.assertEqual(generated.tokens, (eos,))
        self.assertTrue(generated.hit_eos)
        self.assertEqual(generated.rounds[0].drafted, (eos,))
        self.assertEqual(generated.rounds[0].n_proposed, 1)
        self.assertIsNone(generated.rounds[0].bonus_token)
        self.assertFalse(generated.rounds[0].rejected)
        self.assertEqual(generated.cache_tokens, (eos,))
        self.assertLess(generated.rounds[0].n_proposed, 4)

    def test_rejected_eos_is_replaced_and_not_followed(self) -> None:
        eos = 2
        generated = speculative_generate(
            CategoricalTable([0.2, 0.3, 0.5]),
            CategoricalTable([0.1, 0.1, 0.8]),
            gamma=4,
            max_new_tokens=1,
            rng=ScriptedRandom([0.50, 0.90, 0.10]),
            eos_id=eos,
        )
        # Residual (p-q)_+ = (0.1, 0.2, 0) / 0.3, and u=0.10 selects token 0.
        self.assertEqual(generated.tokens, (0,))
        self.assertIsNone(generated.rounds[0].bonus_token)
        self.assertTrue(generated.rounds[0].rejected)
        self.assertNotIn(eos, generated.tokens)

    def test_padding_and_short_proposal_do_not_emit_extra_tokens(self) -> None:
        pad = 3
        generated = speculative_generate(
            CategoricalTable([0.4, 0.3, 0.2, 0.1]),
            CategoricalTable([0.2, 0.2, 0.2, 0.4]),
            gamma=4,
            max_new_tokens=2,
            rng=ScriptedRandom([0.05, 0.70, 0.10, 0.70]),
            pad_id=pad,
        )
        self.assertEqual(strip_padding([1, 0, pad, pad], pad), [1, 0])
        self.assertEqual(generated.tokens, (0,))
        self.assertNotIn(pad, generated.cache_tokens)
        self.assertEqual(generated.rounds[0].n_proposed, 1)

        capped = speculative_generate(
            CategoricalTable(HAND_P),
            CategoricalTable(HAND_Q),
            gamma=4,
            max_new_tokens=1,
            rng=ScriptedRandom([0.05, 0.10]),
        )
        self.assertEqual(len(capped.tokens), 1)
        self.assertIsNone(capped.rounds[0].bonus_token)
        self.assertLess(capped.rounds[0].n_proposed, 4)

        no_bonus = speculative_generate(
            CategoricalTable(HAND_P),
            CategoricalTable(HAND_Q),
            gamma=2,
            max_new_tokens=2,
            rng=ScriptedRandom([0.05, 0.05, 0.10, 0.10]),
        )
        self.assertEqual(len(no_bonus.tokens), 2)
        self.assertTrue(no_bonus.rounds[0].suppressed_bonus)
        self.assertIsNone(no_bonus.rounds[0].bonus_token)

    def test_hf_cache_length_matches_accepted_prefix(self) -> None:
        import torch
        from types import SimpleNamespace

        from task3.src.speculative import SpeculativeDecoder

        class FakeCache:
            def __init__(self) -> None:
                self.length = 0

            def get_seq_length(self) -> int:
                return self.length

            def crop(self, max_length: int) -> None:
                if max_length < 0:
                    max_length = self.length + max_length
                if self.length > max_length:
                    self.length = max_length

        class FakeCausal(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.device = torch.device("cpu")

            def forward(self, input_ids, past_key_values=None, use_cache=True, cache_position=None):
                del use_cache, cache_position
                if past_key_values is None:
                    past_key_values = FakeCache()
                past_key_values.length += int(input_ids.shape[-1])
                logits = torch.full((1, int(input_ids.shape[-1]), 4), -1.0)
                logits[..., 0] = 2.0
                return SimpleNamespace(logits=logits, past_key_values=past_key_values)

        prompt = [1, 2, 3]
        decoder = SpeculativeDecoder(FakeCausal(), FakeCausal(), np.random.default_rng(SEED))
        decoder.prefill(prompt)
        result = decoder.generate(gamma=3, max_new_tokens=5, eos_id=None)
        expected = prompt + list(result.tokens)
        self.assertEqual(len(result.tokens), 5)
        self.assertEqual(decoder.target.token_ids, expected)
        self.assertEqual(decoder.draft.token_ids, expected)
        self.assertEqual(decoder.target.cache.get_seq_length(), len(expected))
        self.assertEqual(decoder.draft.cache.get_seq_length(), len(expected))

    def test_markov_chisquare_gate(self) -> None:
        joint = markov_sequence_probabilities(TARGET_INITIAL, TARGET_TRANSITIONS, LENGTH)
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=12)
        expected_counts = joint * N_SEQUENCES
        min_expected = float(expected_counts.min())
        self.assertGreaterEqual(min_expected, 10.0)

        correct = _pearson(_draw_markov("positive"), joint)
        wrong = _pearson(_draw_markov("target"), joint)
        self.assertGreaterEqual(correct["min_expected"], 10.0)
        self.assertGreaterEqual(wrong["min_expected"], 10.0)

        print("\nTarget-only length-3 joint probabilities (also the chi-square null):")
        print("sequence\tprobability\texpected\tobserved_correct\tobserved_wrong")
        joints = []
        flat = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    probability = float(joint[a, b, c])
                    expected = float(correct["expected"][flat])
                    observed_correct = int(correct["observed"][flat])
                    observed_wrong = int(wrong["observed"][flat])
                    print(
                        f"{a}{b}{c}\t{probability:.8f}\t{expected:.2f}\t{observed_correct}\t{observed_wrong}"
                    )
                    joints.append(
                        {
                            "sequence": [a, b, c],
                            "probability": probability,
                            "expected": expected,
                            "observed_correct": observed_correct,
                            "observed_wrong": observed_wrong,
                        }
                    )
                    flat += 1
        print(
            f"correct sampler chi-square={correct['statistic']:.6f} df={correct['df']} "
            f"p={correct['p_value']:.6g} min_expected={correct['min_expected']:.2f}"
        )
        print(
            f"wrong residual-from-p chi-square={wrong['statistic']:.6f} df={wrong['df']} "
            f"p={wrong['p_value']:.6g} min_expected={wrong['min_expected']:.2f}"
        )

        summary = {
            "seed": SEED,
            "n_sequences": N_SEQUENCES,
            "gamma": GAMMA,
            "length": LENGTH,
            "df": correct["df"],
            "null": "target ancestral joint, which is the target-only law",
            "min_expected_count": min_expected,
            "correct_positive_residual": {
                "statistic": correct["statistic"],
                "p_value": correct["p_value"],
                "rejected_at_0.01": correct["p_value"] < 0.01,
            },
            "wrong_residual_from_target": {
                "statistic": wrong["statistic"],
                "p_value": wrong["p_value"],
                "rejected_at_0.01": wrong["p_value"] < 0.01,
            },
            "joint": joints,
        }
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        self.assertGreaterEqual(correct["p_value"], 0.01)
        self.assertLess(wrong["p_value"], 0.01)


if __name__ == "__main__":
    unittest.main()
