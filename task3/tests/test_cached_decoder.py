"""The cached speculative decoder against a no-cache reference on tiny random Qwen2 models.

The reference runs the same sampler with a full forward over the whole sequence
for every distribution it needs, and consumes the same NumPy RNG stream. The
cached decoder must emit identical tokens and identical per-round target and
draft rows (FP32 tolerance), for gamma 1, 2 and 4, including full acceptance
with a bonus token, rejection at the first proposal, EOS in the middle of a
round, and a token cap that ends a round early. No weights are downloaded.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from task3.src.evaluate import cache_kv_max_abs, cache_matches_full_forward, split_prefill_max_abs
from task3.src.speculative import (
    SpeculativeDecoder,
    logits_to_probs,
    sample_categorical,
    select_continuation,
)

VOCAB = 48
ROW_TOL = 1e-5


def tiny_qwen(seed: int, sharpness: float = 6.0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config).eval()
    with torch.no_grad():
        # Random init gives near-uniform rows; sharpen them so target and draft disagree.
        model.lm_head.weight.mul_(sharpness)
    return model


def _full_probs(model, ids: list[int]) -> np.ndarray:
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    return np.stack([logits_to_probs(row) for row in logits])


def reference_generate(target, draft, prompt, *, gamma, max_new, eos_id, rng):
    """Same sampler, no cache: every distribution comes from a full forward."""
    committed = list(prompt)
    generated: list[int] = []
    rounds = []
    rows = []
    while len(generated) < max_new:
        budget = min(gamma, max_new - len(generated))
        drafted: list[int] = []
        q_rows = []
        for _ in range(budget):
            q = _full_probs(draft, committed + drafted)[-1]
            token = sample_categorical(q, rng)
            drafted.append(token)
            q_rows.append(q)
            if eos_id is not None and token == eos_id:
                break
        p_all = _full_probs(target, committed + drafted)
        p_rows = p_all[len(committed) - 1 : len(committed) - 1 + len(drafted)]
        bonus = p_all[len(committed) - 1 + len(drafted)]
        continuation = select_continuation(
            drafted,
            np.stack(q_rows),
            p_rows,
            bonus,
            gamma=gamma,
            max_new=max_new - len(generated),
            rng=rng,
            eos_id=eos_id,
            residual="positive",
        )
        rounds.append(continuation)
        rows.append({"q": np.stack(q_rows), "p": p_rows, "bonus": bonus})
        committed.extend(continuation.new_tokens)
        generated.extend(continuation.new_tokens)
        if not continuation.new_tokens or continuation.hit_eos:
            break
    return generated, rounds, rows


def cached_generate(target, draft, prompt, *, gamma, max_new, eos_id, seed):
    decoder = SpeculativeDecoder(target, draft, np.random.default_rng(seed), record_rows=True)
    with torch.no_grad():
        decoder.prefill(prompt)
        result = decoder.generate(gamma=gamma, max_new_tokens=max_new, eos_id=eos_id)
    return decoder, result


class CachedDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = tiny_qwen(11)
        cls.draft = tiny_qwen(12)
        cls.coverage = {"first_rejection": 0, "bonus": 0, "eos_mid_round": 0, "cap_short_round": 0}

    def _compare(self, target, draft, prompt, *, gamma, max_new, eos_id, seed):
        decoder, result = cached_generate(target, draft, prompt, gamma=gamma, max_new=max_new, eos_id=eos_id, seed=seed)
        ref_tokens, ref_rounds, ref_rows = reference_generate(
            target, draft, prompt, gamma=gamma, max_new=max_new, eos_id=eos_id, rng=np.random.default_rng(seed)
        )
        context = f"gamma={gamma} seed={seed} eos={eos_id} max_new={max_new}"
        self.assertEqual(list(result.tokens), ref_tokens, context)
        self.assertEqual(len(result.rounds), len(ref_rounds), context)
        for mine, theirs in zip(result.rounds, ref_rounds):
            self.assertEqual(mine, theirs, context)
        for mine, theirs in zip(decoder.row_log, ref_rows):
            self.assertLess(float(np.abs(mine["q"] - theirs["q"]).max()), ROW_TOL, context)
            self.assertLess(float(np.abs(mine["p"] - theirs["p"]).max()), ROW_TOL, context)
            self.assertLess(float(np.abs(mine["bonus"] - theirs["bonus"]).max()), ROW_TOL, context)
        # Work accounting under the new round structure.
        for continuation, timing in zip(result.rounds, result.timings):
            self.assertEqual(timing.draft_forwards, continuation.n_proposed, context)
            self.assertEqual(timing.target_input_tokens, continuation.n_proposed + 1, context)
            self.assertLessEqual(timing.draft_input_tokens, continuation.n_proposed + 1, context)
        # Caches sit on the emitted sequence minus its pending tail, then sync exactly.
        expected = list(prompt) + list(result.tokens)
        self.assertEqual(decoder.committed, expected)
        self.assertEqual(decoder.target.token_ids, expected[:-1])
        self.assertIn(decoder.draft.seq_len, (len(expected) - 1, len(expected) - 2))
        with torch.no_grad():
            decoder.sync_caches()
        self.assertLess(cache_kv_max_abs(target, decoder.target.cache, expected), 1e-5)
        self.assertLess(cache_kv_max_abs(draft, decoder.draft.cache, expected), 1e-5)
        for index, item in enumerate(result.rounds):
            if item.rejected and item.n_accepted == 0:
                self.coverage["first_rejection"] += 1
            if item.bonus_token is not None:
                self.coverage["bonus"] += 1
            if item.hit_eos and item.residual_token is None and item.bonus_token is None and item.n_accepted < gamma:
                self.coverage["eos_mid_round"] += 1
            if item.n_proposed < gamma and index == len(result.rounds) - 1 and not item.hit_eos:
                self.coverage["cap_short_round"] += 1
        return result

    def test_matches_reference_for_each_gamma(self) -> None:
        rng = np.random.default_rng(6304)
        for gamma in (1, 2, 4):
            for seed in range(6):
                prompt = [int(x) for x in rng.integers(0, VOCAB, size=int(rng.integers(1, 7)))]
                self._compare(self.target, self.draft, prompt, gamma=gamma, max_new=19, eos_id=None, seed=seed)
        self.assertGreater(self.coverage["first_rejection"], 0)
        self.assertGreater(self.coverage["bonus"], 0)
        self.assertGreater(self.coverage["cap_short_round"], 0)

    def test_full_acceptance_takes_a_bonus_every_round(self) -> None:
        prompt = [3, 1, 4, 1, 5]
        for gamma in (1, 2, 4):
            # Draft equal to target: p/q = 1, so every proposal is accepted.
            result = self._compare(self.target, self.target, prompt, gamma=gamma, max_new=4 * (gamma + 1), eos_id=None, seed=gamma)
            self.assertTrue(all(item.n_accepted == gamma and item.bonus_token is not None for item in result.rounds))
            self.assertEqual(len(result.rounds), 4)
            # A cap that is not a multiple of gamma + 1 ends with a short round and no bonus.
            capped = self._compare(self.target, self.target, prompt, gamma=gamma, max_new=gamma + 1 + gamma, eos_id=None, seed=gamma)
            self.assertEqual(len(capped.tokens), 2 * gamma + 1)
            self.assertTrue(capped.rounds[-1].suppressed_bonus)
            self.assertIsNone(capped.rounds[-1].bonus_token)

    def test_eos_inside_a_round(self) -> None:
        prompt = [7, 8, 9]
        found = 0
        for gamma in (2, 4):
            for seed in range(8):
                _decoder, plain = cached_generate(self.target, self.draft, prompt, gamma=gamma, max_new=24, eos_id=None, seed=seed)
                for eos in sorted(set(plain.tokens)):
                    _decoder, trial = cached_generate(self.target, self.draft, prompt, gamma=gamma, max_new=24, eos_id=eos, seed=seed)
                    last = trial.rounds[-1]
                    # EOS proposed and accepted before the proposal budget ran out.
                    if trial.hit_eos and last.residual_token is None and last.bonus_token is None and last.n_accepted < gamma:
                        result = self._compare(self.target, self.draft, prompt, gamma=gamma, max_new=24, eos_id=eos, seed=seed)
                        self.assertEqual(result.tokens[-1], eos)
                        self.assertNotIn(eos, result.tokens[:-1])
                        self.assertEqual(last.n_proposed, last.n_accepted)
                        found += 1
                        break
        self.assertGreater(found, 0)
        self.assertGreater(self.coverage["eos_mid_round"], 0)

    def test_split_prefill_and_cache_check_helpers_fp32(self) -> None:
        ids = [int(x) for x in np.random.default_rng(1).integers(0, VOCAB, size=30)]
        self.assertLess(split_prefill_max_abs(self.target, ids, split=11), 1e-5)
        report = cache_matches_full_forward(self.target, self.draft, ids[:6], eos_id=None, max_new_tokens=20)
        self.assertTrue(report["length_ok"])
        self.assertLess(report["next_logit_max_abs"], 1e-5)
        self.assertLess(report["target_kv_max_abs"], 1e-5)
        self.assertLess(report["draft_kv_max_abs"], 1e-5)
        self.assertGreater(report["rounds_with_rejection"], 0)


if __name__ == "__main__":
    unittest.main()
