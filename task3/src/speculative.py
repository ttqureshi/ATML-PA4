"""Exact speculative sampling.

The target distribution p and the draft distribution q are next-token distributions
at the current prefix. A draft token x is accepted with probability min(1, p(x) / q(x)).
A rejection draws from the normalized positive residual (p - q)_+. If every one of
the gamma proposals is accepted, one bonus token is drawn from the target. Draft and
target caches are then moved to exactly the tokens that were returned.

The same acceptance rule is used for the categorical tables in the CPU gate and for
the Qwen cache path. On-policy and off-policy distillation do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def acceptance_probability(p_token: float, q_token: float) -> float:
    """Acceptance probability min(1, p(x) / q(x)) for one proposed token."""
    if q_token <= 0.0:
        return 1.0 if p_token <= 0.0 else 0.0
    ratio = p_token / q_token
    if ratio >= 1.0:
        return 1.0
    if ratio <= 0.0:
        return 0.0
    return ratio


def positive_residual(target: np.ndarray, draft: np.ndarray) -> np.ndarray:
    """Normalized (p - q)_+. Returns a copy that sums to one."""
    residual = np.maximum(np.asarray(target, dtype=np.float64) - np.asarray(draft, dtype=np.float64), 0.0)
    total = float(residual.sum())
    if total <= 0.0:
        # Rejection has probability zero when p <= q everywhere (so p = q).
        # Returning p keeps a later caller from dividing by zero.
        fallback = np.asarray(target, dtype=np.float64).copy()
        mass = float(fallback.sum())
        if mass <= 0.0:
            raise ValueError("residual and target are both empty")
        fallback /= mass
        return fallback
    residual /= total
    return residual


def sample_categorical(probs: Sequence[float] | np.ndarray, rng) -> int:
    """Inverse-CDF sample. `rng.random()` must return a float in [0, 1)."""
    u = float(rng.random())
    if u < 0.0:
        u = 0.0
    last = len(probs) - 1
    if last < 0:
        raise ValueError("cannot sample from an empty distribution")
    acc = 0.0
    for i in range(last):
        acc += float(probs[i])
        if u < acc:
            return i
    return last


@dataclass(frozen=True)
class Continuation:
    """Tokens emitted by one speculative round, not including the prior prefix."""

    new_tokens: tuple[int, ...]
    drafted: tuple[int, ...]
    n_accepted: int
    n_examined: int
    rejected: bool
    residual_token: int | None
    bonus_token: int | None
    hit_eos: bool
    suppressed_bonus: bool

    @property
    def n_proposed(self) -> int:
        return len(self.drafted)

    @property
    def n_emitted(self) -> int:
        return len(self.new_tokens)


def select_continuation(
    drafted: Sequence[int],
    q_rows: np.ndarray,
    p_rows: np.ndarray,
    bonus_row: np.ndarray | None,
    *,
    gamma: int,
    max_new: int,
    rng,
    eos_id: int | None,
    residual: str,
) -> Continuation:
    """Accept, reject, or take a bonus. Does not touch caches.

    `q_rows[t]` and `p_rows[t]` are the draft and target distributions that
    proposed and scored `drafted[t]`. `bonus_row` is the target distribution
    after every drafted token. It is used only when all `gamma` tokens are
    accepted. `residual="target"` is the deliberately wrong sampler that draws
    the rejection token from p instead of (p - q)_+.
    """
    drafted_list = [int(token) for token in drafted]
    n = len(drafted_list)
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    if max_new < 0:
        raise ValueError("max_new must be non-negative")
    q_rows = np.asarray(q_rows, dtype=np.float64)
    p_rows = np.asarray(p_rows, dtype=np.float64)
    if n == 0:
        return Continuation((), (), 0, 0, False, None, None, False, False)
    if q_rows.shape[0] != n or p_rows.shape[0] != n:
        raise ValueError("distribution rows must match the drafted tokens")
    if residual not in ("positive", "target"):
        raise ValueError("residual must be 'positive' or 'target'")

    n_accepted = 0
    rejected = False
    examined = 0
    for t, token in enumerate(drafted_list):
        examined += 1
        prob = acceptance_probability(float(p_rows[t, token]), float(q_rows[t, token]))
        if float(rng.random()) <= prob:
            n_accepted += 1
            if eos_id is not None and token == eos_id:
                break
        else:
            rejected = True
            break

    tokens = drafted_list[:n_accepted]
    hit_eos = bool(tokens) and eos_id is not None and tokens[-1] == eos_id
    residual_token = None
    bonus_token = None
    suppressed_bonus = False

    if rejected:
        if residual == "positive":
            dist = positive_residual(p_rows[n_accepted], q_rows[n_accepted])
        else:
            dist = np.asarray(p_rows[n_accepted], dtype=np.float64)
        residual_token = sample_categorical(dist, rng)
        tokens.append(residual_token)
        hit_eos = eos_id is not None and residual_token == eos_id
    elif n_accepted == n and n == gamma and not hit_eos:
        room = len(tokens) < max_new
        if room and bonus_row is not None:
            bonus_token = sample_categorical(np.asarray(bonus_row, dtype=np.float64), rng)
            tokens.append(bonus_token)
            hit_eos = eos_id is not None and bonus_token == eos_id
        else:
            suppressed_bonus = True

    if len(tokens) > max_new:
        tokens = tokens[:max_new]
        n_accepted = min(n_accepted, len(tokens))
        hit_eos = bool(tokens) and eos_id is not None and tokens[-1] == eos_id
        if bonus_token is not None and (not tokens or tokens[-1] != bonus_token):
            bonus_token = None
        if residual_token is not None and (not tokens or tokens[-1] != residual_token):
            residual_token = None
        suppressed_bonus = True

    return Continuation(
        new_tokens=tuple(tokens),
        drafted=tuple(drafted_list),
        n_accepted=n_accepted,
        n_examined=examined,
        rejected=rejected,
        residual_token=residual_token,
        bonus_token=bonus_token,
        hit_eos=hit_eos,
        suppressed_bonus=suppressed_bonus,
    )


class CategoricalTable:
    """Tiny next-token table with an explicit token cache.

    `initial` is the distribution on an empty cache. With `transitions`, the
    row `transitions[last]` is the distribution after a non-empty cache. Without
    transitions the table is i.i.d. at every position.
    """

    def __init__(self, initial: Sequence[float] | np.ndarray, transitions: np.ndarray | None = None):
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        if self.initial.ndim != 1:
            raise ValueError("initial distribution must be one-dimensional")
        self.transitions = None if transitions is None else np.asarray(transitions, dtype=np.float64).copy()
        if self.transitions is not None:
            if self.transitions.shape != (self.initial.size, self.initial.size):
                raise ValueError("transitions must have shape (vocab, vocab)")
        self.tokens: list[int] = []

    @property
    def vocab_size(self) -> int:
        return int(self.initial.size)

    def length(self) -> int:
        return len(self.tokens)

    def probs(self) -> np.ndarray:
        if self.transitions is None or not self.tokens:
            return self.initial
        return self.transitions[self.tokens[-1]]

    def push(self, token: int) -> None:
        self.tokens.append(int(token))

    def rollback(self, length: int) -> None:
        if length < 0 or length > len(self.tokens):
            raise ValueError(f"cannot roll cache of length {len(self.tokens)} back to {length}")
        del self.tokens[length:]


def _seed_tables(target: CategoricalTable, draft: CategoricalTable, prefix: Sequence[int]) -> None:
    if target.length() or draft.length():
        raise ValueError("tables must start empty; pass the prefix to speculative_generate")
    for token in prefix:
        target.push(int(token))
        draft.push(int(token))


def speculative_generate(
    target: CategoricalTable,
    draft: CategoricalTable,
    *,
    gamma: int,
    max_new_tokens: int,
    rng,
    prefix: Sequence[int] = (),
    eos_id: int | None = None,
    pad_id: int | None = None,
    residual: str = "positive",
    trace: bool = True,
) -> "TableGeneration":
    """Draw up to `max_new_tokens` from the target law using exact speculative sampling.

    Both caches are seeded with `prefix` and, after every round, have length equal
    to the accepted prefix (the original prefix plus every token returned so far).
    """
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft vocabularies differ")
    _seed_tables(target, draft, prefix)
    generated: list[int] = []
    rounds: list[Continuation] = []
    hit_eos = False
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        continuation, _ = _table_round(
            target,
            draft,
            gamma=gamma,
            max_new=remaining,
            rng=rng,
            eos_id=eos_id,
            pad_id=pad_id,
            residual=residual,
        )
        if trace:
            rounds.append(continuation)
        if not continuation.new_tokens:
            break
        generated.extend(continuation.new_tokens)
        if continuation.hit_eos:
            hit_eos = True
            break
    stopped_early = len(generated) >= max_new_tokens and not hit_eos
    return TableGeneration(
        tokens=tuple(generated),
        rounds=tuple(rounds),
        hit_eos=hit_eos,
        length_capped=stopped_early,
        cache_tokens=tuple(target.tokens),
        draft_cache_tokens=tuple(draft.tokens),
    )


@dataclass(frozen=True)
class TableGeneration:
    tokens: tuple[int, ...]
    rounds: tuple[Continuation, ...]
    hit_eos: bool
    length_capped: bool
    cache_tokens: tuple[int, ...]
    draft_cache_tokens: tuple[int, ...]


def _table_round(
    target: CategoricalTable,
    draft: CategoricalTable,
    *,
    gamma: int,
    max_new: int,
    rng,
    eos_id: int | None,
    pad_id: int | None,
    residual: str,
) -> tuple[Continuation, int]:
    base = target.length()
    if draft.length() != base or target.tokens != draft.tokens:
        raise RuntimeError("draft and target caches diverged before the round")
    proposal_budget = min(gamma, max_new)
    drafted: list[int] = []
    q_rows: list[np.ndarray] = []
    for _ in range(proposal_budget):
        q_dist = np.asarray(draft.probs(), dtype=np.float64)
        token = sample_categorical(q_dist, rng)
        if pad_id is not None and token == pad_id:
            break
        drafted.append(token)
        q_rows.append(q_dist.copy())
        draft.push(token)
        if eos_id is not None and token == eos_id:
            break

    p_rows: list[np.ndarray] = []
    for token in drafted:
        p_rows.append(np.asarray(target.probs(), dtype=np.float64).copy())
        target.push(token)
    bonus_row = np.asarray(target.probs(), dtype=np.float64).copy() if drafted else None
    if drafted:
        continuation = select_continuation(
            drafted,
            np.stack(q_rows),
            np.stack(p_rows),
            bonus_row,
            gamma=gamma,
            max_new=max_new,
            rng=rng,
            eos_id=eos_id,
            residual=residual,
        )
    else:
        continuation = select_continuation(
            [],
            np.zeros((0, target.vocab_size)),
            np.zeros((0, target.vocab_size)),
            None,
            gamma=gamma,
            max_new=max_new,
            rng=rng,
            eos_id=eos_id,
            residual=residual,
        )

    final_tokens = list(target.tokens[:base]) + list(continuation.new_tokens)
    target.rollback(base)
    draft.rollback(base)
    for token in continuation.new_tokens:
        target.push(token)
        draft.push(token)
    if target.tokens != final_tokens or draft.tokens != final_tokens:
        raise RuntimeError("cache rollback did not land on the accepted prefix")
    if target.length() != len(final_tokens) or draft.length() != len(final_tokens):
        raise RuntimeError("cache length does not equal the accepted prefix")
    return continuation, base


def markov_sequence_probabilities(
    initial: np.ndarray,
    transitions: np.ndarray,
    length: int,
) -> np.ndarray:
    """Exact joint P(x_1..x_length) under a first-order Markov target.

    `transitions[i, j]` is P(next = j | current = i). The returned array has
    shape `(vocab,) * length` and is the target-only ancestral distribution.
    """
    from itertools import product

    initial = np.asarray(initial, dtype=np.float64)
    transitions = np.asarray(transitions, dtype=np.float64)
    vocab = int(initial.size)
    if length < 1:
        raise ValueError("length must be positive")
    joint = np.zeros((vocab,) * length, dtype=np.float64)
    for seq in product(range(vocab), repeat=length):
        prob = float(initial[seq[0]])
        for left, right in zip(seq, seq[1:]):
            prob *= float(transitions[left, right])
        joint[seq] = prob
    return joint


def strip_padding(token_ids: Sequence[int], pad_id: int) -> list[int]:
    """Drop a proposal at the first pad. Pad is not part of the accepted prefix."""
    kept: list[int] = []
    for token in token_ids:
        if int(token) == pad_id:
            break
        kept.append(int(token))
    return kept


def logits_to_probs(logits, temperature: float = 1.0, top_p: float = 1.0) -> np.ndarray:
    """Temperature-1, top-p-1 softmax by default. Returns a float64 probability vector."""
    import torch

    scores = logits.detach().float()
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if temperature != 1.0:
        scores = scores / temperature
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        ordered, indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        remove = cumulative - ordered > top_p
        ordered = ordered.masked_fill(remove, 0.0)
        renorm = torch.zeros_like(probs)
        renorm.scatter_(-1, indices, ordered)
        mass = renorm.sum()
        if float(mass) <= 0.0:
            raise ValueError("top-p filter removed every token")
        probs = renorm / mass
    return probs.cpu().numpy().astype(np.float64)


def cache_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache.length)


class KVCacheState:
    """Token ids plus a transformers cache, kept at the same length.

    `crop` uses the cache object's `crop` when the model has one. After every
    forward or rollback, `seq_len` equals both `len(token_ids)` and the cache
    length. `next_logits` is the next-token distribution at that prefix.
    """

    def __init__(self, model):
        self.model = model
        self.cache = None
        self.token_ids: list[int] = []
        self.next_logits = None

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)

    def prefill(self, token_ids: Sequence[int]) -> None:
        self.cache = None
        self.token_ids = []
        self.next_logits = None
        if len(token_ids) == 0:
            raise ValueError("prefill needs at least one token")
        self.forward_tokens([int(token) for token in token_ids])

    def forward_tokens(self, token_ids: Sequence[int]):
        import torch

        tokens = [int(token) for token in token_ids]
        if not tokens:
            raise ValueError("nothing to forward")
        device = _module_device(self.model)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        past_len = cache_length(self.cache)
        if past_len != len(self.token_ids):
            raise RuntimeError(
                f"cache length {past_len} does not match stored prefix {len(self.token_ids)}"
            )
        position = torch.arange(past_len, past_len + len(tokens), device=device)
        kwargs = {"input_ids": ids, "use_cache": True, "cache_position": position}
        if self.cache is not None:
            kwargs["past_key_values"] = self.cache
        output = self.model(**kwargs)
        self.cache = output.past_key_values
        self.token_ids.extend(tokens)
        logits = output.logits[0]
        self.next_logits = logits[-1]
        observed = cache_length(self.cache)
        if observed != len(self.token_ids):
            raise RuntimeError(f"forward left cache length {observed}, expected {len(self.token_ids)}")
        return logits

    def crop_to(self, length: int) -> None:
        if length < 0 or length > len(self.token_ids):
            raise ValueError(f"cannot crop cache of length {len(self.token_ids)} to {length}")
        if length == len(self.token_ids):
            return
        if length == 0:
            self.cache = None
            self.token_ids = []
            self.next_logits = None
            return
        if self.cache is None or not hasattr(self.cache, "crop"):
            raise RuntimeError("cache cannot be rolled back")
        self.cache.crop(length)
        self.token_ids = self.token_ids[:length]
        self.next_logits = None
        observed = cache_length(self.cache)
        if observed != length:
            raise RuntimeError(f"rollback left cache length {observed}, expected {length}")

    def align(self, final_ids: Sequence[int]) -> None:
        """Roll the cache forward or back until it equals `final_ids`."""
        final = [int(token) for token in final_ids]
        keep = 0
        limit = min(len(self.token_ids), len(final))
        while keep < limit and self.token_ids[keep] == final[keep]:
            keep += 1
        self.crop_to(keep)
        missing = final[keep:]
        if missing:
            self.forward_tokens(missing)
        elif self.next_logits is None and self.token_ids:
            last = self.token_ids[-1]
            self.crop_to(len(self.token_ids) - 1)
            self.forward_tokens([last])
        if self.token_ids != final:
            raise RuntimeError("aligned cache tokens do not equal the accepted prefix")
        if cache_length(self.cache) != len(final):
            raise RuntimeError("aligned cache length does not equal the accepted prefix")


def _module_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", None) not in (None, "meta"):
        try:
            if device.type != "meta":
                return device
        except AttributeError:
            return device
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


@dataclass(frozen=True)
class RoundTiming:
    draft_proposal_s: float
    target_verification_s: float
    acceptance_residual_s: float
    cache_rollback_s: float


@dataclass(frozen=True)
class SpeculativeResult:
    tokens: tuple[int, ...]
    rounds: tuple[Continuation, ...]
    timings: tuple[RoundTiming, ...]
    hit_eos: bool
    length_capped: bool
    prefill_s: float
    cache_tokens: tuple[int, ...]


class SpeculativeDecoder:
    """Exact speculative sampler on two cached causal language models.

    Random choices use `rng.random()`, the same inverse-CDF sampler as the
    categorical gate, so acceptance does not depend on a CUDA multinomial.
    Temperature 1 and top-p 1 leave the model softmax unchanged.
    """

    def __init__(self, target_model, draft_model, rng, temperature: float = 1.0, top_p: float = 1.0):
        if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("temperature must be positive and top_p must lie in (0, 1]")
        self.target = KVCacheState(target_model)
        self.draft = KVCacheState(draft_model)
        self.rng = rng
        self.temperature = temperature
        self.top_p = top_p
        self.prefill_s = 0.0

    def prefill(self, token_ids: Sequence[int]) -> None:
        from shared.timing import cuda_sync

        cuda_sync()
        start = _now()
        ids = [int(token) for token in token_ids]
        self.target.prefill(ids)
        self.draft.prefill(ids)
        cuda_sync()
        self.prefill_s = _now() - start
        if self.target.token_ids != ids or self.draft.token_ids != ids:
            raise RuntimeError("prefill cache does not match the prompt")

    def generate(
        self,
        *,
        gamma: int,
        max_new_tokens: int,
        eos_id: int | None = None,
        pad_id: int | None = None,
        residual: str = "positive",
        ignore_eos: bool = False,
    ) -> SpeculativeResult:
        if self.target.seq_len == 0:
            raise RuntimeError("call prefill before generate")
        stop_id = None if ignore_eos else eos_id
        generated: list[int] = []
        rounds: list[Continuation] = []
        timings: list[RoundTiming] = []
        hit_eos = False
        while len(generated) < max_new_tokens:
            continuation, timing = self.one_round(
                gamma=gamma,
                max_new=max_new_tokens - len(generated),
                eos_id=stop_id,
                pad_id=pad_id,
                residual=residual,
            )
            rounds.append(continuation)
            timings.append(timing)
            if not continuation.new_tokens:
                break
            generated.extend(continuation.new_tokens)
            if continuation.hit_eos:
                hit_eos = True
                break
        return SpeculativeResult(
            tokens=tuple(generated),
            rounds=tuple(rounds),
            timings=tuple(timings),
            hit_eos=hit_eos,
            length_capped=len(generated) >= max_new_tokens and not hit_eos,
            prefill_s=self.prefill_s,
            cache_tokens=tuple(self.target.token_ids),
        )

    def one_round(
        self,
        *,
        gamma: int,
        max_new: int,
        eos_id: int | None,
        pad_id: int | None,
        residual: str,
    ) -> tuple[Continuation, RoundTiming]:
        from shared.timing import cuda_sync

        if self.target.token_ids != self.draft.token_ids:
            raise RuntimeError("draft and target caches diverged before the round")
        base_ids = list(self.target.token_ids)
        budget = min(gamma, max_new)

        cuda_sync()
        draft_start = _now()
        drafted: list[int] = []
        q_rows: list[np.ndarray] = []
        for _ in range(budget):
            draft_probs = logits_to_probs(self.draft.next_logits, self.temperature, self.top_p)
            token = sample_categorical(draft_probs, self.rng)
            if pad_id is not None and token == pad_id:
                break
            drafted.append(token)
            q_rows.append(draft_probs)
            self.draft.forward_tokens([token])
            if eos_id is not None and token == eos_id:
                break
        cuda_sync()
        draft_s = _now() - draft_start

        verify_start = _now()
        bonus_row = None
        if drafted:
            first = logits_to_probs(self.target.next_logits, self.temperature, self.top_p)
            verified = self.target.forward_tokens(drafted)
            p_rows = [first]
            for index in range(1, len(drafted)):
                p_rows.append(logits_to_probs(verified[index - 1], self.temperature, self.top_p))
            bonus_row = logits_to_probs(verified[-1], self.temperature, self.top_p)
            stacked_q = np.stack(q_rows)
            stacked_p = np.stack(p_rows)
        else:
            vocab = int(self.draft.next_logits.shape[-1]) if self.draft.next_logits is not None else 1
            stacked_q = np.zeros((0, vocab))
            stacked_p = np.zeros((0, vocab))
        cuda_sync()
        verify_s = _now() - verify_start

        decide_start = _now()
        continuation = select_continuation(
            drafted,
            stacked_q,
            stacked_p,
            bonus_row,
            gamma=gamma,
            max_new=max_new,
            rng=self.rng,
            eos_id=eos_id,
            residual=residual,
        )
        cuda_sync()
        decide_s = _now() - decide_start

        rollback_start = _now()
        final_ids = base_ids + list(continuation.new_tokens)
        self.target.align(final_ids)
        self.draft.align(final_ids)
        cuda_sync()
        rollback_s = _now() - rollback_start
        if self.target.seq_len != len(final_ids) or self.draft.seq_len != len(final_ids):
            raise RuntimeError("cache rollback length does not equal the accepted prefix")
        return continuation, RoundTiming(draft_s, verify_s, decide_s, rollback_s)


def _now() -> float:
    import time

    return time.perf_counter()


def sequence_counts(samples: np.ndarray, vocab: int) -> np.ndarray:
    """Count rows of an integer array with shape (n, length)."""
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n, length)")
    length = samples.shape[1]
    counts = np.zeros((vocab,) * length, dtype=np.int64)
    for row in samples:
        counts[tuple(int(token) for token in row)] += 1
    return counts
