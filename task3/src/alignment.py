"""Common-prefix distribution alignment between the target and every draft.

Banks (all scored by every draft, on the same prefix ids and positions):

- `teacher` and `original_draft`: the supplied banks from the course
  `setup_data.py --prefix-banks` (prompt_id, domain in {math, short}, bank,
  position, prompt, response_prefix_token_ids = seq[:position - 1],
  source_length, source_generation_seed).
- `exact_spec_original_gamma4`: generated here with the exact sampler and the
  original draft at gamma = 4; the pure target law (temperature 1, top-p 1).
- `teacher_no_rep_penalty`: plain target sampling (temperature 1, top-p 1, no
  repetition penalty), same caps and seeds as `teacher`. The supplied banks
  come from `generate()`, which keeps Qwen2.5's generation_config
  repetition_penalty = 1.1, so this bank is the matched-law control for the
  speculative bank in the consistency check.
- `student_{method}_seed{seed}`: each trained student's own frozen rollouts,
  with the producing checkpoint recorded; every draft is scored on each.

Every prefix row is one token position, so all averages are token-weighted.
A (bank, prompt) group shares one source response, so one forward of the
longest prefix gives the next-token distributions at all of its positions:
the logit at index len(prompt) - 1 + (position - 1) predicts response token
`position`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from shared.bootstrap import bootstrap_paired_mean_diff
from task3.src.distill import load_jsonl, render_chat
from task3.src.metrics import DOMAINS, POSITION_BINS, aggregate_token_metrics, position_bin

SUPPLIED_BANKS = ("teacher", "original_draft")
SPEC_BANK = "exact_spec_original_gamma4"
TARGET_LAW_BANK = "teacher_no_rep_penalty"
CONSISTENCY_NOTE = (
    "target_vs_speculative_consistency compares the supplied teacher bank (course generate() call, which keeps "
    "Qwen2.5's generation_config repetition_penalty=1.1) with exact speculative prefixes (pure temperature-1 target "
    "law), so a gap there can come from the penalty. target_law_control_consistency compares an unpenalized "
    "temperature-1 target bank with the same speculative bank: that is the matched-law check of the sampler."
)
METRIC_KEYS = ("forward_kl", "reverse_kl", "tv", "greedy_agreement", "expected_acceptance")
TOP_K = 5
REQUIRED_FIELDS = ("prompt_id", "domain", "bank", "position", "prompt", "response_prefix_token_ids")


def domain_of(prompt_id: str, stored: str | None = None) -> str:
    """math / familiar / shifted. The supplied banks store `short` for both extraction splits."""
    if stored in DOMAINS:
        return stored
    if prompt_id.startswith("GSM8K"):
        return "math"
    if "familiar" in prompt_id:
        return "familiar"
    if "shifted" in prompt_id:
        return "shifted"
    raise ValueError(f"cannot assign a domain to {prompt_id} (stored {stored!r})")


def load_prefix_bank(path: Path) -> list[dict]:
    """Parse the supplied bank file. A missing file is an error, never an empty bank."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prefix bank {path}. Build it with `python task3/run.py prefix_banks` on CUDA "
            "(the course setup_data.py --prefix-banks call fails under transformers 4.57.1)."
        )
    rows = []
    for record in load_jsonl(path):
        missing = [key for key in REQUIRED_FIELDS if key not in record]
        if missing:
            raise ValueError(f"prefix bank record lacks {missing}: {sorted(record)}")
        position = int(record["position"])
        prefix = [int(token) for token in record["response_prefix_token_ids"]]
        if position < 1 or len(prefix) != position - 1:
            raise ValueError(f"{record['prompt_id']}/{record['bank']}: position {position} with {len(prefix)} prefix tokens")
        if record["bank"] not in SUPPLIED_BANKS:
            raise ValueError(f"unexpected supplied bank {record['bank']!r}")
        prompt_id = str(record["prompt_id"])
        rows.append(
            {
                "prompt_id": prompt_id,
                "domain": domain_of(prompt_id, record.get("domain")),
                "bank": str(record["bank"]),
                "position": position,
                "prompt": record["prompt"],
                "response_prefix_token_ids": prefix,
                "producer": str(record["bank"]),
                "source_length": record.get("source_length"),
                "source_generation_seed": record.get("source_generation_seed"),
            }
        )
    if not rows:
        raise ValueError(f"prefix bank {path} is empty")
    return rows


def group_rows(rows: Sequence[dict]) -> list[dict]:
    """Group rows that are prefixes of one source response, keyed by (bank, producer, prompt)."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["bank"], row.get("producer", row["bank"]), row["prompt_id"])].append(row)
    groups = []
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda row: int(row["position"]))
        longest = list(members[-1]["response_prefix_token_ids"])
        pending: list[dict] = []
        for row in members:
            prefix = list(row["response_prefix_token_ids"])
            if prefix == longest[: len(prefix)]:
                pending.append(row)
            else:
                # Not one source response: score it on its own.
                groups.append({"key": key, "prefix": prefix, "rows": [row]})
        if pending:
            groups.append({"key": key, "prefix": longest, "rows": pending})
    return groups


def _row_logits(model, ids: list[int], indices: list[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=False).logits[0]
    return logits[torch.tensor(indices, device=logits.device)].float()


def pair_metrics(teacher_logits: torch.Tensor, draft_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Row-wise forward KL, reverse KL, TV, greedy agreement and 1 - TV over the full vocabulary."""
    if teacher_logits.shape != draft_logits.shape:
        raise RuntimeError(f"logit shapes differ: {tuple(teacher_logits.shape)} vs {tuple(draft_logits.shape)}")
    p_log = torch.log_softmax(teacher_logits.double(), dim=-1)
    q_log = torch.log_softmax(draft_logits.to(p_log.device).double(), dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    tv = 0.5 * (p - q).abs().sum(-1)
    return {
        "forward_kl": (p * (p_log - q_log)).sum(-1),
        "reverse_kl": (q * (q_log - p_log)).sum(-1),
        "tv": tv,
        "greedy_agreement": (p.argmax(-1) == q.argmax(-1)).double(),
        "expected_acceptance": 1.0 - tv,
    }


def _top(logits: torch.Tensor) -> tuple[list[list[int]], list[list[float]]]:
    probs = torch.softmax(logits.double(), dim=-1)
    values, indices = probs.topk(TOP_K, dim=-1)
    return indices.tolist(), values.tolist()


def score_bank(teacher, drafts: dict, tokenizer, rows: Sequence[dict], *, keep_top: bool = True) -> list[dict]:
    """Score every draft against the teacher on the same rows. One teacher forward per group."""
    scored = []
    for group in group_rows(rows):
        first = group["rows"][0]
        prompt_ids = render_chat(tokenizer, first["prompt"])
        ids = prompt_ids + list(group["prefix"])
        indices = [len(prompt_ids) - 1 + int(row["position"]) - 1 for row in group["rows"]]
        teacher_logits = _row_logits(teacher, ids, indices)
        p_log = torch.log_softmax(teacher_logits.double(), dim=-1)
        entropy = (-(p_log.exp() * p_log).sum(-1)).tolist()
        teacher_top = _top(teacher_logits) if keep_top else None
        for name, model in drafts.items():
            draft_logits = _row_logits(model, ids, indices).to(teacher_logits.device)
            metrics = {key: value.tolist() for key, value in pair_metrics(teacher_logits, draft_logits).items()}
            draft_top = _top(draft_logits) if keep_top else None
            for index, row in enumerate(group["rows"]):
                out = {
                    "prompt_id": row["prompt_id"],
                    "domain": row["domain"],
                    "bank": row["bank"],
                    "producer": row.get("producer", row["bank"]),
                    "position": int(row["position"]),
                    "bin": position_bin(int(row["position"])),
                    "source_length": row.get("source_length"),
                    "draft": name,
                    "teacher_entropy": entropy[index],
                }
                for key in METRIC_KEYS:
                    out[key] = float(metrics[key][index])
                if keep_top:
                    out["teacher_top"] = [teacher_top[0][index], teacher_top[1][index]]
                    out["draft_top"] = [draft_top[0][index], draft_top[1][index]]
                scored.append(out)
    return scored


def _breakdown(rows: Sequence[dict]) -> dict:
    by_domain = defaultdict(list)
    by_bin = defaultdict(list)
    by_both = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
        by_bin[row["bin"]].append(row)
        by_both[f"{row['domain']}|{row['bin']}"].append(row)
    short = [row for row in rows if int(row["position"]) <= 64]
    long = [row for row in rows if 65 <= int(row["position"]) <= 192]
    return {
        "overall": aggregate_token_metrics(list(rows)),
        "positions_1_64": aggregate_token_metrics(short),
        "positions_65_192": aggregate_token_metrics(long),
        "by_domain": {domain: aggregate_token_metrics(by_domain.get(domain, [])) for domain in DOMAINS},
        "by_bin": {name: aggregate_token_metrics(by_bin.get(name, [])) for name, _lo, _hi in POSITION_BINS},
        "by_domain_and_bin": {
            f"{domain}|{name}": aggregate_token_metrics(by_both.get(f"{domain}|{name}", []))
            for domain in DOMAINS
            for name, _lo, _hi in POSITION_BINS
        },
    }


def prompt_cluster_bootstrap(rows_a: Sequence[dict], rows_b: Sequence[dict], metric: str, *, seed: int = 6304, n_resamples: int = 2000) -> dict:
    """Token-weighted difference a - b on matched (prompt, position) rows, resampling whole prompts."""
    left = {(row["prompt_id"], row["position"]): float(row[metric]) for row in rows_a}
    right = {(row["prompt_id"], row["position"]): float(row[metric]) for row in rows_b}
    keys = sorted(set(left) & set(right))
    if not keys:
        return {"n_tokens": 0}
    prompts = sorted({key[0] for key in keys})
    index = {prompt: i for i, prompt in enumerate(prompts)}
    sums = np.zeros(len(prompts))
    counts = np.zeros(len(prompts))
    for key in keys:
        sums[index[key[0]]] += left[key] - right[key]
        counts[index[key[0]]] += 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(prompts), size=(n_resamples, len(prompts)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean_diff": float(sums.sum() / counts.sum()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_tokens": int(counts.sum()),
        "n_prompts": len(prompts),
        "unmatched_tokens": len(set(left) ^ set(right)),
    }


def _method_seeds(names: Sequence[str]) -> dict[str, list[str]]:
    methods: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if "_seed" in name:
            methods[name.rsplit("_seed", 1)[0]].append(name)
    return dict(methods)


def summarize_scores(rows: Sequence[dict]) -> dict:
    """Per bank and draft breakdowns, seed dispersion, and prompt-clustered paired differences."""
    by_bank_draft: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_bank_draft[row["bank"]][row["draft"]].append(row)
    report: dict = {}
    for bank, per_draft in sorted(by_bank_draft.items()):
        drafts = sorted(per_draft)
        key_sets = {name: {(row["prompt_id"], row["position"]) for row in per_draft[name]} for name in drafts}
        reference = key_sets[drafts[0]]
        entry = {
            "producer": sorted({row["producer"] for rows_ in per_draft.values() for row in rows_}),
            "same_prefix_ids_for_every_draft": all(keys == reference for keys in key_sets.values()),
            "n_prompts": len({key[0] for key in reference}),
            "drafts": {name: _breakdown(per_draft[name]) for name in drafts},
            "seed_dispersion": {},
            "paired_differences": {},
        }
        for method, names in _method_seeds(drafts).items():
            entry["seed_dispersion"][method] = {
                metric: {
                    "per_seed": {name: entry["drafts"][name]["overall"][metric] for name in names},
                    "range": max(entry["drafts"][name]["overall"][metric] for name in names)
                    - min(entry["drafts"][name]["overall"][metric] for name in names),
                }
                for metric in METRIC_KEYS
            }
        pairs = []
        if "original" in per_draft:
            pairs += [(name, "original") for name in drafts if name != "original"]
        for name in drafts:
            if name.startswith("on_policy_seed"):
                other = "off_policy_seed" + name.rsplit("_seed", 1)[1]
                if other in per_draft:
                    pairs.append((name, other))
        for left, right in pairs:
            entry["paired_differences"][f"{left}_minus_{right}"] = {
                metric: prompt_cluster_bootstrap(per_draft[left], per_draft[right], metric)
                for metric in ("expected_acceptance", "forward_kl", "reverse_kl")
            }
        report[bank] = entry
    return report


def _prompt_means(rows: Sequence[dict], stat: str) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(stat)
        if value is not None:
            sums[row["prompt_id"]].append(float(value))
    return {prompt: float(np.mean(values)) for prompt, values in sums.items()}


def consistency_check(rows: Sequence[dict], *, bank_a: str = "teacher", bank_b: str = SPEC_BANK, draft: str = "original") -> dict:
    """Teacher-generated vs exact-speculative prefixes: both should follow the target-prefix law.

    For each statistic, the per-prompt mean is computed in each bank (positions
    differ between banks, prompts and sampling settings match), and the paired
    prompt bootstrap gives a 95% interval for bank_a - bank_b. Statistics: the
    target's next-token entropy, the original draft's forward KL and 1 - TV at
    the prefix, and the source response length (EOS before 192 tokens).
    Separate tables per domain and position bin; `consistent` is CI covers 0.
    """
    stats = ("teacher_entropy", "expected_acceptance", "forward_kl", "source_length")
    a_rows = [row for row in rows if row["bank"] == bank_a and row["draft"] == draft]
    b_rows = [row for row in rows if row["bank"] == bank_b and row["draft"] == draft]

    def compare(left: list[dict], right: list[dict]) -> dict:
        out = {}
        for stat in stats:
            if stat == "source_length":
                left_means = {row["prompt_id"]: float(row["source_length"]) for row in left if row.get("source_length") is not None}
                right_means = {row["prompt_id"]: float(row["source_length"]) for row in right if row.get("source_length") is not None}
            else:
                left_means = _prompt_means(left, stat)
                right_means = _prompt_means(right, stat)
            prompts = sorted(set(left_means) & set(right_means))
            if len(prompts) < 2:
                out[stat] = {"n_prompts": len(prompts)}
                continue
            boot = bootstrap_paired_mean_diff(
                np.asarray([left_means[p] for p in prompts]), np.asarray([right_means[p] for p in prompts]), seed=6304
            )
            boot["mean_a"] = float(np.mean([left_means[p] for p in prompts]))
            boot["mean_b"] = float(np.mean([right_means[p] for p in prompts]))
            boot["consistent"] = bool(boot["ci_low"] <= 0.0 <= boot["ci_high"])
            boot["n_prompts"] = len(prompts)
            chosen = set(prompts)
            boot["n_tokens_a"] = sum(1 for row in left if row["prompt_id"] in chosen)
            boot["n_tokens_b"] = sum(1 for row in right if row["prompt_id"] in chosen)
            out[stat] = boot
        return out

    report = {
        "bank_a": bank_a,
        "bank_b": bank_b,
        "draft_used_for_draft_statistics": draft,
        "overall": compare(a_rows, b_rows),
        "by_domain": {},
        "by_bin": {},
        "token_level": {
            metric: {
                bank_a: aggregate_token_metrics(a_rows).get(metric),
                bank_b: aggregate_token_metrics(b_rows).get(metric),
            }
            for metric in METRIC_KEYS
        },
    }
    for domain in DOMAINS:
        report["by_domain"][domain] = compare(
            [row for row in a_rows if row["domain"] == domain], [row for row in b_rows if row["domain"] == domain]
        )
    for name, _lo, _hi in POSITION_BINS:
        report["by_bin"][name] = compare(
            [row for row in a_rows if row["bin"] == name], [row for row in b_rows if row["bin"] == name]
        )
    flags = [
        value.get("consistent")
        for table in [report["overall"], *report["by_domain"].values(), *report["by_bin"].values()]
        for value in table.values()
        if "consistent" in value
    ]
    report["n_comparisons"] = len(flags)
    report["n_ci_excluding_zero"] = int(sum(1 for flag in flags if not flag))
    report["note"] = "About 5% of intervals may exclude 0 by chance; a systematic gap across tables suggests a sampler bug."
    return report


DISAGREEMENT_RULE = (
    "For each shared bank (teacher, original_draft, exact_spec_original_gamma4) and each domain, the "
    "{k} prefixes with the largest TV(target, original draft) among positions where the greedy tokens "
    "differ; ties broken by prompt_id then position. Chosen before inspecting any text."
)


def select_disagreements(rows: Sequence[dict], k: int = 2, banks: Sequence[str] = (*SUPPLIED_BANKS, SPEC_BANK)) -> list[tuple]:
    chosen = []
    for bank in banks:
        for domain in DOMAINS:
            candidates = [
                row
                for row in rows
                if row["bank"] == bank and row["domain"] == domain and row["draft"] == "original" and row["greedy_agreement"] == 0.0
            ]
            candidates.sort(key=lambda row: (-row["tv"], row["prompt_id"], row["position"]))
            chosen.extend((bank, row["prompt_id"], row["position"]) for row in candidates[:k])
    return chosen


def decode_disagreements(rows: Sequence[dict], bank_rows: Sequence[dict], tokenizer, k: int = 2, tail_tokens: int = 48) -> dict:
    """Decode the selected aligned disagreements: prefix tail, target and per-draft top tokens."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["bank"], row["prompt_id"], row["position"])].append(row)
    prefixes = {(row["bank"], row["prompt_id"], row["position"]): row for row in bank_rows}
    examples = []
    for key in select_disagreements(rows, k=k):
        source = prefixes.get(key)
        if source is None:
            continue
        scored = sorted(by_key[key], key=lambda row: row["draft"])
        teacher_ids, teacher_probs = scored[0]["teacher_top"]
        examples.append(
            {
                "bank": key[0],
                "prompt_id": key[1],
                "position": key[2],
                "domain": source["domain"],
                "prompt": source["prompt"],
                "prefix_tail_text": tokenizer.decode(list(source["response_prefix_token_ids"])[-tail_tokens:]),
                "target_top": [[tokenizer.decode([token]), prob] for token, prob in zip(teacher_ids, teacher_probs)],
                "drafts": {
                    row["draft"]: {
                        "top": [[tokenizer.decode([token]), prob] for token, prob in zip(*row["draft_top"])],
                        "tv": row["tv"],
                        "forward_kl": row["forward_kl"],
                        "greedy_agreement": row["greedy_agreement"],
                    }
                    for row in scored
                },
            }
        )
    return {"rule": DISAGREEMENT_RULE.format(k=k), "examples": examples}


def compact(rows: Sequence[dict]) -> list[dict]:
    """Drop the top-k lists before saving per-row scores."""
    return [{key: value for key, value in row.items() if key not in ("teacher_top", "draft_top")} for row in rows]


def bank_positions(response_len: int, prompt_id: str, bank: str) -> list[int]:
    """Same position budget and seeding rule as the course prefix-bank script, under this bank's name.

    Up to 8 + 8 + 16 = 32 positions in [1, 64] stratified by [1,16], [17,32],
    [33,64], plus up to 16 long positions in [65, 192].
    """
    rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    chosen: list[int] = []
    for low, high, count in ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)):
        candidates = list(range(low, min(high, response_len) + 1))
        chosen.extend(sorted(rng.sample(candidates, min(count, len(candidates)))))
    return chosen


def prefix_rows_from_generation(
    prompt_id: str,
    prompt: str,
    domain: str,
    token_ids: Sequence[int],
    bank: str,
    *,
    producer: str,
    generation_seed: int | None = None,
) -> list[dict]:
    rows = []
    for position in bank_positions(len(token_ids), prompt_id, bank):
        rows.append(
            {
                "prompt_id": prompt_id,
                "domain": domain_of(prompt_id, domain),
                "bank": bank,
                "position": position,
                "prompt": prompt,
                "response_prefix_token_ids": [int(token) for token in token_ids[: position - 1]],
                "producer": producer,
                "source_length": len(token_ids),
                "source_generation_seed": generation_seed,
            }
        )
    return rows


def exact_spec_bank(target, draft, tokenizer, examples, *, gamma: int = 4, max_new: int = 192) -> list[dict]:
    """Bank (iii): exact speculative sampling with the original draft, EOS-aware, 192-token cap."""
    from task3.src.speculative import SpeculativeDecoder

    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    rows = []
    for index, example in enumerate(examples):
        seed = 6304 + 10007 * index + 4444
        decoder = SpeculativeDecoder(target, draft, np.random.default_rng(seed))
        with torch.inference_mode():
            decoder.prefill(render_chat(tokenizer, example.text))
            generated = decoder.generate(gamma=gamma, max_new_tokens=max_new, eos_id=eos_id)
        rows.extend(
            prefix_rows_from_generation(
                example.source_id, example.text, example.domain, generated.tokens, SPEC_BANK,
                producer=f"original_draft_gamma{gamma}", generation_seed=seed,
            )
        )
    return rows


def student_bank(model, name: str, checkpoint: str, tokenizer, examples, *, seed: int, max_new: int = 192) -> list[dict]:
    """A trained student's own frozen rollouts, recording the producing checkpoint."""
    from task3.src.distill import sample_response

    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    rows = []
    for index, example in enumerate(examples):
        rollout_seed = seed + 10007 * index
        tokens, _reason = sample_response(
            model, render_chat(tokenizer, example.text), max_new_tokens=max_new, eos_id=eos_id,
            rng=np.random.default_rng(rollout_seed),
        )
        for row in prefix_rows_from_generation(
            example.source_id, example.text, example.domain, tokens, f"student_{name}",
            producer=checkpoint, generation_seed=rollout_seed,
        ):
            rows.append(row)
    return rows


def target_law_bank(target, tokenizer, examples, *, max_new: int = 192) -> list[dict]:
    """Matched-law control: plain temperature-1 target samples, EOS-aware, 192-token cap, teacher-bank seeds."""
    from task3.src.distill import sample_response

    eos_id = None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
    rows = []
    for index, example in enumerate(examples):
        seed = 6304 + 10007 * index
        tokens, _reason = sample_response(
            target, render_chat(tokenizer, example.text), max_new_tokens=max_new, eos_id=eos_id,
            rng=np.random.default_rng(seed),
        )
        rows.extend(
            prefix_rows_from_generation(
                example.source_id, example.text, example.domain, tokens, TARGET_LAW_BANK,
                producer="target_t1_no_repetition_penalty", generation_seed=seed,
            )
        )
    return rows
