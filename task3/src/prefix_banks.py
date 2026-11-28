"""The supplied teacher / original-draft prefix banks, rebuilt like the course script.

The course data repo (atml-pa4-datasets @ 6a0a9ec, scripts/setup_data.py,
`prefix_banks()`) samples one response per prompt and model with
`model.generate(ids, max_new_tokens=192, do_sample=True, temperature=1.0,
top_p=1.0, top_k=0, generator=gen, pad_token_id=eos)`. The pinned
transformers 4.57.1 rejects `generator=` ("model_kwargs are not used by the
model"), so the script cannot run. This module keeps everything else of that
function: pinned model revisions, the tokenizer-equality check, the prompt
list and order, the chat template, the per-prompt seeds
6304 + 10007 * j (+ 123 for the draft), the position RNG and bins, the record
fields and the output file. Only the sampling call is replaced by an explicit
KV-cached loop that reproduces what `generate` does with those arguments:

- the model's own generation_config is merged with the call's kwargs, exactly
  as `generate` merges them; for Qwen2.5-*-Instruct this keeps
  repetition_penalty = 1.1 and eos_token_id = [151645, 151643];
- next-token logits are cast to float32, the repetition penalty is applied
  over the prompt plus every generated token, temperature 1 / top-k 0 /
  top-p 1 leave the scores unchanged, and the token is
  `torch.multinomial(softmax(scores), 1, generator=gen)` with the CUDA
  generator the script seeds;
- sampling stops after an EOS token (kept in the sequence, as `generate`
  returns it) or after 192 tokens.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Callable, Sequence

import torch

from task3.src.runtime import revision_lock_path

COURSE_SCRIPT = "atml-pa4-datasets@6a0a9ec9da7ef3d313ee3036cfec493886723644 scripts/setup_data.py prefix_banks()"
BANK_FILE = "prefix_banks_locked.jsonl.gz"
META_FILE = "prefix_banks_build.json"
MAX_NEW_TOKENS = 192
BASE_SEED = 6304
SEED_STRIDE = 10007
# (bank name, revision-lock key, seed offset), in the order the course script generates them.
BANKS = (("teacher", "model_target", 0), ("original_draft", "model_draft", 123))
REPOS = {"model_target": "Qwen/Qwen2.5-1.5B-Instruct", "model_draft": "Qwen/Qwen2.5-0.5B-Instruct"}
TOKENIZER_PROBES = ("Example 123", "Hello there", "A complicated example with \U0001d70b")
POSITION_BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))
RECORD_FIELDS = (
    "domain",
    "position",
    "prompt",
    "prompt_id",
    "response_prefix_token_ids",
    "bank",
    "source_generation_seed",
    "source_length",
)
# generate() options that would add a logits processor or warper this sampler does not implement.
_UNSUPPORTED_DEFAULTS = {
    "guidance_scale": (None, 1.0),
    "sequence_bias": (None,),
    "encoder_repetition_penalty": (None, 1.0),
    "no_repeat_ngram_size": (None, 0),
    "encoder_no_repeat_ngram_size": (None, 0),
    "bad_words_ids": (None,),
    "min_length": (None, 0),
    "min_new_tokens": (None, 0),
    "forced_bos_token_id": (None,),
    "forced_eos_token_id": (None,),
    "remove_invalid_values": (None, False),
    "exponential_decay_length_penalty": (None,),
    "suppress_tokens": (None,),
    "begin_suppress_tokens": (None,),
    "min_p": (None,),
    "typical_p": (None, 1.0),
    "epsilon_cutoff": (None, 0.0),
    "eta_cutoff": (None, 0.0),
    "watermarking_config": (None,),
    "renormalize_logits": (None, False),
}
DEVIATION = (
    "The course script calls model.generate(..., generator=gen); the pinned transformers 4.57.1 rejects the "
    "generator kwarg (ValueError: model_kwargs are not used by the model: ['generator']). generate's internal "
    "sampler is replaced by an equivalent explicit KV-cached loop: float32 next-token logits, the model "
    "generation_config's logits processors that generate would apply with temperature=1, top_p=1, top_k=0 "
    "(here only the repetition penalty), softmax, torch.multinomial(probs, 1, generator=gen) with gen = "
    "torch.Generator('cuda').manual_seed(6304 + 10007*j + (0 or 123)), stop after any generation_config EOS id "
    "(kept) or 192 tokens. Prompts, chat template, seeds, positions and record fields are unchanged."
)


def load_revisions(data_root: Path) -> dict[str, str]:
    path = revision_lock_path(data_root)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; the pinned model revisions come from the course revision lock")
    revisions = json.loads(path.read_text(encoding="utf-8")).get("revisions", {})
    for key in REPOS:
        value = revisions.get(key)
        if not isinstance(value, str) or len(value) < 40:
            raise ValueError(f"{path} has no full commit hash for {key}: {value!r}")
    return revisions


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bank_prompts(data_root: Path) -> list[tuple[str, str, str]]:
    """(prompt_id, prompt, domain) in the course order: 60 GSM8K test questions, then short_test.jsonl.

    The course script reads gsm8k[row_index]['question'] for each row of
    data/public_manifests/gsm8k_test_ids.jsonl. data/public_locked/
    gsm8k_test_locked.jsonl.gz holds exactly those rows (order, source_row_index
    and the full example, written by the same script); its sha256 is checked
    here, and the manifest is cross-checked when it is present.
    """
    root = Path(data_root)
    locked = root / "data" / "public_locked" / "gsm8k_test_locked.jsonl.gz"
    if not locked.is_file():
        raise FileNotFoundError(f"missing {locked}")
    rows = _read_jsonl(locked)
    for row in rows:
        digest = hashlib.sha256(json.dumps(row["example"], sort_keys=True, default=str).encode("utf8")).hexdigest()
        if digest != row["source_sha256"]:
            raise ValueError(f"{locked}: order {row['order']} does not match its source_sha256")
    manifest = root / "data" / "public_manifests" / "gsm8k_test_ids.jsonl"
    if manifest.is_file():
        expected = [(int(item["order"]), int(item["row_index"])) for item in _read_jsonl(manifest)]
        found = [(int(row["order"]), int(row["source_row_index"])) for row in rows]
        if expected != found:
            raise ValueError(f"{locked} rows differ from the manifest {manifest}")
    prompts = [(f"GSM8K-{int(row['order']):04d}", row["example"]["question"], "math") for row in rows]
    short = root / "data" / "short_form" / "short_test.jsonl"
    if not short.is_file():
        raise FileNotFoundError(f"missing {short}")
    prompts.extend((item["id"], item["input"], "short") for item in _read_jsonl(short))
    return prompts


def generation_seed(index: int, bank: str) -> int:
    offset = dict((name, off) for name, _key, off in BANKS)[bank]
    return BASE_SEED + SEED_STRIDE * index + offset


def course_positions(sequence_length: int, prompt_id: str, bank: str) -> list[int]:
    """The course position draw, verbatim: one string-seeded RNG per (prompt, bank), four bins."""
    rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    positions: list[int] = []
    for lo, hi, n in POSITION_BINS:
        candidates = list(range(lo, min(hi, sequence_length) + 1))
        positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
    return positions


def course_records(prompt_id: str, prompt: str, domain: str, bank: str, sequence: Sequence[int], seed: int) -> list[dict]:
    seq = [int(token) for token in sequence]
    return [
        dict(
            prompt_id=prompt_id,
            domain=domain,
            bank=bank,
            position=position,
            prompt=prompt,
            response_prefix_token_ids=seq[: position - 1],
            source_length=len(seq),
            source_generation_seed=seed,
        )
        for position in course_positions(len(seq), prompt_id, bank)
    ]


def chat_ids(tokenizer, prompt: str) -> list[int]:
    """The course call, apply_chat_template(..., tokenize=True, add_generation_prompt=True), as a list."""
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def check_tokenizers(target_tokenizer, draft_tokenizer) -> None:
    for text in TOKENIZER_PROBES:
        if target_tokenizer.encode(text) != draft_tokenizer.encode(text):
            raise RuntimeError("Target/draft tokenizers differ")


def sampling_settings(model, pad_token_id: int | None) -> dict:
    """What generate() would use for this call: model generation_config updated with the call's kwargs."""
    config = copy.deepcopy(model.generation_config)
    config.update(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=pad_token_id, max_new_tokens=MAX_NEW_TOKENS)
    unsupported = {
        name: getattr(config, name, None)
        for name, allowed in _UNSUPPORTED_DEFAULTS.items()
        if getattr(config, name, None) not in allowed
    }
    if unsupported:
        raise RuntimeError(f"generation_config sets options this sampler does not reproduce: {unsupported}")
    eos = config.eos_token_id
    if eos is None:
        eos_ids: list[int] = []
    elif isinstance(eos, (list, tuple)):
        eos_ids = [int(token) for token in eos]
    else:
        eos_ids = [int(eos)]
    penalty = config.repetition_penalty
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0 if penalty is None else float(penalty),
        "eos_token_ids": eos_ids,
        "pad_token_id": pad_token_id,
    }


def penalized_scores(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """transformers' RepetitionPenaltyLogitsProcessor on one float32 row: seen tokens are pushed down."""
    if penalty == 1.0:
        return logits
    return torch.where(seen, torch.where(logits < 0, logits * penalty, logits / penalty), logits)


def _device(model) -> torch.device:
    return next(model.parameters()).device


def sample_like_generate(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    repetition_penalty: float,
    generator: torch.Generator | None = None,
    forced: Sequence[int] | None = None,
    trace: list | None = None,
) -> list[int]:
    """KV-cached ancestral sampling with generate()'s scoring. Returns the new tokens (EOS included).

    `forced` feeds the given tokens instead of sampling (used to compare the
    per-step distributions with generate's output_scores); `trace` collects the
    per-step sampling distributions on the CPU.
    """
    device = _device(model)
    stop = {int(token) for token in eos_token_ids}
    prompt = [int(token) for token in prompt_ids]
    if not prompt:
        raise ValueError("empty prompt")
    steps = max_new_tokens if forced is None else min(max_new_tokens, len(forced))
    tokens: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
        past = output.past_key_values
        logits = output.logits[0, -1].to(dtype=torch.float32, copy=True)
        seen = torch.zeros(logits.shape[-1], dtype=torch.bool, device=device)
        seen[torch.tensor(prompt, device=device)] = True
        length = len(prompt)
        for step in range(steps):
            probs = torch.softmax(penalized_scores(logits, seen, repetition_penalty), dim=-1)
            if trace is not None:
                trace.append(probs.detach().cpu())
            token = int(forced[step]) if forced is not None else int(torch.multinomial(probs, 1, generator=generator))
            tokens.append(token)
            if token in stop or step + 1 == steps:
                break
            seen[token] = True
            output = model(
                input_ids=torch.tensor([[token]], device=device),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([length], device=device),
                logits_to_keep=1,
            )
            past = output.past_key_values
            logits = output.logits[0, -1].to(dtype=torch.float32, copy=True)
            length += 1
    return tokens


def generate_equivalence(model, prompt_ids: Sequence[int], settings: dict, n_steps: int = 8) -> dict:
    """Run the real generate() (no generator) for a few steps, then replay its tokens through this sampler.

    Returns the largest per-step gap between softmax(generate's processed
    scores) and this sampler's distribution. A missing or extra logits
    processor shows up as a gap of order 0.1 or more.
    """
    device = _device(model)
    ids = torch.tensor([list(prompt_ids)], device=device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens=n_steps,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=settings["pad_token_id"],
            output_scores=True,
            return_dict_in_generate=True,
        )
    sequence = out.sequences[0, ids.shape[1] :].tolist()
    trace: list = []
    sample_like_generate(
        model, prompt_ids, max_new_tokens=len(sequence), eos_token_ids=settings["eos_token_ids"],
        repetition_penalty=settings["repetition_penalty"], forced=sequence, trace=trace,
    )
    gaps = [
        float((torch.softmax(score[0].float().cpu(), dim=-1) - mine).abs().max())
        for score, mine in zip(out.scores, trace)
    ]
    return {"n_steps": len(gaps), "max_abs_prob_gap": max(gaps) if gaps else None, "tokens": sequence}


def generate_bank(
    model,
    tokenizer,
    prompts: Sequence[tuple[int, str, str, str]],
    bank: str,
    settings: dict,
    *,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """Records for one bank. `prompts` holds (index j in the full course list, prompt_id, prompt, domain)."""
    device = _device(model)
    records: list[dict] = []
    for count, (index, prompt_id, prompt, domain) in enumerate(prompts, start=1):
        seed = generation_seed(index, bank)
        generator = torch.Generator(device=device).manual_seed(seed)
        sequence = sample_like_generate(
            model,
            chat_ids(tokenizer, prompt),
            max_new_tokens=settings["max_new_tokens"],
            eos_token_ids=settings["eos_token_ids"],
            repetition_penalty=settings["repetition_penalty"],
            generator=generator,
        )
        records.extend(course_records(prompt_id, prompt, domain, bank, sequence, seed))
        if count % 20 == 0:
            progress(f"{bank} {count} / {len(prompts)}")
    return records


def write_bank(path: Path, records: Sequence[dict]) -> None:
    """The course save_lines format (sort_keys, ensure_ascii=False, gzip), written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def bank_summary(records: Sequence[dict]) -> dict:
    out: dict = {}
    for bank, _key, _offset in BANKS:
        rows = [row for row in records if row["bank"] == bank]
        lengths = {row["prompt_id"]: int(row["source_length"]) for row in rows}
        out[bank] = {
            "n_records": len(rows),
            "n_prompts": len(lengths),
            "mean_source_length": sum(lengths.values()) / len(lengths) if lengths else None,
            "n_at_192_tokens": sum(1 for value in lengths.values() if value >= MAX_NEW_TOKENS),
        }
    return out


def prompts_digest(prompts: Sequence[tuple]) -> str:
    return hashlib.sha256(json.dumps([list(item) for item in prompts], ensure_ascii=False).encode("utf8")).hexdigest()
