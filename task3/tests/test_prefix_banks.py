"""The prefix-bank rebuild (task3/src/prefix_banks.py) on CPU with tiny random Qwen2 models.

Checks that the explicit KV-cached sampler draws each token from the same
distribution as a full no-cache forward and as transformers' generate() with
the course arguments (repetition penalty from the generation config included),
that positions, seeds and records equal the course function's, and that the
written file is what alignment.load_prefix_bank reads. No weights are downloaded.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from task3.src import prefix_banks as pb
from task3.src.alignment import load_prefix_bank
from task3.tests.test_alignment import FakeTokenizer
from task3.tests.test_cached_decoder import tiny_qwen

PROMPT = [1, 4, 9, 9, 17, 30, 9, 3]


def course_snippet(seq, pid, prompt, domain, which, seed):
    """The record loop of setup_data.py prefix_banks(), verbatim apart from the arguments."""
    output = []
    rng = random.Random(f'PA4-banks-v1:6304:{pid}:{which}')
    positions = []
    for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
        candidates = list(range(lo, min(hi, len(seq)) + 1))
        positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
    for pos in positions:
        output.append(dict(prompt_id=pid, domain=domain, bank=which, position=pos, prompt=prompt,
                           response_prefix_token_ids=seq[:pos - 1], source_length=len(seq), source_generation_seed=seed))
    return output


def qwen_like(seed: int, penalty: float = 1.1):
    model = tiny_qwen(seed, sharpness=3.0)
    model.generation_config.eos_token_id = [2, 47]
    model.generation_config.repetition_penalty = penalty
    return model


def full_forward_probs(model, prompt, tokens, penalty):
    """Reference: one no-cache forward per step, then generate()'s processing."""
    rows = []
    for step in range(len(tokens)):
        ids = list(prompt) + list(tokens[:step])
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float()
        seen = torch.zeros(logits.shape[-1], dtype=torch.bool)
        seen[torch.tensor(ids)] = True
        rows.append(torch.softmax(pb.penalized_scores(logits, seen, penalty), dim=-1))
    return rows


class PrefixBankTests(unittest.TestCase):
    def test_positions_records_and_seeds_match_course(self) -> None:
        rng = random.Random(0)
        for length in (0, 1, 5, 16, 17, 33, 64, 65, 120, 192):
            seq = [rng.randrange(5, 40) for _ in range(length)]
            for pid in ("GSM8K-0000", "SHORT-shifted-0029"):
                for which in ("teacher", "original_draft"):
                    self.assertEqual(
                        pb.course_records(pid, "p", "math", which, seq, 11),
                        course_snippet(seq, pid, "p", "math", which, 11),
                    )
        self.assertEqual(pb.generation_seed(0, "teacher"), 6304)
        self.assertEqual(pb.generation_seed(3, "original_draft"), 6304 + 10007 * 3 + 123)
        self.assertEqual(len(pb.course_positions(192, "GSM8K-0001", "teacher")), 48)

    def test_cached_sampler_matches_full_forward(self) -> None:
        for penalty in (1.1, 1.0):
            model = qwen_like(51, penalty)
            trace: list = []
            generator = torch.Generator(device="cpu").manual_seed(7)
            tokens = pb.sample_like_generate(
                model, PROMPT, max_new_tokens=40, eos_token_ids=[], repetition_penalty=penalty, generator=generator, trace=trace
            )
            self.assertEqual(len(tokens), 40)
            reference = full_forward_probs(model, PROMPT, tokens, penalty)
            for mine, ref in zip(trace, reference):
                self.assertLess(float((mine - ref).abs().max()), 1e-5)

    def test_matches_generate_processed_scores(self) -> None:
        model = qwen_like(52)
        settings = pb.sampling_settings(model, pad_token_id=2)
        self.assertEqual(settings["repetition_penalty"], 1.1)
        self.assertEqual(settings["eos_token_ids"], [2, 47])
        self.assertEqual((settings["temperature"], settings["top_p"], settings["top_k"]), (1.0, 1.0, 0))
        torch.manual_seed(3)
        check = pb.generate_equivalence(model, PROMPT, settings, n_steps=12)
        self.assertGreater(check["n_steps"], 0)
        self.assertLess(check["max_abs_prob_gap"], 1e-5)
        # The check notices a sampler that drops the repetition penalty.
        torch.manual_seed(3)
        wrong = pb.generate_equivalence(model, PROMPT, dict(settings, repetition_penalty=1.0), n_steps=12)
        self.assertGreater(wrong["max_abs_prob_gap"], 1e-3)

    def test_eos_kept_cap_and_seeding(self) -> None:
        model = qwen_like(53)
        kept = pb.sample_like_generate(model, PROMPT, max_new_tokens=10, eos_token_ids=[2, 47], repetition_penalty=1.1, forced=[5, 6, 47, 7])
        self.assertEqual(kept, [5, 6, 47])
        capped = pb.sample_like_generate(model, PROMPT, max_new_tokens=3, eos_token_ids=[], repetition_penalty=1.1, generator=torch.Generator().manual_seed(1))
        self.assertEqual(len(capped), 3)

        def draw(seed):
            return pb.sample_like_generate(model, PROMPT, max_new_tokens=20, eos_token_ids=[], repetition_penalty=1.1, generator=torch.Generator().manual_seed(seed))

        self.assertEqual(draw(9), draw(9))
        self.assertNotEqual(draw(9), draw(10))

    def test_unsupported_generation_option_fails_fast(self) -> None:
        model = qwen_like(54)
        model.generation_config.no_repeat_ngram_size = 3
        with self.assertRaises(RuntimeError):
            pb.sampling_settings(model, pad_token_id=2)

    def test_prompt_list_and_bank_file(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        write_course_data(root, n_math=2, n_familiar=1, n_shifted=1)
        prompts = pb.bank_prompts(root)
        self.assertEqual([pid for pid, _text, _domain in prompts], ["GSM8K-0000", "GSM8K-0001", "SHORT-familiar-0000", "SHORT-shifted-0000"])
        self.assertEqual([domain for _pid, _text, domain in prompts], ["math", "math", "short", "short"])

        tokenizer = FakeTokenizer()
        indexed = [(index, pid, text, domain) for index, (pid, text, domain) in enumerate(prompts)]
        records = []
        for bank, _key, _offset in pb.BANKS:
            model = qwen_like(55 if bank == "teacher" else 56)
            settings = pb.sampling_settings(model, pad_token_id=2)
            records += pb.generate_bank(model, tokenizer, indexed, bank, dict(settings, max_new_tokens=70), progress=lambda _text: None)
        path = root / "out" / pb.BANK_FILE
        pb.write_bank(path, records)
        with gzip.open(path, "rt", encoding="utf8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), len(records))
        for line in lines:
            record = json.loads(line)
            self.assertEqual(set(record), set(pb.RECORD_FIELDS))
            self.assertEqual(list(record), sorted(record))
        rows = load_prefix_bank(path)
        self.assertEqual({row["bank"] for row in rows}, {"teacher", "original_draft"})
        self.assertEqual({row["domain"] for row in rows}, {"math", "familiar", "shifted"})
        seeds = {(row["bank"], row["prompt_id"]): row["source_generation_seed"] for row in rows}
        self.assertEqual(seeds[("original_draft", "SHORT-familiar-0000")], 6304 + 10007 * 2 + 123)
        summary = pb.bank_summary(records)
        self.assertEqual(summary["teacher"]["n_prompts"], 4)

        locked = root / "data" / "public_locked" / "gsm8k_test_locked.jsonl.gz"
        with gzip.open(locked, "rt", encoding="utf8") as handle:
            first = [json.loads(line) for line in handle]
        first[0]["example"]["question"] = "tampered"
        with gzip.open(locked, "wt", encoding="utf8") as handle:
            for row in first:
                handle.write(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            pb.bank_prompts(root)


def write_course_data(root: Path, *, n_math: int, n_familiar: int, n_shifted: int, n_train_math: int = 0, n_train_short: int = 0) -> None:
    """A miniature course data repo in the real file layout (revision lock, locked GSM8K, short form)."""
    data = root / "data"
    (data / "public_locked").mkdir(parents=True, exist_ok=True)
    (data / "short_form").mkdir(parents=True, exist_ok=True)
    (data / "public_manifests").mkdir(parents=True, exist_ok=True)
    (data / "local_source_revisions.json").write_text(
        json.dumps(
            {
                "repos": {"model_target": "Qwen/Qwen2.5-1.5B-Instruct", "model_draft": "Qwen/Qwen2.5-0.5B-Instruct", "gsm8k": "openai/gsm8k"},
                "revisions": {"model_target": "a" * 40, "model_draft": "b" * 40, "gsm8k": "c" * 40},
            }
        ),
        encoding="utf-8",
    )

    def gsm_rows(n: int, tag: str):
        rows = []
        for order in range(n):
            example = {"question": f"{tag} question {order}: Natalia sold {order + 3} clips, then half as many.", "answer": f"steps\n#### {order + 4}"}
            digest = hashlib.sha256(json.dumps(example, sort_keys=True, default=str).encode("utf8")).hexdigest()
            rows.append({"order": order, "source_row_index": 10 + order, "example": example, "source_sha256": digest})
        return rows

    for name, rows in (("gsm8k_test_locked", gsm_rows(n_math, "test")), ("gsm8k_train_locked", gsm_rows(n_train_math, "train"))):
        with gzip.open(data / "public_locked" / f"{name}.jsonl.gz", "wt", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    with (data / "public_manifests" / "gsm8k_test_ids.jsonl").open("w", encoding="utf8") as handle:
        for order in range(n_math):
            handle.write(json.dumps({"order": order, "row_index": 10 + order}) + "\n")
    short_test = [
        {"id": f"SHORT-{split}-{i:04d}", "input": f"Read this report date: May {i + 1}, 2020 ({split}). Return only the month.", "answer": "May", "family": i, "split": split}
        for split, count in (("familiar", n_familiar), ("shifted", n_shifted))
        for i in range(count)
    ]
    short_train = [
        {"id": f"SHORT-train-{i:04d}", "input": f"Shipping label: destination code PK, serial {i}. Output the code.", "answer": "PK", "family": i, "split": "train"}
        for i in range(n_train_short)
    ]
    for name, rows in (("short_test", short_test), ("short_train", short_train)):
        (data / "short_form" / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


if __name__ == "__main__":
    unittest.main()
