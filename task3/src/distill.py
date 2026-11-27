"""Forward-KL LoRA distillation of the 0.5B draft toward the 1.5B teacher.

Off-policy and on-policy runs both minimize KL(teacher || student) on response
prefixes. The sampled token is not a hard label. On-policy rollouts are drawn
from the current student and are not differentiated. A 40,000 teacher-scored
token quota (30,000 math and 10,000 short) is shared by the accounting, with
one optimizer step per 256 scored tokens and a short final step allowed.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from shared.paths import course_data_root
from shared.seed import set_seed
from task3.src.speculative import KVCacheState, logits_to_probs, sample_categorical

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_SEEDS = (6304, 6305)
METHODS = ("off_policy", "on_policy")
LOSS_NAME = "forward_kl"

MATH_QUOTA = 30_000
SHORT_QUOTA = 10_000
SCORED_TOKEN_QUOTA = MATH_QUOTA + SHORT_QUOTA
MATH_RESPONSE_CAP = 192
SHORT_RESPONSE_CAP = 32
STEP_TOKENS = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
TEMPERATURE = 1.0
TOP_P = 1.0

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SHORT_SCHEDULE_SALT = 7919

TOKEN_CHECK_PROMPTS = (
    "Example 123",
    "Hello there",
    "A complicated example with π",
    "What is 2+2?",
    "Return only the month.",
)


def loss_for_method(method: str) -> str:
    """Both prefix sources use forward KL of the teacher against the student."""
    if method not in METHODS:
        raise ValueError(f"unknown distillation method {method!r}")
    return LOSS_NAME


def lora_trainable_count(shapes: Sequence[tuple[int, int]], rank: int = LORA_R) -> int:
    """Trainable A/B parameters for frozen linear maps of sizes (in, out)."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    return sum(rank * in_features + out_features * rank for in_features, out_features in shapes)


class LoRALinearStandIn(nn.Module):
    """Frozen linear layer plus a rank-r adapter. B starts at zero, so the map is unchanged."""

    def __init__(self, in_features: int, out_features: int, rank: int = LORA_R, alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("this assignment uses LoRA dropout 0")
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = inputs @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return self.base(inputs) + adapted * self.scaling


def build_lora_config():
    """PEFT config for the student. Importing PEFT does not download Qwen."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def per_token_forward_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) summed over the vocabulary, one value per token.

    Teacher probabilities are detached. The sampled token is not a target id.
    """
    teacher_log = torch.log_softmax(teacher_logits.float(), dim=-1).detach()
    student_log = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_prob = teacher_log.exp()
    return (teacher_prob * (teacher_log - student_log)).sum(dim=-1)


def response_token_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    prompt_len: int,
    n_tokens: int,
) -> torch.Tensor:
    """KL at each response position. Logits at index prompt_len - 1 predict the first response token."""
    if prompt_len < 1:
        raise ValueError("the chat prompt must contain at least one token")
    if n_tokens < 1:
        raise ValueError("at least one response token is required")
    start = prompt_len - 1
    teacher_slice = teacher_logits[:, start : start + n_tokens, :]
    student_slice = student_logits[:, start : start + n_tokens, :]
    if teacher_slice.shape[1] != n_tokens or student_slice.shape[1] != n_tokens:
        raise RuntimeError("logits do not cover the scored response prefix")
    kl = per_token_forward_kl(teacher_slice, student_slice)
    if kl.ndim == 2:
        kl = kl.squeeze(0)
    return kl


def shuffled_order(n_items: int, seed: int) -> list[int]:
    if n_items < 1:
        raise ValueError("round-robin needs at least one prompt")
    order = list(range(n_items))
    random.Random(seed).shuffle(order)
    return order


def cycle_indices(n_items: int, n_draws: int, seed: int) -> list[int]:
    """Seeded permutation, then reuse from the start of that same permutation."""
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    order = shuffled_order(n_items, seed)
    return [order[draw % n_items] for draw in range(n_draws)]


def presentations_for_quota(lengths: Sequence[int], quota: int) -> list[dict]:
    """Score response prefixes in order, cycling the pool until `quota` tokens are reached.

    The last presentation is truncated when the quota ends inside that response.
    """
    if quota < 0:
        raise ValueError("quota must be non-negative")
    lengths = [int(length) for length in lengths]
    if quota == 0:
        return []
    if not lengths:
        raise ValueError("cannot fill a token quota from an empty pool")
    presentations: list[dict] = []
    scored = 0
    cursor = 0
    idle = 0
    while scored < quota:
        index = cursor % len(lengths)
        available = lengths[index]
        cursor += 1
        if available <= 0:
            idle += 1
            if idle > len(lengths):
                raise RuntimeError("every pooled response is empty")
            continue
        idle = 0
        take = min(available, quota - scored)
        presentations.append(
            {
                "index": index,
                "generated_tokens": available,
                "scored_tokens": take,
                "truncated": take < available,
            }
        )
        scored += take
    return presentations


def optimizer_update_sizes(scored_counts: Sequence[int], step_tokens: int = STEP_TOKENS) -> list[int]:
    """Greedy token groups. Full groups have `step_tokens`; the last group may be shorter."""
    if step_tokens < 1:
        raise ValueError("step size must be positive")
    pending = 0
    groups: list[int] = []
    for count in scored_counts:
        left = int(count)
        if left < 0:
            raise ValueError("scored token counts must be non-negative")
        while left:
            room = step_tokens - pending
            take = min(room, left)
            pending += take
            left -= take
            if pending == step_tokens:
                groups.append(step_tokens)
                pending = 0
    if pending:
        groups.append(pending)
    return groups


@dataclass
class PromptExample:
    source_id: str
    text: str
    domain: str
    answer: str
    family: int | None = None

    @property
    def category(self) -> str:
        return "math" if self.domain == "math" else "short"


def load_jsonl(path: Path) -> list[dict]:
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_gsm8k(path: Path, id_prefix: str) -> list[PromptExample]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Prepare it with the course data repo's setup_data.py --public; this module does not download it."
        )
    examples = []
    for row in load_jsonl(path):
        example = row["example"]
        examples.append(
            PromptExample(
                source_id=f"{id_prefix}{int(row['order']):04d}",
                text=example["question"],
                domain="math",
                answer=example["answer"],
            )
        )
    return examples


def load_short_form(path: Path) -> list[PromptExample]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Clone the pinned course data repo; this module does not download it."
        )
    examples = []
    for row in load_jsonl(path):
        split = str(row.get("split", "train"))
        if split == "familiar":
            domain = "familiar"
        elif split == "shifted":
            domain = "shifted"
        else:
            domain = "short"
        family = row.get("family")
        examples.append(
            PromptExample(
                source_id=str(row["id"]),
                text=row["input"],
                domain=domain,
                answer=str(row["answer"]),
                family=None if family is None else int(family),
            )
        )
    return examples


def default_data_paths(root: Path | None = None) -> dict[str, Path]:
    base = course_data_root() if root is None else Path(root)
    return {
        "gsm8k_train": base / "data" / "public_locked" / "gsm8k_train_locked.jsonl.gz",
        "gsm8k_test": base / "data" / "public_locked" / "gsm8k_test_locked.jsonl.gz",
        "short_train": base / "data" / "short_form" / "short_train.jsonl",
        "short_test": base / "data" / "short_form" / "short_test.jsonl",
        "prefix_banks": base / "data" / "prefix_banks" / "prefix_banks_locked.jsonl.gz",
    }


def load_training_pool(root: Path | None = None) -> tuple[list[PromptExample], list[PromptExample]]:
    paths = default_data_paths(root)
    math_examples = load_gsm8k(paths["gsm8k_train"], "GSM8K-train-")
    short_examples = load_short_form(paths["short_train"])
    return math_examples, short_examples


def load_eval_pool(root: Path | None = None) -> list[PromptExample]:
    paths = default_data_paths(root)
    math_examples = load_gsm8k(paths["gsm8k_test"], "GSM8K-")
    short_examples = load_short_form(paths["short_test"])
    return math_examples + short_examples


def assert_no_eval_leak(train: Iterable[PromptExample], evaluation: Iterable[PromptExample]) -> None:
    overlap = {item.source_id for item in train} & {item.source_id for item in evaluation}
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(f"evaluation ids are in the training pool: {sample}")


def render_chat(tokenizer, user_text: str) -> list[int]:
    """One non-thinking chat template for every condition."""
    messages = [{"role": "user", "content": user_text}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def assert_same_token_ids(first_tokenizer, second_tokenizer, prompts: Sequence[str] = TOKEN_CHECK_PROMPTS) -> None:
    for prompt in prompts:
        left = render_chat(first_tokenizer, prompt)
        right = render_chat(second_tokenizer, prompt)
        if left != right:
            raise RuntimeError(f"target and draft token ids differ for prompt {prompt!r}")


def category_seed(train_seed: int, category: str) -> int:
    if category == "math":
        return train_seed
    if category == "short":
        return train_seed + SHORT_SCHEDULE_SALT
    raise ValueError(category)


def rollout_seed(train_seed: int, category: str, draw: int) -> int:
    salt = 0 if category == "math" else 17
    return int(train_seed + 10007 * draw + salt)


@dataclass
class RolloutRecord:
    source_id: str
    rollout_seed: int
    token_ids: list[int]
    termination_reason: str
    teacher_scored_tokens: int
    category: str
    method: str
    train_seed: int
    response_cap: int
    prompt_token_ids: list[int] = field(default_factory=list)

    def scored_token_ids(self) -> list[int]:
        return self.token_ids[: self.teacher_scored_tokens]


def sample_response(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_id: int | None,
    rng,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> tuple[list[int], str]:
    """Autoregressive sample. The caller must not differentiate through the returned ids."""
    if max_new_tokens < 1:
        return [], "truncated_quota"
    state = KVCacheState(model)
    with torch.inference_mode():
        state.prefill(prompt_ids)
        tokens: list[int] = []
        reason = "length"
        for _ in range(max_new_tokens):
            probs = logits_to_probs(state.next_logits, temperature, top_p)
            token = sample_categorical(probs, rng)
            tokens.append(token)
            state.forward_tokens([token])
            if eos_id is not None and token == eos_id:
                reason = "eos"
                break
    return tokens, reason


def parameter_report(model) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _require_cuda(action: str) -> None:
    if not torch.cuda.is_available():
        raise SystemExit(f"{action} requires CUDA and will not download weights on CPU.")


def load_frozen_causal(model_name: str, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def load_lora_student(seed: int):
    """Fresh FP16 0.5B backbone with an FP32 LoRA initialized from `seed`."""
    from peft import get_peft_model

    _require_cuda("loading the LoRA student")
    model, tokenizer = load_frozen_causal(DRAFT_MODEL)
    set_seed(seed)
    model = get_peft_model(model, build_lora_config())
    trainable = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("LoRA attached no trainable parameters")
    return model, tokenizer, trainable


def response_kl_tensor(teacher, student, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> torch.Tensor:
    if not response_ids:
        raise ValueError("response prefix is empty")
    ids = list(prompt_ids) + list(response_ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        student_logits = student(input_ids=input_ids, use_cache=False).logits
    return response_token_kl(teacher_logits, student_logits, len(prompt_ids), len(response_ids))


def scaled_optimizer_step(scaler, optimizer, parameters: Sequence[torch.nn.Parameter], n_tokens: int) -> bool:
    """Unscale, divide by the supervised tokens, clip, then step.

    Returns True when GradScaler skips the step because of non-finite gradients.
    """
    if n_tokens < 1:
        raise ValueError("refusing an empty optimizer step")
    scaler.unscale_(optimizer)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(n_tokens)
    torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(scaler.get_scale()) < scale_before


@dataclass
class UpdateAccumulator:
    step_tokens: int = STEP_TOKENS
    pending_count: int = 0
    pending_value: float = 0.0
    updates: list[dict] = field(default_factory=list)
    skipped_steps: int = 0

    def add_kl(self, kl: torch.Tensor, scaler, optimizer, parameters) -> None:
        offset = 0
        total = int(kl.shape[0])
        while offset < total:
            room = self.step_tokens - self.pending_count
            take = min(room, total - offset)
            chunk = kl[offset : offset + take]
            scaler.scale(chunk.sum()).backward(retain_graph=offset + take < total)
            self.pending_count += take
            self.pending_value += float(chunk.detach().sum().cpu())
            offset += take
            if self.pending_count == self.step_tokens:
                self._flush(scaler, optimizer, parameters)

    def flush(self, scaler, optimizer, parameters) -> None:
        if self.pending_count:
            self._flush(scaler, optimizer, parameters)

    def _flush(self, scaler, optimizer, parameters) -> None:
        n_tokens = self.pending_count
        loss = self.pending_value / n_tokens
        skipped = scaled_optimizer_step(scaler, optimizer, parameters, n_tokens)
        self.skipped_steps += int(skipped)
        self.updates.append(
            {
                "step": len(self.updates) + 1,
                "loss": loss,
                "learning_rate": LEARNING_RATE,
                "tokens_in_update": n_tokens,
                "short_update": n_tokens != self.step_tokens,
                "skipped": skipped,
                "wall_time_s": time.perf_counter(),
            }
        )
        self.pending_count = 0
        self.pending_value = 0.0


def _category_cap(category: str) -> int:
    if category == "math":
        return MATH_RESPONSE_CAP
    if category == "short":
        return SHORT_RESPONSE_CAP
    raise ValueError(category)


def _category_quota(category: str) -> int:
    if category == "math":
        return MATH_QUOTA
    if category == "short":
        return SHORT_QUOTA
    raise ValueError(category)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _generate_category(
    model,
    tokenizer,
    examples: Sequence[PromptExample],
    *,
    category: str,
    train_seed: int,
    method: str,
    quota: int,
    eos_id: int | None,
    stop_at_quota: bool,
) -> list[RolloutRecord]:
    """Round-robin generation until the category quota is covered.

    Off-policy generation uses the response cap and may overshoot; scoring truncates.
    On-policy generation stops at the remaining quota so unscored tokens are not drawn.
    """
    order = shuffled_order(len(examples), category_seed(train_seed, category))
    cap = _category_cap(category)
    records: list[RolloutRecord] = []
    covered = 0
    draw = 0
    while covered < quota:
        example = examples[order[draw % len(order)]]
        seed = rollout_seed(train_seed, category, draw)
        remaining = quota - covered
        max_new = min(cap, remaining) if stop_at_quota else cap
        prompt_ids = render_chat(tokenizer, example.text)
        tokens, reason = sample_response(
            model,
            prompt_ids,
            max_new_tokens=max_new,
            eos_id=eos_id,
            rng=np.random.default_rng(seed),
        )
        scored = min(len(tokens), remaining)
        if scored < len(tokens) or (reason == "length" and max_new < cap):
            reason = "truncated_quota"
        records.append(
            RolloutRecord(
                source_id=example.source_id,
                rollout_seed=seed,
                token_ids=tokens,
                termination_reason=reason,
                teacher_scored_tokens=scored,
                category=category,
                method=method,
                train_seed=train_seed,
                response_cap=cap,
                prompt_token_ids=prompt_ids,
            )
        )
        covered += scored
        draw += 1
        if draw > quota + len(examples):
            raise RuntimeError(f"{category} generation did not reach the token quota")
    planned = presentations_for_quota([len(record.token_ids) for record in records], quota)
    if len(planned) != len(records) or sum(item["scored_tokens"] for item in planned) != quota:
        raise RuntimeError("quota planner did not consume the category budget")
    for record, item in zip(records, planned):
        record.teacher_scored_tokens = int(item["scored_tokens"])
        if item["truncated"]:
            record.termination_reason = "truncated_quota"
    return records


def _on_policy_category(
    student,
    tokenizer,
    teacher,
    examples: Sequence[PromptExample],
    *,
    category: str,
    train_seed: int,
    quota: int,
    eos_id: int | None,
    accumulator: UpdateAccumulator,
    scaler,
    optimizer,
    parameters,
) -> list[RolloutRecord]:
    """Sample from the current student, score with the teacher, then continue."""
    records: list[RolloutRecord] = []
    covered = 0
    draw = 0
    cap = _category_cap(category)
    order = shuffled_order(len(examples), category_seed(train_seed, category))
    while covered < quota:
        example = examples[order[draw % len(order)]]
        seed = rollout_seed(train_seed, category, draw)
        remaining = quota - covered
        max_new = min(cap, remaining)
        prompt_ids = render_chat(tokenizer, example.text)
        student.eval()
        tokens, reason = sample_response(
            student,
            prompt_ids,
            max_new_tokens=max_new,
            eos_id=eos_id,
            rng=np.random.default_rng(seed),
        )
        scored = min(len(tokens), remaining)
        if scored < len(tokens) or (reason == "length" and max_new < cap):
            reason = "truncated_quota"
        record = RolloutRecord(
            source_id=example.source_id,
            rollout_seed=seed,
            token_ids=tokens,
            termination_reason=reason,
            teacher_scored_tokens=scored,
            category=category,
            method="on_policy",
            train_seed=train_seed,
            response_cap=cap,
            prompt_token_ids=prompt_ids,
        )
        records.append(record)
        _score_records(teacher, student, [record], accumulator, scaler, optimizer, parameters)
        covered += scored
        draw += 1
        if draw > quota + len(examples):
            raise RuntimeError(f"{category} on-policy generation did not reach the token quota")
    return records


def _score_records(teacher, student, records: Sequence[RolloutRecord], accumulator: UpdateAccumulator, scaler, optimizer, parameters) -> None:
    student.train()
    for record in records:
        scored_ids = record.scored_token_ids()
        if not scored_ids:
            continue
        offset = 0
        while offset < len(scored_ids):
            room = accumulator.step_tokens - accumulator.pending_count
            take = min(room, len(scored_ids) - offset)
            # A fresh forward of the scored prefix. Only the newly included tail
            # enters this microbatch, so a response can cross a 256-token boundary.
            prefix = scored_ids[: offset + take]
            kl = response_kl_tensor(teacher, student, record.prompt_token_ids, prefix)[offset:]
            accumulator.add_kl(kl, scaler, optimizer, parameters)
            offset += take


def train_method(
    method: str,
    seed: int,
    output_dir: Path,
    *,
    data_root: Path | None = None,
) -> dict:
    """Train one seed. Refuses to run, and does not download weights, without CUDA."""
    if loss_for_method(method) != LOSS_NAME:
        raise RuntimeError("on-policy training must stay on forward KL")
    _require_cuda(f"distill {method} seed {seed}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    math_examples, short_examples = load_training_pool(data_root)
    evaluation = load_eval_pool(data_root)
    assert_no_eval_leak([*math_examples, *short_examples], evaluation)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    teacher, teacher_tokenizer = load_frozen_causal(TARGET_MODEL)
    student, student_tokenizer, trainable = load_lora_student(seed)
    assert_same_token_ids(teacher_tokenizer, student_tokenizer)
    counts = parameter_report(student)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    eos_id = None if teacher_tokenizer.eos_token_id is None else int(teacher_tokenizer.eos_token_id)
    accumulator = UpdateAccumulator()

    optimizer.zero_grad(set_to_none=True)
    teacher.eval()
    generation_started = time.perf_counter()
    scoring_s = 0.0
    if method == "off_policy":
        math_records = _generate_category(
            teacher,
            teacher_tokenizer,
            math_examples,
            category="math",
            train_seed=seed,
            method=method,
            quota=MATH_QUOTA,
            eos_id=eos_id,
            stop_at_quota=False,
        )
        short_records = _generate_category(
            teacher,
            teacher_tokenizer,
            short_examples,
            category="short",
            train_seed=seed,
            method=method,
            quota=SHORT_QUOTA,
            eos_id=eos_id,
            stop_at_quota=False,
        )
        generation_s = time.perf_counter() - generation_started
        scoring_started = time.perf_counter()
        _score_records(teacher, student, math_records + short_records, accumulator, scaler, optimizer, trainable)
        accumulator.flush(scaler, optimizer, trainable)
        scoring_s = time.perf_counter() - scoring_started
    else:
        # Generate from the student that will actually be updated, then score
        # that rollout before drawing the next one. Stepping inside the rollout
        # does not rewrite tokens that were already sampled.
        math_records = _on_policy_category(
            student,
            student_tokenizer,
            teacher,
            math_examples,
            category="math",
            train_seed=seed,
            quota=MATH_QUOTA,
            eos_id=eos_id,
            accumulator=accumulator,
            scaler=scaler,
            optimizer=optimizer,
            parameters=trainable,
        )
        short_records = _on_policy_category(
            student,
            student_tokenizer,
            teacher,
            short_examples,
            category="short",
            train_seed=seed,
            quota=SHORT_QUOTA,
            eos_id=eos_id,
            accumulator=accumulator,
            scaler=scaler,
            optimizer=optimizer,
            parameters=trainable,
        )
        accumulator.flush(scaler, optimizer, trainable)
        generation_s = time.perf_counter() - generation_started
    records = math_records + short_records

    adapter_dir = output_dir / "adapter"
    student.save_pretrained(adapter_dir)
    student_tokenizer.save_pretrained(adapter_dir)
    _write_jsonl(output_dir / "rollouts.jsonl", [_public_rollout(record) for record in records])
    peak = int(torch.cuda.max_memory_allocated())
    origin = started
    for update in accumulator.updates:
        update["wall_time_s"] = float(update["wall_time_s"] - origin)
    summary = {
        "method": method,
        "seed": seed,
        "loss": LOSS_NAME,
        "parameters": counts,
        "math_scored_tokens": sum(record.teacher_scored_tokens for record in math_records),
        "short_scored_tokens": sum(record.teacher_scored_tokens for record in short_records),
        "optimizer_steps": len(accumulator.updates),
        "skipped_or_overflowed_steps": accumulator.skipped_steps,
        "short_final_update": bool(accumulator.updates and accumulator.updates[-1]["short_update"]),
        "generation_wall_time_s": generation_s,
        "scoring_wall_time_s": scoring_s,
        "on_policy_times_are_interleaved": method == "on_policy",
        "mean_generated_tokens": float(np.mean([len(record.token_ids) for record in records])),
        "mean_scored_tokens": float(np.mean([record.teacher_scored_tokens for record in records])),
        "wall_time_s": time.perf_counter() - started,
        "peak_memory_bytes": peak,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "step_tokens": STEP_TOKENS,
        "updates": accumulator.updates,
        "adapter_dir": str(adapter_dir),
    }
    _write_json(output_dir / "train_log.json", summary)
    return summary


def _public_rollout(record: RolloutRecord) -> dict:
    payload = asdict(record)
    return payload


def merge_adapter_into_fresh_backbone(adapter_dir: Path, merged_dir: Path) -> dict[str, float]:
    """Merge LoRA into a newly loaded FP16 0.5B copy and record the logit error."""
    from peft import PeftModel

    _require_cuda("adapter merge")
    adapter_dir = Path(adapter_dir)
    merged_dir = Path(merged_dir)
    base, tokenizer = load_frozen_causal(DRAFT_MODEL)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    peft_model.eval()
    probe = render_chat(tokenizer, TOKEN_CHECK_PROMPTS[0])
    input_ids = torch.tensor([probe], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        unmerged = peft_model(input_ids=input_ids, use_cache=False).logits.float()
    merged = peft_model.merge_and_unload()
    merged.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        merged_logits = merged(input_ids=input_ids, use_cache=False).logits.float()
    error = (unmerged - merged_logits).abs()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    report = {
        "max_abs_logit_error": float(error.max().cpu()),
        "mean_abs_logit_error": float(error.mean().cpu()),
        "probe_tokens": len(probe),
    }
    _write_json(Path(merged_dir) / "merge_error.json", report)
    return report


def trainable_names_match_targets(parameter_names: Sequence[str]) -> bool:
    """True when every trainable name mentions a LoRA target and no other projection family."""
    if not parameter_names:
        return False
    allowed = tuple(LORA_TARGET_MODULES)
    for name in parameter_names:
        if "lora_" not in name:
            return False
        if not any(module in name for module in allowed):
            return False
    return True
