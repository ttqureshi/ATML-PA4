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


class PhaseClock:
    """Synchronized wall time per training phase (teacher rollout, scoring, backward, ...)."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def measure(self, name: str):
        from contextlib import contextmanager

        from shared.timing import cuda_sync

        @contextmanager
        def _span():
            cuda_sync()
            start = time.perf_counter()
            try:
                yield
            finally:
                cuda_sync()
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start
                self.calls[name] = self.calls.get(name, 0) + 1

        return _span()


def _null_span():
    from contextlib import nullcontext

    return nullcontext()


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return _null_span()


def response_kl_tensor(teacher, student, prompt_ids: Sequence[int], response_ids: Sequence[int], clock: PhaseClock | None = None) -> torch.Tensor:
    """Per-token KL(teacher || student) at every response position of one microbatch.

    One forward of prompt + response through each model; the logit at index
    len(prompt) - 1 + j is the next-token distribution before response token j.
    The teacher pass is under no_grad; temperature is 1 for both models.
    """
    if not response_ids:
        raise ValueError("response prefix is empty")
    ids = list(prompt_ids) + list(response_ids)
    device = next(student.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with clock.measure("teacher_scoring") if clock else _null_span():
        with torch.no_grad(), _autocast(device):
            teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
    with clock.measure("student_forward_and_loss") if clock else _null_span():
        with _autocast(device):
            student_logits = student(input_ids=input_ids, use_cache=False).logits
        if teacher_logits.shape[-1] != student_logits.shape[-1]:
            raise RuntimeError("teacher and student vocabularies differ; the KL needs the common full vocabulary")
        return response_token_kl(teacher_logits, student_logits, len(prompt_ids), len(response_ids))


def scaled_optimizer_step(scaler, optimizer, parameters: Sequence[torch.nn.Parameter], n_tokens: int) -> dict:
    """Unscale, divide by the supervised tokens, clip, then step.

    Returns `skipped` (GradScaler found non-finite gradients and skipped the
    step), the pre-clip global gradient norm of the token-averaged loss, and the
    loss scale before and after the update.
    """
    if n_tokens < 1:
        raise ValueError("refusing an empty optimizer step")
    scaler.unscale_(optimizer)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(n_tokens)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP))
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scale_after = float(scaler.get_scale())
    return {
        "skipped": scale_after < scale_before,
        "grad_norm_before_clip": grad_norm if math.isfinite(grad_norm) else None,
        "loss_scale_before": scale_before,
        "loss_scale_after": scale_after,
    }


@dataclass
class UpdateAccumulator:
    """Groups supervised tokens into optimizer updates of `step_tokens` (last may be short).

    The logged loss is the token-averaged forward KL of the update, which is
    also the objective the gradient was normalized by.
    """

    step_tokens: int = STEP_TOKENS
    pending_count: int = 0
    pending_value: float = 0.0
    updates: list[dict] = field(default_factory=list)
    skipped_steps: int = 0
    scored_tokens: int = 0
    clock: PhaseClock | None = None
    pending_categories: dict[str, int] = field(default_factory=dict)
    category: str = ""

    def add_kl(self, kl: torch.Tensor, scaler, optimizer, parameters) -> None:
        offset = 0
        total = int(kl.shape[0])
        while offset < total:
            room = self.step_tokens - self.pending_count
            take = min(room, total - offset)
            chunk = kl[offset : offset + take]
            with self.clock.measure("backward") if self.clock else _null_span():
                scaler.scale(chunk.sum()).backward(retain_graph=offset + take < total)
            self.pending_count += take
            self.scored_tokens += take
            self.pending_categories[self.category] = self.pending_categories.get(self.category, 0) + take
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
        with self.clock.measure("optimizer") if self.clock else _null_span():
            step = scaled_optimizer_step(scaler, optimizer, parameters, n_tokens)
        self.skipped_steps += int(step["skipped"])
        self.updates.append(
            {
                "step": len(self.updates) + 1,
                "loss": loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "tokens_in_update": n_tokens,
                "teacher_scored_tokens_cumulative": self.scored_tokens,
                "tokens_by_category": dict(self.pending_categories),
                "short_update": n_tokens != self.step_tokens,
                **step,
                "wall_time_s": time.perf_counter(),
            }
        )
        self.pending_count = 0
        self.pending_value = 0.0
        self.pending_categories = {}


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
    clock: PhaseClock | None = None,
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
        with clock.measure("teacher_rollout_prep") if clock else _null_span():
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
        with accumulator.clock.measure("student_rollout") if accumulator.clock else _null_span():
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
        accumulator.category = record.category
        offset = 0
        while offset < len(scored_ids):
            room = accumulator.step_tokens - accumulator.pending_count
            take = min(room, len(scored_ids) - offset)
            # A fresh forward of the scored prefix. Only the newly included tail
            # enters this microbatch, so a response can cross a 256-token boundary.
            prefix = scored_ids[: offset + take]
            kl = response_kl_tensor(teacher, student, record.prompt_token_ids, prefix, accumulator.clock)[offset:]
            accumulator.add_kl(kl, scaler, optimizer, parameters)
            offset += take


def distill_results_dir(method: str, seed: int) -> Path:
    """Exported copy of the training evidence (results/ is what leaves the GPU box)."""
    from shared.paths import REPO_ROOT

    return REPO_ROOT / "results" / "task3" / "distill" / f"{method}_seed{seed}"


def _length_summary(records: Sequence[RolloutRecord]) -> dict:
    if not records:
        return {"n_presentations": 0}
    lengths = np.asarray([len(record.token_ids) for record in records], dtype=np.float64)
    scored = np.asarray([record.teacher_scored_tokens for record in records], dtype=np.float64)
    reasons: dict[str, int] = {}
    for record in records:
        reasons[record.termination_reason] = reasons.get(record.termination_reason, 0) + 1
    return {
        "n_presentations": len(records),
        "n_unique_prompts": len({record.source_id for record in records}),
        "generated_tokens_total": int(lengths.sum()),
        "generated_tokens_mean": float(lengths.mean()),
        "generated_tokens_median": float(np.median(lengths)),
        "generated_tokens_max": int(lengths.max()),
        "scored_tokens_total": int(scored.sum()),
        "unscored_generated_tokens": int(lengths.sum() - scored.sum()),
        "termination_reasons": reasons,
        "at_response_cap": int(sum(1 for record in records if len(record.token_ids) >= record.response_cap)),
        "quota_truncated": int(sum(1 for record in records if record.termination_reason == "truncated_quota")),
    }


def train_method(
    method: str,
    seed: int,
    output_dir: Path,
    *,
    data_root: Path | None = None,
    results_out: Path | None = None,
) -> dict:
    """Train one seed. Refuses to run, and does not download weights, without CUDA.

    Writes train_log.json and rollouts.jsonl.gz both next to the adapter and
    under results/task3/distill/{method}_seed{seed}/ so the evidence is exported.
    """
    if loss_for_method(method) != LOSS_NAME:
        raise RuntimeError("on-policy training must stay on forward KL")
    _require_cuda(f"distill {method} seed {seed}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_out = distill_results_dir(method, seed) if results_out is None else Path(results_out)
    math_examples, short_examples = load_training_pool(data_root)
    evaluation = load_eval_pool(data_root)
    assert_no_eval_leak([*math_examples, *short_examples], evaluation)

    torch.cuda.reset_peak_memory_stats()
    clock = PhaseClock()
    started = time.perf_counter()
    with clock.measure("model_load"):
        teacher, teacher_tokenizer = load_frozen_causal(TARGET_MODEL)
        student, student_tokenizer, trainable = load_lora_student(seed)
    assert_same_token_ids(teacher_tokenizer, student_tokenizer)
    counts = parameter_report(student)
    teacher_counts = parameter_report(teacher)
    trainable_names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    if not trainable_names_match_targets(trainable_names):
        raise RuntimeError("trainable parameters are not exactly the LoRA q/k/v/o adapters")
    if any(parameter.dtype != torch.float32 for parameter in trainable):
        raise RuntimeError("adapter parameters must be FP32")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    eos_id = None if teacher_tokenizer.eos_token_id is None else int(teacher_tokenizer.eos_token_id)
    accumulator = UpdateAccumulator(clock=clock)

    optimizer.zero_grad(set_to_none=True)
    teacher.eval()
    loop_started = time.perf_counter()
    if method == "off_policy":
        pools = {}
        for category, examples, quota in (("math", math_examples, MATH_QUOTA), ("short", short_examples, SHORT_QUOTA)):
            pools[category] = _generate_category(
                teacher,
                teacher_tokenizer,
                examples,
                category=category,
                train_seed=seed,
                method=method,
                quota=quota,
                eos_id=eos_id,
                stop_at_quota=False,
                clock=clock,
            )
        math_records, short_records = pools["math"], pools["short"]
        _score_records(teacher, student, math_records + short_records, accumulator, scaler, optimizer, trainable)
        accumulator.flush(scaler, optimizer, trainable)
    else:
        # Generate from the student that will actually be updated, then score
        # that rollout before drawing the next one. Stepping inside the rollout
        # does not rewrite tokens that were already sampled.
        on_policy = {}
        for category, examples, quota in (("math", math_examples, MATH_QUOTA), ("short", short_examples, SHORT_QUOTA)):
            on_policy[category] = _on_policy_category(
                student,
                student_tokenizer,
                teacher,
                examples,
                category=category,
                train_seed=seed,
                quota=quota,
                eos_id=eos_id,
                accumulator=accumulator,
                scaler=scaler,
                optimizer=optimizer,
                parameters=trainable,
            )
        math_records, short_records = on_policy["math"], on_policy["short"]
        accumulator.flush(scaler, optimizer, trainable)
    loop_s = time.perf_counter() - loop_started
    records = math_records + short_records
    peak = int(torch.cuda.max_memory_allocated())

    adapter_dir = output_dir / "adapter"
    with clock.measure("adapter_save"):
        student.save_pretrained(adapter_dir)
        student_tokenizer.save_pretrained(adapter_dir)
    rollouts = [_public_rollout(record) for record in records]
    _write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    _write_jsonl_gz(results_out / "rollouts.jsonl.gz", rollouts)
    for update in accumulator.updates:
        update["wall_time_s"] = float(update["wall_time_s"] - started)
    phases = dict(clock.seconds)
    # Off-policy pays for an offline teacher-response pool; on-policy pays for student rollouts.
    generation_s = phases.get("teacher_rollout_prep", 0.0) + phases.get("student_rollout", 0.0)
    training_s = sum(phases.get(name, 0.0) for name in ("teacher_scoring", "student_forward_and_loss", "backward", "optimizer"))
    total_s = time.perf_counter() - started
    summary = {
        "method": method,
        "seed": seed,
        "loss": LOSS_NAME,
        "kl_direction": "KL(p_teacher || q_student), full common vocabulary, temperature 1",
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES),
            "n_trainable_tensors": len(trainable_names),
            "trainable_parameter_names_sample": trainable_names[:8],
        },
        "parameters": counts,
        "parameter_counts": {
            "student_backbone_frozen": counts["frozen"],
            "adapter_trainable": counts["trainable"],
            "student_total_with_adapter": counts["total"],
            "teacher_total": teacher_counts["total"],
        },
        "math_scored_tokens": sum(record.teacher_scored_tokens for record in math_records),
        "short_scored_tokens": sum(record.teacher_scored_tokens for record in short_records),
        "teacher_scored_tokens_total": accumulator.scored_tokens,
        "rollouts": {"math": _length_summary(math_records), "short": _length_summary(short_records)},
        "optimizer_steps": len(accumulator.updates),
        "skipped_or_overflowed_steps": accumulator.skipped_steps,
        "skipped_steps": [update["step"] for update in accumulator.updates if update["skipped"]],
        "short_final_update": bool(accumulator.updates and accumulator.updates[-1]["short_update"]),
        "phase_seconds": phases,
        "phase_calls": dict(clock.calls),
        "offline_teacher_response_preparation_s": phases.get("teacher_rollout_prep", 0.0),
        "student_rollout_s": phases.get("student_rollout", 0.0),
        "teacher_scoring_s": phases.get("teacher_scoring", 0.0),
        "student_forward_and_loss_s": phases.get("student_forward_and_loss", 0.0),
        "backward_s": phases.get("backward", 0.0),
        "optimizer_s": phases.get("optimizer", 0.0),
        "generation_wall_time_s": generation_s,
        "scoring_and_training_wall_time_s": training_s,
        "training_loop_wall_time_s": loop_s,
        "on_policy_times_are_interleaved": method == "on_policy",
        "mean_generated_tokens": float(np.mean([len(record.token_ids) for record in records])),
        "mean_scored_tokens": float(np.mean([record.teacher_scored_tokens for record in records])),
        "wall_time_s": total_s,
        "end_to_end_cost_s": total_s,
        "peak_memory_bytes": peak,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "step_tokens": STEP_TOKENS,
        "category_order": ["math", "short"],
        "loss_vs_teacher_scored_tokens": [
            [update["teacher_scored_tokens_cumulative"], update["loss"]] for update in accumulator.updates
        ],
        "updates": accumulator.updates,
        "adapter_dir": str(adapter_dir),
    }
    _write_json(output_dir / "train_log.json", summary)
    _write_json(results_out / "train_log.json", summary)
    return summary


def _write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _public_rollout(record: RolloutRecord) -> dict:
    payload = asdict(record)
    return payload


MERGE_PROBE_PROMPTS = TOKEN_CHECK_PROMPTS + (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Read this report date: May 25, 2020. Return only the month.",
    "Shipping label: destination code PK, serial 253. Output the destination code.",
)


def merge_adapter_into_fresh_backbone(adapter_dir: Path, merged_dir: Path, report_path: Path | None = None) -> dict[str, float]:
    """Merge LoRA into a newly loaded FP16 0.5B copy and record the logit error.

    The probe batch is MERGE_PROBE_PROMPTS (fixed, chat-formatted, one forward
    each). The unmerged PEFT model (FP16 base, FP32 adapter, FP16 autocast) and
    the merged FP16 model see the same ids. Merge wall time is the deployment
    cost that goes into the break-even calculation. The report is written next
    to the merged weights and, when given, to `report_path` under results/.
    """
    from peft import PeftModel

    from shared.timing import cuda_sync

    _require_cuda("adapter merge")
    adapter_dir = Path(adapter_dir)
    merged_dir = Path(merged_dir)
    cuda_sync()
    started = time.perf_counter()
    base, tokenizer = load_frozen_causal(DRAFT_MODEL)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    peft_model.eval()
    cuda_sync()
    load_s = time.perf_counter() - started
    probes = [render_chat(tokenizer, prompt) for prompt in MERGE_PROBE_PROMPTS]
    unmerged = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for probe in probes:
            ids = torch.tensor([probe], dtype=torch.long, device="cuda")
            unmerged.append(peft_model(input_ids=ids, use_cache=False).logits.float()[0])
    cuda_sync()
    merge_started = time.perf_counter()
    merged = peft_model.merge_and_unload()
    merged.eval()
    cuda_sync()
    merge_s = time.perf_counter() - merge_started
    max_abs = 0.0
    abs_sum = 0.0
    n_values = 0
    prob_max_abs = 0.0
    argmax_agree = 0
    n_positions = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for probe, reference in zip(probes, unmerged):
            ids = torch.tensor([probe], dtype=torch.long, device="cuda")
            logits = merged(input_ids=ids, use_cache=False).logits.float()[0]
            error = (reference - logits).abs()
            max_abs = max(max_abs, float(error.max()))
            abs_sum += float(error.sum())
            n_values += int(error.numel())
            prob_gap = (torch.softmax(reference, -1) - torch.softmax(logits, -1)).abs().max()
            prob_max_abs = max(prob_max_abs, float(prob_gap))
            argmax_agree += int((reference.argmax(-1) == logits.argmax(-1)).sum())
            n_positions += int(reference.shape[0])
    save_started = time.perf_counter()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    save_s = time.perf_counter() - save_started
    report = {
        "adapter_dir": str(adapter_dir),
        "probe_prompts": list(MERGE_PROBE_PROMPTS),
        "probe_tokens": [len(probe) for probe in probes],
        "max_abs_logit_error": max_abs,
        "mean_abs_logit_error": abs_sum / max(1, n_values),
        "max_abs_probability_error": prob_max_abs,
        "argmax_agreement": argmax_agree / max(1, n_positions),
        "n_positions": n_positions,
        "load_s": load_s,
        "merge_s": merge_s,
        "save_s": save_s,
        "merge_cost_s": load_s + merge_s + save_s,
    }
    _write_json(merged_dir / "merge_error.json", report)
    if report_path is not None:
        _write_json(Path(report_path), report)
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
