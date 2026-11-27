"""Distillation objective audit on tiny random Qwen2 models (CPU, no downloads).

The training path (`_score_records` -> `response_kl_tensor` -> `UpdateAccumulator`
-> `scaled_optimizer_step`) is compared against an independent reference that
computes, for every supervised response token j, KL(p_T(. | prompt, y_<j) ||
q_theta(. | prompt, y_<j)) in float64 from a separate forward of exactly that
prefix. The logged loss of every optimizer update must equal the mean of the
reference token KLs in that update, and the gradient handed to the optimizer
must equal the gradient of that mean after global-norm clipping at 1.0.
Prompt tokens are never supervised, the scored count respects quota
truncation, and updates split at `step_tokens` with a short final update.
"""

from __future__ import annotations

import unittest

import torch

from task3.src.distill import (
    GRAD_CLIP,
    RolloutRecord,
    UpdateAccumulator,
    _score_records,
    build_lora_config,
    trainable_names_match_targets,
)
from task3.tests.test_cached_decoder import tiny_qwen

STEP = 5


class CaptureSGD(torch.optim.SGD):
    """lr = 0, so parameters stay fixed; records the gradient seen at each step."""

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.seen: list[list[torch.Tensor]] = []

    def step(self, closure=None):
        self.seen.append([p.grad.detach().clone() for group in self.param_groups for p in group["params"]])
        return super().step(closure)


def _student():
    from peft import get_peft_model

    model = get_peft_model(tiny_qwen(22, sharpness=3.0), build_lora_config())
    torch.manual_seed(5)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.normal_(0.0, 0.2)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return model, trainable


def _reference_token_kl(teacher, student, prompt, response, j) -> torch.Tensor:
    ids = torch.tensor([list(prompt) + list(response[:j])])
    with torch.no_grad():
        p_log = torch.log_softmax(teacher(input_ids=ids, use_cache=False).logits[0, -1].double(), -1)
    q_log = torch.log_softmax(student(input_ids=ids, use_cache=False).logits[0, -1].double(), -1)
    return (p_log.exp() * (p_log - q_log)).sum()


class DistillLossTests(unittest.TestCase):
    def test_loss_and_gradient_match_independent_forward_kl(self) -> None:
        teacher = tiny_qwen(21, sharpness=3.0)
        student, trainable = _student()
        names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable_names_match_targets(names))
        prompt = [5, 9, 2, 33, 7]
        records = [
            RolloutRecord("a", 1, [4, 8, 15, 16, 23, 42 % 48, 1], "eos", 7, "math", "off_policy", 6304, 192, prompt),
            # Quota truncation: only the first four of six tokens are scored.
            RolloutRecord("b", 2, [3, 1, 4, 1, 5, 9], "truncated_quota", 4, "short", "off_policy", 6304, 32, prompt[:3]),
        ]
        optimizer = CaptureSGD(trainable)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        accumulator = UpdateAccumulator(step_tokens=STEP)
        _score_records(teacher, student, records, accumulator, scaler, optimizer, trainable)
        accumulator.flush(scaler, optimizer, trainable)

        supervised = [(record, j) for record in records for j in range(record.teacher_scored_tokens)]
        self.assertEqual(len(supervised), 11)
        self.assertEqual([u["tokens_in_update"] for u in accumulator.updates], [5, 5, 1])
        self.assertEqual(accumulator.scored_tokens, 11)
        self.assertEqual([u["teacher_scored_tokens_cumulative"] for u in accumulator.updates], [5, 10, 11])
        self.assertEqual(accumulator.updates[1]["tokens_by_category"], {"math": 2, "short": 3})
        self.assertTrue(accumulator.updates[-1]["short_update"])
        self.assertEqual(len(optimizer.seen), 3)

        groups = [supervised[0:5], supervised[5:10], supervised[10:11]]
        for update, group, seen in zip(accumulator.updates, groups, optimizer.seen):
            kls = [_reference_token_kl(teacher, student, r.prompt_token_ids, r.token_ids, j) for r, j in group]
            loss = torch.stack(kls).mean()
            self.assertAlmostEqual(update["loss"], float(loss), places=5)
            grads = torch.autograd.grad(loss, trainable, allow_unused=True)
            grads = [torch.zeros_like(p) if g is None else g.float() for g, p in zip(grads, trainable)]
            norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))
            self.assertAlmostEqual(update["grad_norm_before_clip"], norm, delta=1e-4 * max(1.0, norm))
            factor = min(1.0, GRAD_CLIP / (norm + 1e-6))
            for mine, ref in zip(seen, grads):
                self.assertTrue(torch.allclose(mine, ref * factor, rtol=1e-3, atol=1e-6))

        # The KL direction matters: the logged loss is not the reverse KL.
        r, j = supervised[2]
        ids = torch.tensor([list(r.prompt_token_ids) + list(r.token_ids[:j])])
        with torch.no_grad():
            p_log = torch.log_softmax(teacher(input_ids=ids).logits[0, -1].double(), -1)
            q_log = torch.log_softmax(student(input_ids=ids).logits[0, -1].double(), -1)
        forward = float((p_log.exp() * (p_log - q_log)).sum())
        reverse = float((q_log.exp() * (q_log - p_log)).sum())
        self.assertGreater(abs(forward - reverse), 1e-4)


if __name__ == "__main__":
    unittest.main()
