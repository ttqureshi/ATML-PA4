# Task 3 — exact speculative sampling and LoRA distillation

Target and teacher: `Qwen/Qwen2.5-1.5B-Instruct` (frozen). Draft and student: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on `q_proj`, `k_proj`, `v_proj`, and `o_proj` (`r=8`, `alpha=16`, dropout 0). One chat template is used for every condition. Temperature is 1 and top-p is 1. The reference seed is 6304; the second distillation seed is 6305.

Unit tests do not download weights. `smoke` and `distill` refuse to start when CUDA is unavailable.

```bash
python -m unittest discover -s task3/tests -v
python task3/run.py unit
```

GPU stages, in order, after `setup_data.py --public` and `--prefix-banks` on a CUDA machine: `smoke`, `baseline`, `distill`, `merge`, `eval`, `analyze`. Do not run those stages on this CPU checkout.

## Sampler contract

Let `p` be the target distribution and `q` the draft distribution at the current prefix.

1. The draft proposes up to `gamma` tokens, and fewer when it hits EOS, pad, or the remaining token cap.
2. Proposal `x` is accepted with probability `min(1, p(x) / q(x))`.
3. On rejection, the token is drawn from the normalized positive residual `(p - q)_+`. It is not drawn from `p`.
4. If all `gamma` proposals are accepted and none is EOS, one bonus token is drawn from the target. A short proposal or an accepted EOS does not take a bonus. A full accept that would pass the length cap suppresses the bonus (final-token truncation).
5. Both caches are then rolled back or advanced until their lengths and token ids equal the returned prefix. Pad tokens are not part of that prefix.

Random choices use one inverse-CDF draw per categorical sample and one uniform per acceptance test, from a NumPy generator. The length-3 Markov gate (`task3/tests/test_toy_sampler.py`, seed 6304, `gamma=2`, 100,000 sequences) compares sequence counts with the target's ancestral joint. That joint is the target-only law. The negative control draws the residual from `p` and must be rejected at `p < 0.01`. The summary is `task3/results_export/toy_sampler_chisquare.json`.

`1 - TV(p, q)` is the expected acceptance of one proposal and equals `sum_x min(p(x), q(x))`.

The analytical speedup is

```
S_hat(gamma) = (1 + sum_{j=1}^{gamma} alpha^j) / (gamma * c + v_gamma)
```

`c` is an isolated draft step divided by an isolated target step. `v_gamma` is a `gamma`-token target verification divided by one target step. The break-even acceptance is the smallest `alpha` in `[0, 1]` with `S_hat > 1`, or absent when even `alpha = 1` does not clear the cost. Measured throughput is paired with an adjacent target-only run; repetition order alternates. Bootstrap intervals use `shared.bootstrap`. Round timers use `shared.timing.cuda_sync` and are split into draft proposal, target verification, acceptance/residual sampling, and cache rollback. Prefill and host overhead are separate.

The primary speed cap is 64 new tokens and is EOS-aware. A 24-prompt math diagnostic generates exactly 64 tokens with EOS ignored, for timing only. Answer-quality caps are 512 (math) and 32 (extraction). Speculative samples follow the target and are not used to rank the student.

## Distillation accounting

Both off-policy and on-policy training minimize forward KL, `KL(teacher || student)`, over the full vocabulary at every response prefix. The sampled token is not a hard label. On-policy tokens are drawn from the current student and the generation pass is not differentiated.

| Item | Value |
|---|---|
| Seeds and methods | 6304 and 6305, each from a fresh LoRA, off-policy and on-policy |
| Scored-token quota | 40,000 per run: 30,000 math then 10,000 short |
| Response caps | 192 math, 32 short |
| Update | microbatch 1; one AdamW step per 256 scored tokens; a short final step is allowed |
| Optimizer | AdamW, LR `3e-4`, weight decay 0.01, clip global norm 1.0 after unscale |
| Precision | FP16 frozen backbone, FP32 adapters and AdamW state, FP16 autocast, dynamic GradScaler |
| Loss scale | sum of token KLs divided by the number of supervised tokens in that update |

Prompts are a seeded shuffle inside each category (`seed` for math, `seed + 7919` for short) and then repeat from the start of that same shuffle. Off-policy training freezes a teacher pool first, cycling prompts until the quota is covered, then scores that pool. The last response is truncated when the quota ends inside it, and the termination reason is `truncated_quota`. On-policy training generates one student response, scores it, and only then draws the next response, so an optimizer step is visible to later rollouts. Teacher generation is counted in the off-policy wall time; student generation is counted in the on-policy wall time. Equal scored-token totals do not imply equal wall time. Each rollout stores source id, rollout seed, token ids, termination reason, and the teacher-scored token count. Skipped or overflowed GradScaler steps are logged. Evaluation ids never enter the training pool.

Each adapter is merged into a fresh FP16 0.5B copy. One probe batch records the merged-versus-unmerged logit error. The unmerged adapter is timed only as a wrapper-overhead note.

Prefix metrics are token-weighted forward KL, reverse KL, total variation, greedy agreement, and expected acceptance `1 - TV`. Splits are math, familiar extraction, and shifted extraction. Position bins are `[1, 16]`, `[17, 32]`, `[33, 64]`, and `65–192`. Teacher and `exact_spec_original_gamma4` prefixes are the same target law. Drafts are also scored on frozen rollouts from the other trained students, matched by prefix id.
