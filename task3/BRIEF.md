# Task 3 brief — speculative decoding and distillation

Implement this task only. Do not edit `task1/`, `task2/`, or `shared/`.

## Models

- Target and teacher: `Qwen/Qwen2.5-1.5B-Instruct`, frozen.
- Draft and student: `Qwen/Qwen2.5-0.5B-Instruct`, frozen backbone, LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` with `r=8`, `alpha=16`, dropout 0.
- One chat template for every condition. Confirm target and draft token IDs match on a fixed prompt set before any timing. Temperature 1, top-p 1.
- Primary speed cap: 64 new tokens, EOS-aware. Separate answer-quality cap: 512 for math, 32 for extraction. A 24-prompt math diagnostic generates exactly 64 tokens for timing only and is not an answer-quality run.

## Exact speculative sampling

Let `p` be the target distribution and `q` the draft distribution at the current prefix.

- Draft proposes up to `gamma` tokens.
- Accept proposal `x` with probability `min(1, p(x) / q(x))`.
- On rejection, sample from the normalized positive residual `(p - q)_+`.
- If all `gamma` tokens are accepted, sample one bonus token from the target.
- Roll the draft and target caches back to exactly the returned prefix. An incorrect rollback fails the tests even if the acceptance test is right.
- Handle EOS, variable proposal length, final-token truncation, and padding.

**Correctness gate, CPU, before any Qwen run.** A 3-symbol Markov target and draft for which the exact length-3 joint can be computed in closed form. Draw 100,000 sequences with seed 6304. Pearson chi-square must not reject the correct sampler at `p < 0.01`, and every expected count must be at least 10. A wrong sampler that draws the residual from `p` instead of the positive residual must be rejected at `p < 0.01`. Report joint sequence probabilities, not only one-step marginals.

Also check, once a GPU exists: split-prefill logits match a full forward, and cache positions stay consistent. That check is a `smoke` stage, not a local download.

## Distillation

Data, after `setup_data.py --public`: 500 GSM8K training examples and 300 short-form training examples. Held-out: 60 GSM8K test and 60 short-form test (30 familiar templates, 30 shifted). Never put evaluation prefixes in the training pool.

Two methods, each from the original student, seeds 6304 and 6305, independently initialized LoRA.

- Scored-token quota 40,000 per run: 30,000 math + 10,000 short.
- Response caps: 192 math, 32 short. Temperature 1, top-p 1.
- Microbatch 1. One optimizer step per 256 teacher-scored tokens. A short final update is allowed.
- AdamW LR `3e-4`, weight decay 0.01. FP16 frozen backbone, FP32 adapters and optimizer states, FP16 autocast with dynamic GradScaler. Unscale before clipping. Clip global norm 1.0. Normalize the KL by the supervised tokens in the update.
- Loss is forward KL of the full vocabulary, `KL(p_teacher || q_student)`, at every response prefix. The sampled token is not a hard label.
- Off-policy: freeze a teacher-generated pool first, then cycle it in a fixed seeded round-robin across the two quotas. Count teacher generation in the cost.
- On-policy: sample from the current student, score with the teacher, do not differentiate through generation. Same quotas.
- Prompt schedule: seeded round-robin inside each quota, reuse prompts after a category is exhausted. Save source id, rollout seed, token ids, termination reason, and teacher-scored token count.
- If the quota ends mid-response, score only the allowed prefix and record truncation.
- Log skipped or overflowed steps, loss, learning rate, lengths, wall time, and peak memory. Equal scored tokens do not imply equal wall time; report both, and the realized optimizer-step count.

## Prefix metrics

Banks from `setup_data.py --prefix-banks`: `teacher` and `original_draft`. Generate `exact_spec_original_gamma4` with this sampler at `gamma=4` from the original student. Teacher and exact-speculative banks are the same target-prefix law; a systematic gap is an implementation bug, not a finding. Also generate each trained student's own prefixes and evaluate both trained students on each other's frozen rollouts. Compare matched prefix IDs only.

On every draft (original, two off-policy seeds, two on-policy seeds) report forward KL, reverse KL, total variation, greedy agreement, and expected acceptance `1 - TV`. Separate math, familiar extraction, and shifted extraction; bins `[1,16]`, `[17,32]`, `[33,64]`, and long positions 65–192. Token-weighted denominators. Show seed dispersion and a few aligned disagreements.

## Execution measurement

Merge each adapter into a fresh FP16 0.5B copy. One merged checkpoint per seed and method, stored off git (Kaggle dataset later). Check merged versus unmerged logits on one batch and record the error. Time one unmerged adapter only as a wrapper-overhead note.

Same sampler, same 120 prompts, `gamma ∈ {2, 4}`, two paired repetitions. Counterbalance order. Time an adjacent target-only run for every condition. Record accepted/examined, accepted/proposed, mean accepted prefix, target passes per output token, draft forwards, tokens/second, peak memory, EOS and truncation. Split each round into draft proposal, target verification, acceptance/residual sampling, and cache rollback. Prefill and host overhead are separate.

Cost model: `c` is draft-step cost over target-step cost, `v_gamma` is verification cost over one target step,

```
S_hat(gamma) = (1 + sum_{j=1}^{gamma} alpha^j) / (gamma * c + v_gamma)
```

Report `c`, `v_gamma`, predicted speedup, measured speedup, and the break-even acceptance (or that none exists). Paired bootstrap 95% intervals over prompts for throughput and acceptance, seeds separate. Use `shared.bootstrap` and `shared.timing`.

Standalone student accuracy uses the answer-length caps and an extractor for the GSM8K final number and the short-form format. Speculative samples follow the target and must not be used to rank the student. Include distillation plus merge cost in a per-request break-even.

## Code layout

- `task3/src/speculative.py`, `distill.py`, `metrics.py`, `evaluate.py`, `alignment.py`, `analysis.py`
- `task3/tests/test_toy_sampler.py` containing the 100,000-draw chi-square gate
- `task3/run.py` stages: `unit`, `smoke`, `baseline`, `distill`, `merge`, `align`, `eval`, `analyze`
- `task3/README.md` with the training accounting and the sampler contract

## Tests that must pass before you finish

- The chi-square gate and the negative control, on CPU.
- Bonus-token path and rejection path on a scripted categorical example with hand-computed probabilities.
- Cache rollback leaves the cache length equal to the accepted prefix.
- EOS stops a proposal and is not followed by a bonus token from past the stop.
- `1 - TV` equals the expected acceptance on a small categorical pair.
- Forward KL matches a hand example.
- LoRA target modules and rank match the brief. A parameter-count check can use a tiny linear stand-in; do not download Qwen.

## Do not

- Download the 1.5B or 0.5B weights locally. Disk space is about 8 GB free.
- Start distillation locally or on Kaggle.
- Copy another student's PA4 repository.
- Commit adapters, merged weights, or rollout dumps. Commit code, configs, and the toy-test output summary only.
