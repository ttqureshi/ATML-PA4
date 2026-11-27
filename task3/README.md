# Task 3 — exact speculative sampling and LoRA distillation

Target and teacher: `Qwen/Qwen2.5-1.5B-Instruct` (frozen). Draft and student: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on `q_proj`, `k_proj`, `v_proj`, and `o_proj` (`r=8`, `alpha=16`, dropout 0). One chat template is used for every condition. Temperature is 1 and top-p is 1. The reference seed is 6304; the second distillation seed is 6305.

Unit tests do not download weights. `smoke` and `distill` refuse to start when CUDA is unavailable.

```bash
python -m unittest discover -s task3/tests -v
python task3/run.py unit
```

GPU stages, in order, after `setup_data.py --public` and `--prefix-banks` on a CUDA machine: `smoke`, `baseline`, `distill`, `merge`, `align`, `eval`, `analyze` (`analyze` is CPU-only). Do not run the GPU stages on this CPU checkout. `align` reports no speed, so it may run on a second GPU (`CUDA_VISIBLE_DEVICES=1`) while `eval` times on the first; never run it next to `baseline` or `eval` on the same GPU.

| Stage | Needs | Writes (under `results/task3/`) |
|---|---|---|
| `smoke` | CUDA | `smoke.json`: token-id check; FP16 split-prefill and cache numbers (diagnostic); FP32 split-prefill, cached next logits and per-layer K/V vs a one-pass cache, hard-gated at `PA4_T3_FP32_GATE` (default `1e-3`) |
| `baseline` | CUDA | `baseline.json` (parts in `baseline_parts/`): original draft, gamma 2 and 4, 120 prompts x 2 paired repetitions, warm setup, isolated costs at 32/128/512 cached tokens, and the 24-prompt ignore-EOS math diagnostic |
| `distill` | CUDA | adapters in `checkpoints/task3/`; `distill/{method}_seed{seed}/train_log.json` and `rollouts.jsonl.gz` |
| `merge` | CUDA | merged FP16 drafts in `checkpoints/task3/`; `merge/{method}_seed{seed}/merge_error.json` (8-prompt logit error, merge cost) |
| `align` | CUDA, merged drafts, `$PA4_DATA_REPO/data/prefix_banks/prefix_banks_locked.jsonl.gz` | `alignment.json`, `alignment_rows.jsonl.gz` (parts in `align_parts/`) |
| `eval` | CUDA, merged drafts, the off-policy seed-6304 adapter | `eval.json` (parts in `eval_parts/`): speed rows, ignore-EOS diagnostic, unmerged-adapter wrapper overhead, standalone accuracy, matched continuations |
| `analyze` | the files above | `analyze.json` |

`align` fails if the supplied prefix-bank file is missing. With `PA4_DEADLINE` (unix seconds) set, every stage exits with code 75 before a piece that could overrun; rerunning resumes from the cached parts.

## Sampler contract

Let `p` be the target distribution and `q` the draft distribution at the current prefix.

1. The draft proposes up to `gamma` tokens, and fewer when it hits EOS, pad, or the remaining token cap.
2. Proposal `x` is accepted with probability `min(1, p(x) / q(x))`.
3. On rejection, the token is drawn from the normalized positive residual `(p - q)_+`. It is not drawn from `p`.
4. If all `gamma` proposals are accepted and none is EOS, one bonus token is drawn from the target. A short proposal or an accepted EOS does not take a bonus. A full accept that would pass the length cap suppresses the bonus (final-token truncation).
5. Both caches are then rolled back to the returned prefix minus its pending tail. Pad tokens are not part of that prefix.

Cache convention. `committed` is the prompt plus every emitted token. Between rounds each cache holds a prefix of `committed` that lacks at least its last token (the pending tail). Prefill caches the prompt minus its last token. A round feeds the draft its pending tokens as the input of the first proposal step (one token, or two after a full accept with a bonus), and feeds the target `[pending, d_1, ..., d_gamma]` in one verification forward whose `gamma + 1` rows give the acceptance distributions and the bonus distribution. Rollback is a crop of each cache to `len(committed) - 1` (or less); the emitted token is never forwarded on its own. A round therefore costs `gamma` draft forwards and one `(gamma + 1)`-token target forward, which is what `c` and `v_gamma` measure. `sync_caches()` feeds the pending tail when a caller needs caches equal to `committed` (the smoke K/V check). `tests/test_cached_decoder.py` checks tokens and per-round rows against a no-cache reference on tiny random Qwen2 models.

Random choices use one inverse-CDF draw per categorical sample and one uniform per acceptance test, from a NumPy generator. The length-3 Markov gate (`task3/tests/test_toy_sampler.py`, seed 6304, `gamma=2`, 100,000 sequences) compares sequence counts with the target's ancestral joint. That joint is the target-only law. The negative control draws the residual from `p` and must be rejected at `p < 0.01`. The summary is `task3/results_export/toy_sampler_chisquare.json`.

`1 - TV(p, q)` is the expected acceptance of one proposal and equals `sum_x min(p(x), q(x))`.

The analytical speedup is

```
S_hat(gamma) = (1 + sum_{j=1}^{gamma} alpha^j) / (gamma * c + v_gamma)
```

`c` is an isolated draft step divided by an isolated target step. `v_gamma` is a `(gamma + 1)`-token target verification (pending token plus `gamma` proposals) divided by one target step; the `gamma`-token width is recorded as `v_gamma_tokens_only`. Isolated costs are measured at 32, 128 and 512 cached tokens. `alpha` is the pooled accepted/examined ratio; accepted/proposed is reported beside it. The break-even acceptance is the smallest `alpha` in `[0, 1]` with `S_hat > 1`, or absent when even `alpha = 1` does not clear the cost. Measured throughput is paired with an adjacent target-only run; repetition order alternates. Bootstrap intervals use `shared.bootstrap`. Round timers use `shared.timing.cuda_sync` and are split into draft proposal, target verification, acceptance/residual sampling, and cache rollback. Prefill and host overhead are separate.

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

Training evidence (`train_log.json`): per-update loss, learning rate, pre-clip gradient norm, loss scale, skipped updates and cumulative teacher-scored tokens (the loss vs scored-token curve); synchronized seconds for model load, offline teacher-response preparation, student rollouts, teacher scoring, student forward and loss, backward, optimizer and adapter save; rollout length and termination summaries per category; peak memory; and backbone, adapter and teacher parameter counts. Categories are trained in order, math then short.

Standalone accuracy: math answers are taken from `oxed{}`, then a `####` marker, then the last number; format compliance means an explicit final-answer marker. Short-form answers are the first non-empty line with a short `label:` prefix removed; `correct` is a strict match, `correct_lenient` accepts the gold answer anywhere as a whole word, and format compliance means one line of at most three words. Each row keeps the decoded text, truncation, and a repeated-4-gram degeneration flag.

Prefix metrics are token-weighted forward KL, reverse KL, total variation, greedy agreement, and expected acceptance `1 - TV`. Splits are math, familiar extraction, and shifted extraction. Position bins are `[1, 16]`, `[17, 32]`, `[33, 64]`, and `65–192`. Teacher and `exact_spec_original_gamma4` prefixes are the same target law. Every draft (original and the four merged students) is scored on every bank: the supplied `teacher` and `original_draft` banks, `exact_spec_original_gamma4`, and each trained student's frozen own-rollout bank (producer checkpoint recorded). One forward of the longest prefix per (bank, prompt) gives all of its positions. `alignment.json` holds, per bank and draft, overall, positions 1-64 vs 65-192, by domain, by bin and by domain x bin; seed dispersion; prompt-clustered bootstrap differences against the original draft and on- vs off-policy; the target-vs-speculative consistency check (per-prompt target entropy, original-draft forward KL and 1 - TV, and source length, teacher bank minus speculative bank, paired prompt bootstrap, overall and per domain and bin); and decoded aligned disagreements chosen by a fixed rule (largest TV between target and original draft among greedy disagreements, two per shared bank and domain).

The wrapper-overhead run times the unmerged PEFT adapter of `off_policy` seed 6304 against its merged copy on 24 fixed prompts (first 8 per domain), both gammas, alternating order, same seeds. Matched continuations are the first two evaluation prompts of each domain.
