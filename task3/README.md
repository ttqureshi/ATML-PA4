# Task 3 — exact speculative sampling and LoRA distillation

Target and teacher: `Qwen/Qwen2.5-1.5B-Instruct` (frozen). Draft and student: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on `q_proj`, `k_proj`, `v_proj`, and `o_proj` (`r=8`, `alpha=16`, dropout 0). One chat template is used for every condition. Temperature is 1 and top-p is 1. The reference seed is 6304; the second distillation seed is 6305.

Unit tests do not download weights. `smoke` and `distill` refuse to start when CUDA is unavailable.

```bash
python -m unittest discover -s task3/tests -v
python task3/run.py unit
```

GPU stages, in order, after `setup_data.py --public` on a CUDA machine (`PA4_DATA_REPO` points at the data repo): `prefix_banks`, `smoke`, `baseline`, `distill`, `merge`, `align`, `eval`, `analyze` (`analyze` is CPU-only). Do not run the GPU stages on this CPU checkout. `align` reports no speed, so it may run on a second GPU (`CUDA_VISIBLE_DEVICES=1`) while `eval` times on the first; never run it next to `baseline` or `eval` on the same GPU.

| Stage | Needs | Writes (under `results/task3/`) |
|---|---|---|
| `prefix_banks` | CUDA, `data/local_source_revisions.json`, `data/public_locked/gsm8k_test_locked.jsonl.gz`, `data/short_form/short_test.jsonl` | `$PA4_DATA_REPO/data/prefix_banks/prefix_banks_locked.jsonl.gz` and `prefix_banks_build.json` (also under `results/task3/`; parts in `prefix_bank_parts/`) |
| `smoke` | CUDA | `smoke.json`: token-id check; FP16 split-prefill and cache numbers (diagnostic); FP32 split-prefill, cached next logits and per-layer K/V vs a one-pass cache, hard-gated at `PA4_T3_FP32_GATE` (default `1e-3`) |
| `baseline` | CUDA | `baseline.json` (parts in `baseline_parts/`): original draft, gamma 2 and 4, 120 prompts x 2 paired repetitions, warm setup, isolated costs at 32/128/512 cached tokens, and the 24-prompt ignore-EOS math diagnostic |
| `distill` | CUDA | adapters in `checkpoints/task3/`; `distill/{method}_seed{seed}/train_log.json` and `rollouts.jsonl.gz` |
| `merge` | CUDA | merged FP16 drafts in `checkpoints/task3/`; `merge/{method}_seed{seed}/merge_error.json` (8-prompt logit error, merge cost) |
| `align` | CUDA, merged drafts, the `prefix_banks` output | `alignment.json`, `alignment_rows.jsonl.gz` (parts in `align_parts/`) |
| `eval` | CUDA, merged drafts, the off-policy seed-6304 adapter | `eval.json` (parts in `eval_parts/`): speed rows, ignore-EOS diagnostic, unmerged-adapter wrapper overhead, standalone accuracy, matched continuations |
| `analyze` | the files above | `analyze.json` |

`align` fails if the supplied prefix-bank file is missing or does not cover all 120 evaluation prompts in both banks. Every stage checks its inputs (data counts 60/30/30 and 500/300, adapters, merged drafts, bank coverage) before it loads a model. Writes are atomic; an unreadable or older-version cache part, an adapter without its `train_log.json`, or a merged draft without its `merge_error.json` is rebuilt. Hub models load at the commits in `data/local_source_revisions.json`. With `PA4_DEADLINE` (unix seconds) set, every stage exits with code 75 before a piece that could overrun; rerunning resumes from the cached parts.

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

## Supplied prefix banks (`prefix_banks`)

The course command `setup_data.py --prefix-banks` crashes under the pinned transformers 4.57.1: `generate()` rejects its `generator=` kwarg. `python task3/run.py prefix_banks` rebuilds the same file with `task3/src/prefix_banks.py`: same pinned model revisions, tokenizer check, prompt list and order (the 60 GSM8K test rows are read from the sha256-checked `gsm8k_test_locked.jsonl.gz`, which holds exactly the manifest rows; the prompt texts must equal the evaluation pool), chat template, per-prompt seeds `6304 + 10007 j (+123 for the draft)`, position RNG and bins, and record fields. Only `generate`'s internal sampler is replaced, by an explicit KV-cached loop that applies what `generate` applies with those arguments (float32 logits, the generation_config repetition penalty, temperature 1, top-k 0, top-p 1, stop after either generation_config EOS id, EOS kept, 192-token cap) and draws with `torch.multinomial(probs, 1, generator=gen)` from the seeded CUDA generator. Before generating, each model replays a short real `generate()` call through the loop and must match its processed distributions (`generate_equivalence_check`). The deviation is recorded in `prefix_banks_build.json`.

Qwen2.5's generation_config sets `repetition_penalty=1.1`, which `generate()` keeps, so the supplied `teacher` bank follows the penalized target law. `align` therefore also builds `teacher_no_rep_penalty` (plain temperature-1 target samples, same seeds and caps) and reports `target_law_control_consistency` (that bank vs exact speculative prefixes) next to the required `target_vs_speculative_consistency`.

## Smoke mode

```bash
for s in prefix_banks smoke baseline distill merge align eval analyze; do python task3/run.py --smoke $s || break; done
```

`--smoke` (or `PA4_SMOKE=1`) runs every stage on the real models with sizes reduced, not logic: 4 evaluation prompts (2 math, 1 familiar, 1 shifted; the prefix banks use the same prompts with their full-list seeds and the 192-token cap), 24-token speed and diagnostic caps, 2 diagnostic prompts, 1 wrapper prompt per domain, a 64-token math answer cap, and 320 + 64 scored distillation tokens per run (one full and one short update). Gammas, repetitions, the FP32 gate, all four distillation runs, all five drafts and every bank are unchanged. Output goes to `results/task3_smoke/` and `checkpoints/task3_smoke/`, every JSON object carries `SMOKE_RUN`, and a full-run `analyze` refuses smoke files. The smoke `analyze` fails unless every expected output exists and `smoke_coverage` is all true, then deletes `checkpoints/task3_smoke/` (about 4 GB of merged drafts). Expected time on one T4: about 20 minutes including the first model download.
