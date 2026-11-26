# Task 2 — Exact attention, MQAR, and KV-cache compression

CPU checks, from the worktree root:

```bash
python -m unittest discover -s task2/tests -v
python task2/run.py smoke
```

`smoke` runs one optimizer step and the budget and causality checks. It does not download weights and it does not write into `results/` or `checkpoints/`.

The course data repo is expected at `external/atml-pa4-datasets` (commit `6a0a9ec9da7ef3d313ee3036cfec493886723644`), or in the directory named by `PA4_DATA_REPO`. Training calls `scripts/mqar_generator.py`. It reads the frozen development and test files and does not rewrite them.

## Architecture

Every MQAR model is the same 2-block pre-norm decoder except for the attention operator: width 128, 4 heads of dimension 32, GELU MLP of width 512, bias-free projections, learned token embeddings, fixed sinusoidal positions, vocabulary 256. Loss is cross-entropy on the value id at the query positions only. Chance levels are `1/64` and `1/K`.

Distance is the query position minus the stored source position. Near is `< 64`, far is `≥ 128`, and the bins are `[0,32), [32,64), [64,128), [128,256)`.

In a 2-layer window-64 model the receptive field is `2*(64-1)=126`, so far queries are unreachable by construction.

Attention operators:

| Kind | What it computes |
|---|---|
| `dense` | Explicit scores, causal mask, stable softmax, multiply by V |
| `window` | Same, with a dense boolean window of 64. This is not a sparse kernel |
| `mixed` | Heads 0–1 use that window. Heads 2–3 use offsets `{0,4,...,252}`. Dense boolean mask |
| `linear` | `phi(x)=ELU(x)+1`. Training uses an inclusive prefix tensor `(B,H,L,D,D)`, not an `L×L` score matrix |
| `gated` | The pre-registered forget gate. `b_h` starts at 2. Persistent `(S,z)` bytes match ungated linear attention |

The tiled operator in Part A is a Python online-softmax loop. Its backward pass is autograd and stores the tile intermediates. It is not FlashAttention.

## Stages

```bash
python task2/run.py unit
python task2/run.py smoke
python task2/run.py attention
python task2/run.py train
python task2/run.py kv
python task2/run.py analyze
```

`attention`, `train`, and `kv` are the GPU stages. On a CPU, `attention` only runs lengths up to 128 so it cannot be mistaken for the official grid. `train` runs the 42 jobs (14 architectures × seeds 6304, 6305, 6306) for 4,500 updates and then evaluates the frozen test file. `--skip-test` leaves the test file unread. `--only gated_K16` selects a job by substring.

`kv` loads `Qwen/Qwen2.5-1.5B-Instruct` in FP16 with `local_files_only=True` and `attn_implementation="sdpa"`. If the weights are absent it stops. It does not download them. Generation is greedy, with one chat template: a single user message and `add_generation_prompt=True`.

## B3

The prediction is in `PREDICTION.md`. The gate is trained only at K=16, mixed regime, three seeds. Development accuracy is recorded on the learning curve and is not used to change the intervention.

## Rotary position ids

Retained KV rows keep the absolute positions they had in the prompt. Those ids are stored beside the cache. RoPE for a new token uses that absolute id (`position_ids`). The causal mask and the append index use the physical cache length (`cache_position`), which is shorter after eviction. Keys already in the cache are not rotated a second time and are not renumbered to `0 .. P-1`. `model.generate` is not used, because it would set the rotary id from the physical length.

Question-aware scores are an FP32 matmul of projected queries against the rotary-positioned cached keys. The SDPA model is still the one that generates. The policy does not see the gold code or `evidence_char_start` / `evidence_char_end`. The question suffix is the text from `QUESTION_BEGIN` onward. The INT4 budget `M_b` is measured on the pre-question context. The question suffix and new tokens are a separate FP16 decode component and are included in live cache bytes.

Quantized groups use `scale * code + offset`, with `offset` equal to the group minimum. Code 0 maps to that offset, so there is no extra integer zero-point. Rounding is PyTorch round-half-to-even. Packing puts the earliest element in the low bits. A trailing `L mod 64` tokens stay FP16. Channel-wise groups run along tokens; token-wise groups run along features.

## What still has to run on a GPU

1. `python task2/run.py attention` — lengths 256 through 8192, FP16 and FP32, ten warmed-up repetitions. Dense is skipped at 8192. The forward sweep uses T in {256, 1024, 4096, 8192}.
2. `python task2/run.py train` — 42 runs of 4,500 updates, then frozen-test accuracy and the test-time window control.
3. `python task2/run.py kv` — 180 prompts, five compressed caches plus the FP16 reference, and the pack/unpack benchmark at batches 1 and 8 and contexts 2048 and 4096.
4. `python task2/run.py analyze` — paired bootstrap of the saved prompt rows.

Checkpoints and `results/` stay out of git.
