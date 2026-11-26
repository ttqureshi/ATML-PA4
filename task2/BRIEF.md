# Task 2 brief — attention and context memory

Implement this task only. Do not edit `task1/`, `task3/`, or `shared/`.

## Part A — exact attention

Synthetic Q, K, V with `B=2`, `H=4`, `d_h=64`, `L ∈ {256, 512, 1024, 2048, 4096, 8192}`, FP16 and FP32 where the GPU supports them.

**A1 dense.** Explicit scores `Q K^T / sqrt(d)`, causal mask, stable softmax, multiply by V. Perturbing a future key or value must leave earlier outputs unchanged. Compare output and input-gradients to an FP32 reference at lengths that fit. Include an all-masked row and an extreme-score row.

**A2 tiled online softmax.** Blockwise forward that does not materialize the `L×L` score matrix. For a running max `m`, normalizer `l`, and accumulator `o`, a tile with scores `s` and values `V` updates

```
m' = max(m, max s)
l' = exp(m - m') * l + sum_j exp(s_j - m')
o' = exp(m - m') * o + sum_j exp(s_j - m') V_j
```

Verify against dense across tile sizes and dtypes. If backward uses autograd and stores intermediates, say so. Do not call this FlashAttention.

**A3 SDPA.** `torch.nn.functional.scaled_dot_product_attention` on the same tensors. Record the backend that actually ran. Compare forward time, forward+backward time, peak allocated and reserved memory, and output/gradient error. Ten warmed-up repetitions via `shared.timing.time_call`. Write unsupported or out-of-memory cells explicitly. Do not force an infeasible dense baseline at L=8192; benchmark the tiled and SDPA paths alone there.

CPU tests use `L≤128` and must pass without a GPU.

## Part B — MQAR

Use the course generator `scripts/mqar_generator.py` and the frozen files `data/mqar/mqar_test.jsonl.gz` (9,216) and `data/mqar/mqar_development.jsonl.gz` (2,304). Do not regenerate the test file. Verify the generator's invariant checks before training. Labels sit on the query positions. Loss is cross-entropy at those positions only. Chance baselines are `1/64` and `1/K`.

Architecture, identical except the attention operator: 2 pre-norm blocks, width 128, 4 heads of dimension 32, MLP width 512, learned token embeddings, sinusoidal positions, vocabulary 256. AdamW peak LR `8e-4`, weight decay 0.01, batch 32, 4,500 updates, 200-step warmup, linear decay to 15% of peak, global grad clip 1.0. Evaluate at updates 500, 1,000, 1,500, 2,000, 3,000, 4,500. Seeds 6304, 6305, 6306 for every trained run. Save the learning curve and the final checkpoint path (checkpoints stay out of git).

Regimes from the supplied generator: near (source–query distance `< 64`), far (`≥ 128`, not a single offset of 128), mixed. Also report bins `[0,32), [32,64), [64,128), [128,256)`. In a 2-layer window-64 model the receptive field is `2*(64-1)=126`, so far queries are unreachable by construction. Say that. Use the stored source position, not the query index, as the distance.

Train:

- Dense causal, mixed regime, `K ∈ {4, 8, 16}`, plus dense `K=16` near-only.
- Sliding window `W=64`, near-only at `K ∈ {4, 8, 16}`, and mixed at `K=16`.
- Test-time-only `W=64` mask on the trained dense model, compared with the window model trained under the mask.
- Causal linear attention, feature map `ELU(x)+1`, mixed regime, `K ∈ {4, 8, 16}`. Recurrent state `S_t = S_{t-1} + phi(k_t) v_t^T`, `z_t = z_{t-1} + phi(k_t)`, output `phi(q)^T S / (phi(q)^T z + eps)`.
- Mixed local/dilated at `K ∈ {4, 16}`: in every block, two heads attend to the preceding 64 positions; two heads attend to 64 causal positions at offsets `{0, 4, 8, ..., 252}`. A dense boolean mask is acceptable if its storage is reported as a dense mask, not a sparse kernel.

Report theoretical pair counts and state bytes at `T=256`. Separately, a forward-only sweep at `T ∈ {256, 1024, 4096, 8192}` with fixed random embeddings, matched shapes, no training.

**B3, pre-registered.** One intervention, no search over alternatives. Per-head forget gate on the linear state, from the lecture's gated recurrent update:

```
g_t = sigmoid(w_h · phi(q_t) + b_h)
S_t = g_t * S_{t-1} + phi(k_t) v_t^T
z_t = g_t * z_{t-1} + phi(k_t)
```

Initialize `b_h = 2` so the gate starts near 0.88 (retain). Feature dimension stays 32, so the persistent state bytes match the ungated linear baseline; the gate parameters are a small extra and must stay within 5% of the baseline trainable count. Train only at `K=16`, mixed regime, three seeds, 4,500 updates. Prediction, written into `task2/PREDICTION.md` before any test-set number is computed: the gate should reduce state interference and raise far-regime accuracy relative to ungated linear attention at K=16, with near-regime accuracy staying close. Report development accuracy as a description, not as a reason to change the intervention.

## Part C — KV cache of Qwen2.5-1.5B-Instruct

FP16, greedy generation, one chat template. Prompts are the 180 rows in `data/kv_retrieval/kv_retrieval_cases.jsonl.gz`. Do not relabel them. Do not give a policy the gold code or the character spans `evidence_char_start` / `evidence_char_end`. Those spans are Unicode slices of `user_text`, not token indices. Record actual token lengths after the tokenizer and chat template.

**C1.** Full-cache logits must match a single full forward pass within tolerance, stay finite, and reproduce greedy generation. Retained tokens keep their original rotary position IDs. Report full-cache exact-match accuracy on every factorial cell before interpreting compressed caches. If the FP16 reference is wrong on an example, keep it in unconditional accuracy and also report accuracy conditional on the reference being correct.

**Budget.** One case-specific budget: the serialized byte size of uniform INT4 channel-wise-key / token-wise-value cache at the pre-question context length. Group size 64. Each group has an FP16 scale and an FP16 offset (4 metadata bytes) plus a packed integer payload. Trailing `L mod 64` tokens stay FP16 for K and V. The question suffix and newly generated tokens are a shared FP16 decode component, reported separately and included in live cache memory.

```
M_b(L) = 2 * n_layers * H_kv * d_h * (floor(L/64) * (64*b/8 + 4) + 2*(L mod 64))
M_16(L) = 4 * n_layers * H_kv * d_h * L
```

Read `n_layers`, `H_kv`, and `d_h` from the model config. Do not hard-code them without an assertion against the config.

**C2 eviction**, both under that budget, with 8 sink tokens and 64 most-recent tokens reserved in both rules.

- Sink-plus-recent: fill the remaining budget with the next most recent positions.
- Question-aware: rank earlier positions by FP32 attention from the question suffix onto rotary-positioned cached keys. Keep the SDPA model for generation. Build scores from query projections; do not switch the whole model to eager attention. Fill the remaining budget with the highest-ranked positions.
- Retention is selected independently per layer and averaged across KV heads. Store one index set per layer and count bytes accordingly. Take the largest legal FP16 position count whose real bytes fit.

**C3 quantization**, group size 64, INT4, same trailing-FP16 rule.

- Token-wise groups for both K and V.
- Channel-wise groups for K and token-wise groups for V.
Same rounding and scale/offset. Measure K error and V error separately, teacher-forced logit KL, and free-running exact match. Quantize the prompt cache and later completed groups the same way.

**C4 hybrid**, same byte budget. Keep 8 sinks and 64 recent tokens in FP16. Add more question-aware positions in FP16. Store the rest as channel-wise-K / token-wise-V INT2, only in complete groups of 64; an incomplete tail stays FP16. Count metadata, position indices, padding, and FP16 payload. Do not exceed the case budget.

**C5.** Five compressed methods plus the FP16 reference, on the same prompts: exact match, reference-answer NLL, post-prefill logit KL, fraction of evidence tokens kept (eviction and hybrid FP16 selection), packed bytes, scoring time, prefill, decode, peak allocated memory. Paired bootstrap of prompts, at least 2,000 resamples, via `shared.bootstrap`, for the principal accuracy differences. Cells of 10 prompts are descriptive only. Separately benchmark pack and unpack at batch sizes 1 and 8 and both context lengths, ten warmed-up repetitions: pack time, unpack time, serialized bytes, and decode-step latency with unpacking versus FP16. Distinguish packed bytes from allocated memory after dequantization.

## Code layout

- `task2/src/` attention, training, KV policies
- `task2/tests/` CPU tests
- `task2/run.py` stages: `unit`, `smoke`, `attention`, `train`, `kv`, `analyze`. Smoke cannot write final results.
- `task2/README.md` and `task2/PREDICTION.md`

## Tests that must pass before you finish

- Future perturbation does not change past dense outputs.
- Tiled output matches dense FP32 on a short sequence for two tile sizes.
- Linear recurrence matches the explicit feature-map sum.
- Window and dilation masks have the stated number of allowed keys.
- The gate adds less than 5% parameters and does not change state bytes.
- `M_b(L)` matches a hand example with small `n_layers, H_kv, d_h, L`.
- Eviction always keeps the 8 sinks and 64 recent tokens and never exceeds the byte budget.
- A completed INT4 group unpacks to the quantized values.

## Do not

- Download Qwen weights locally. Disk space is about 8 GB free.
- Run the 42 training jobs or the 180-prompt suite here.
- Copy another student's PA4 code. Import the course MQAR generator.
- Commit checkpoints or `results/`.
