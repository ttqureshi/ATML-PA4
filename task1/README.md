# Task 1 — pruning and weight-only quantization

This task compresses `Qwen/Qwen2.5-1.5B` (base, not Instruct) without training. Every condition starts from a fresh FP16 copy. Embeddings, RMSNorms, biases, and `lm_head` stay in FP16. Tied embeddings are counted once. The only matrices that are pruned or quantized are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

Numerical evaluation always runs the reconstructed dense FP16 weights. Packed bytes and the unstructured-sparse estimate are storage accounts. They are not timed kernels, and a matrix that contains zeros is not a sparse matmul.

## AWQ simplification

Q2 is not the published AWQ search and it is not an AWQ inference engine. For each eligible projection the activation magnitude of input channel `j` is

```
m_j = mean_t |X_{t,j}|
c_j(alpha) = m_j^alpha / mean_j(m_j^alpha)
```

on the 128 calibration sequences. The grid is `alpha ∈ {0, 0.5, 1}` only. `m_j^0` is defined as 1, including where `m_j` is 0, so alpha 0 is the all-ones scale and the quantizer matches Q1. A non-positive scale is replaced by 1 before it multiplies a weight. The chosen alpha is the one with the smaller calibration output MSE; a tie keeps the smaller alpha.

The weight columns are multiplied by `c_j`, then passed through the same group quantizer as Q1. Before any rounding, `(X / c) (W diag(c))^T` equals `X W^T`. After rounding, the inverse scale is applied as follows:

- `q_proj`, `k_proj`, and `v_proj` share `input_layernorm`. `gate_proj` and `up_proj` share `post_attention_layernorm`. The inverse scale is folded into that RMSNorm only when every consumer has the same `c`. The norm weight is divided by `c` and the projection keeps the scaled-space quantized weight.
- If those consumers disagree, each projection stores its own FP16 channel scales (2 bytes per input channel). For dense evaluation the inverse scale is baked into the reconstructed weight, and the stored scales are still counted.
- `o_proj` reads the attention mix, and `down_proj` reads `SiLU(gate) * up`, which is not a single linear. Both always store FP16 channel scales. `down_proj` is never folded into `up_proj`.

Q3 reuses the channel scales chosen by Q2 at INT4. Only the bit width changes.

## Allocation rules

**Uniform pruning (P1, P2).** In each output row, `floor(s * n_cols)` weights are zeroed. Magnitude ranks by `|W_ij|`. Wanda ranks by `|W_ij| * ||X_j||_2`. Ties zero the lower column index. The original FP16 weights are restored before the next sparsity. P3's zero budget is the sum of these per-row counts, not `round(s * total weights)`.

**P3.** The ranking stays Wanda. The allocation unit is one eligible projection. Sensitivity is the calibration relative MSE `||(W - W_s) X||^2 / ||W X||^2` when that projection alone is Wanda-pruned at the target global sparsity. This is an isolated probe: it does not measure interactions after every projection has been pruned.

The sparsity fraction is set proportional to `1 / (sensitivity + 1e-12) ** tau`, then clipped to `[0.10, 0.85]`. A higher isolated distortion therefore keeps a lower sparsity, up to the clip. `tau` is chosen from `{0.5, 1.0, 2.0}` by the pooled token NLL of the 50% model and the 75% model on the 32 development sequences (first token of each sequence dropped). The smaller tau wins a tie. All three scores are saved, and the winner is frozen before WikiText-2, C4 validation, or HellaSwag is opened.

Integer repair then adds or removes one weight at a time until the total number of zeros equals the uniform budget. The weight moves on the projection whose current fraction is farthest from the clip boundary in the direction of the move (farthest below 0.85 when more zeros are needed, farthest above 0.10 when fewer are needed). Equal distance keeps the earlier projection. Zeros inside a projection are spread across rows as evenly as possible, and each row still drops its lowest Wanda scores.

**2:4.** On the Wanda scores, each group of four contiguous input channels keeps the two highest scores. This is a quality comparison with unstructured 50% Wanda at the same eligible-weight sparsity. It is not a sparse-tensor-core speedup.

**Q1.** Group-wise asymmetric quantization along input columns, group size 128, bits `{8, 4, 3}`:

```
s = (w_max - w_min) / (2^b - 1)
z = clip(round(-w_min / s), 0, 2^b - 1)
q = clip(round(w / s) + z, 0, 2^b - 1)
w_hat = s * (q - z)
```

`round` is half-to-even (`numpy.rint`). A degenerate group (`w_max == w_min`) uses scale 1 so the division is defined; an all-zero group still reconstructs as 0. A trailing partial group repeats its last real weight only while min and max are computed. That repeated edge is already inside the group, so it does not move the range; the pad count is still recorded. Metadata is 4 bytes per group (FP16 scale, uint8 zero-point, one alignment byte). The payload is little-endian bit packing, padded to a byte at the end of the tensor.

**Q3.** Sensitivity is the Q2 calibration output MSE at INT4 (mean squared error of the original-space reconstruction). Every projection starts at 2 bits. The ladder is `2 → 3 → 4 → 8`. Each step promotes the most sensitive projection that is not already at 8 bits. If that one step would exceed the packed-byte budget, the search stops. It does not skip ahead to a less sensitive projection. One map is built against the Q2 uniform-INT4 packed size and one against Q2 uniform-INT3. The byte check includes payload, group metadata, folded or serialized scales, and FP16 tensors outside the eligible set, with tied parameters counted once. The map is written before any test file is opened. Development-set NLL and projection MSE are recorded afterwards and do not change the map.

## Evaluation

Perplexity is `exp(sum of token NLL / N)`. `N` is the locked `scored_tokens` count. The first token of each window is not scored. WikiText-2 and C4 validation each contribute 50,000 scored tokens.

HellaSwag uses the 2,000 locked validation rows. The query is `preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())` and each ending is `preprocess(ending)`, with the lm-evaluation-harness bracket cleanup. The score is the conditional token log-likelihood of the ending given the query. `acc_norm` divides by the character length of the preprocessed ending, not by the token count. FP16, the three 50% pruning methods, and Q1/Q2/Q3 at INT4 and INT3 use all 2,000 rows. Every other condition uses the first 512 orders.

Paired intervals use `shared.bootstrap`. HellaSwag resamples examples. Perplexity resamples whole windows, recomputes pooled NLL, and reports the helper's interval on that NLL difference together with the exponentiated point perplexities.

Latency uses `shared.timing.time_call` with 2 warmup passes and 5 repetitions, synchronized on CUDA when a GPU is present. Prefill is one dense forward at 512 and at 2048 tokens. The 64-token decode figure includes a 16-token prompt prefill plus 64 cached greedy steps, and it is labeled that way. Peak allocated bytes are CUDA peaks. On CPU they are omitted rather than invented.

## Reproduce

From the repository root, after the locked files exist under `external/atml-pa4-datasets` (commit `6a0a9ec9da7ef3d313ee3036cfec493886723644`):

```bash
python -m unittest discover -s task1/tests -v
python -m unittest shared.test_bootstrap
python task1/run.py unit
python task1/run.py smoke --output task1/runs/smoke
```

`smoke` uses a tiny random block, writes only to the path you give it, and refuses `results/` and `task1/results/`. It sets `PA4_FORBID_DOWNLOAD=1` and will error if Qwen loading is reached.

These stages need the checkpoint and a CUDA GPU. They are not part of the CPU tests:

```bash
python task1/run.py baseline --output task1/results
python task1/run.py prune --output task1/results
python task1/run.py quantize --output task1/results
python task1/run.py resources --output task1/results
python task1/run.py analyze --output task1/results
```

`prune` freezes `tau` in `allocation/p3_tau.json` before it opens a test file. `quantize` freezes both Q3 maps before it opens a test file. `resources` only fills latency that is still missing. `--skip-latency` on the quality stages leaves that work for `resources`. Do not commit `results/`, checkpoints, or the data checkout.
