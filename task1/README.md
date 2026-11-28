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

Q3 and Q3m reuse the channel scales chosen by Q2 at INT4 (and the INT4 fold/store plan). Only the bit width changes.

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

**Q3 (v1, sensitivity ladder; kept as the unsuccessful design).** Sensitivity is the Q2 calibration output MSE at INT4 (mean squared error of the original-space reconstruction). Every projection starts at 2 bits. The ladder is `2 → 3 → 4 → 8`. Projections are ranked by sensitivity (descending, ties by module name). Each step promotes the most sensitive projection that is neither at 8 bits nor blocked by one ladder step. If that step would exceed the packed-byte budget, that projection is blocked (it keeps its current width) and the next most sensitive one is tried. The search stops when every projection is at 8 bits or blocked. One map is built against the Q2 uniform-INT4 packed size and one against Q2 uniform-INT3. The byte check includes payload, group metadata, folded or serialized scales, and FP16 tensors outside the eligible set, with tied parameters counted once. The map is written before any test file is opened. Development-set NLL and projection MSE are recorded afterwards and do not change the map.

The ladder spends the budget on a few projections: it drives the most sensitive ones to 8 bits and leaves most at 2 bits. That failure is visible on calibration and development data alone (total calibration MSE far above Q2's at the same bytes, development NLL 8.28 at the INT4 budget), which motivates Q3m.

**Q3m (v2, multiple-choice knapsack).** For every eligible projection the calibration output MSE is measured at each `b ∈ {2, 3, 4, 8}` with that projection's frozen Q2 INT4 alpha and the INT4 fold/store plan (the INT4 alphas and INT4 MSE are not searched again: they are read from `allocation/q2_alpha_search.json`, which `quantize` writes before any test file is opened, and must equal the alpha and MSE recorded for every projection in `conditions/q2_b4`). Every projection starts at 2 bits. At each step every upgrade of every projection from its current width to any higher allowed width (including `2 → 8` directly) is considered if its MSE reduction `mse[current] - mse[new]` is positive and its extra packed bytes keep the total at or under the budget. The upgrade with the largest reduction per extra byte is taken; ties go to the module name, then the smaller width. The search stops when no such upgrade exists. Packed bytes use the same byte function as Q2/Q3. Budgets are the `packed_bytes` of `q2_b4` and `q2_b3`, recomputed and checked. Greedy-by-ratio is not an exact knapsack solver: on small instances the last unusable bytes can leave it slightly behind another feasible map, which is why the saved allocation records the v1 and Q2 calibration totals under the same MSE table. Maps are written to `allocation/q3m_b4.json` and `allocation/q3m_b3.json` before any test file is opened; `allocation/q3m_development_diagnostic.json` follows and does not change them.

## Evaluation

Perplexity is `exp(sum of token NLL / N)`. `N` is the locked `scored_tokens` count. The first token of each window is not scored. WikiText-2 and C4 validation each contribute 50,000 scored tokens.

HellaSwag uses the 2,000 locked validation rows. The query is `preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())` and each ending is `preprocess(ending)`, with the lm-evaluation-harness bracket cleanup. The score is the conditional token log-likelihood of the ending given the query. `acc_norm` divides by the character length of the preprocessed ending, not by the token count. FP16, the three 50% pruning methods, and Q1/Q2/Q3/Q3m at INT4 and INT3 use all 2,000 rows. Every other condition uses the first 512 orders.

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

`task1/tests/test_pipeline_e2e.py` runs baseline → prune → quantize → quantize_q3m → resources → analyze through the real stage functions on a seeded tiny `Qwen2ForCausalLM` built from a config, with small synthetic locked files under a temporary `PA4_DATA_REPO`. Nothing is downloaded. It checks every summary, allocation file, bootstrap pair and figure, that no selection stage reads a test file before its freeze, that a finished directory resumes without reloading the model, and that smoke and real directories refuse each other.

### GPU pipeline smoke (run this before every full run)

```bash
python task1/run.py pipeline-smoke --output /kaggle/working/task1_smoke
```

It runs all six stages with the real checkpoint, one subprocess per stage like the notebook, under the smoke profile (`--smoke` on a single stage, or `PA4_T1_SMOKE=1`; sizes in `task1/src/run_profile.py`): 8 calibration and 4 development sequences cut to 128 tokens, 2 windows per perplexity suite, 16 HellaSwag items (8 for the diagnostic subset), only the last block's seven projections compressed, one timed repetition with an 8-token decode. Every condition, P3's tau search, 2:4, Q1/Q2/Q3/Q3m, fold and stored scales, the test-file gate, the resources rebuild (`--remeasure --only` one condition per recipe branch) and analyze (bootstrap and figures) still run. The directory gets `SMOKE_RUN_NOT_RESULTS.txt`, every summary says `"smoke": true`, figures are stamped, and real stages refuse that directory (smoke stages likewise refuse a real one). Expected time on a T4 is about 15 minutes.

These stages need the checkpoint and a CUDA GPU. They are not part of the CPU tests:

```bash
python task1/run.py baseline --output task1/results
python task1/run.py prune --output task1/results
python task1/run.py quantize --output task1/results
python task1/run.py quantize_q3m --output task1/results
python task1/run.py resources --output task1/results
python task1/run.py analyze --output task1/results
```

Every stage checks its inputs before loading weights: locked calibration/development files (128 and 32 sequences), the existence of the three test files (without opening them), and for `quantize_q3m` the `q2_b4`/`q2_b3` summaries. JSON outputs are written atomically, so a session killed mid-write never leaves a truncated summary. A stage whose conditions are all on disk returns without loading the model. `prune` freezes `tau` in `allocation/p3_tau.json` before it opens a test file. `quantize` saves its alpha search in `allocation/q2_alpha_search.json` (reused on resume) and freezes both Q3 maps before it opens a test file. `quantize_q3m` needs the finished `q2_b4` and `q2_b3` summaries, freezes both Q3m maps before it opens a test file, and skips any finished piece when rerun. `analyze` includes Q3m (and the Q3m−Q2 and Q3m−Q3 bootstrap pairs) whenever its conditions exist. `resources` only fills latency that is still missing. `--skip-latency` on the quality stages leaves that work for `resources`. Do not commit `results/`, checkpoints, or the data checkout.
