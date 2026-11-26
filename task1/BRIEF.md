# Task 1 brief — pruning and quantization

Implement this task only. Do not edit `task2/`, `task3/`, or `shared/`.

## Models and data

- Start every condition from a fresh FP16 copy of `Qwen/Qwen2.5-1.5B` (base, not Instruct). No training.
- Eligible matrices in every block: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Leave embeddings, norms, biases, and `lm_head` uncompressed. Count tied parameters once.
- Calibration: `data/public_locked/c4_calibration_locked.jsonl.gz` (128 × 512). Development selection only: `c4_development_locked.jsonl.gz` (32). Never choose anything on WikiText-2, C4 validation, or HellaSwag.
- Quality: WikiText-2 and C4 validation windows, 50,000 scored tokens each. Perplexity is `exp(mean NLL)`. Report `N` and mean NLL. Score each target once. Drop the first token of each window from the denominator. The locked files already store `score_target_offsets` and `scored_tokens`.
- HellaSwag: 2,000 locked validation examples. Score four endings by conditional log-likelihood of the ending given the context. Report accuracy and character-length-normalized accuracy (`acc_norm` = length in characters, not tokens). Full 2,000-example run for: FP16 baseline; magnitude, Wanda, and nonuniform at 50% sparsity; Q1, Q2, and Q3 at INT4 and at INT3. Every other condition: both perplexity suites plus the first 512 HellaSwag IDs only.
- Save per-example HellaSwag scores and per-window NLL sums so bootstrap can be recomputed.

## Algorithms (locked choices — implement these, do not invent a second method)

**P1 magnitude.** Per output row, score `|W_ij|`, zero the smallest fraction `s` in `{0.25, 0.50, 0.75}`. Restore the original weights between sparsities. Excluded parameters never count toward the sparsity target. Record achieved nonzero fraction and calibration layer-output relative MSE.

**P2 Wanda.** Same sparsities and per-row rule, score `|W_ij| * ||X_j||_2` using the same 128 calibration sequences. Save per-projection score summaries.

**P3 nonuniform Wanda.** Ranking rule stays Wanda. Allocation unit is one eligible projection.

- Sensitivity is the calibration output distortion of that projection: relative MSE `||(W - W_s) X||^2 / ||W X||^2` when the projection is Wanda-pruned at the target global sparsity. This is the lecture distortion proxy. Compute it on the 128 calibration sequences only.
- Allocation: `s_l ∝ sensitivity_l ** tau`, clipped to `[0.10, 0.85]`, then integer-weight repair so the total number of zeroed eligible weights equals the uniform P1/P2 budget at that sparsity. Repair by moving one weight at a time on the projection whose sparsity is farthest inside its clip.
- Choose `tau` from `{0.5, 1.0, 2.0}` by mean token NLL of the compressed model on the 32 development sequences only. Freeze the winner before any test file is opened. Save all three development scores.
- Run at global sparsity 50% and 75%.

**2:4 diagnostic.** On the 50% Wanda scores, in each group of 4 contiguous input channels keep the 2 highest scores. Compare quality with unstructured 50% Wanda at the same eligible-weight sparsity. Do not claim sparse-tensor-core speedup.

**Q1 uniform PTQ.** Group-wise asymmetric weight-only quantization, group size 128 along input columns, bits `{8, 4, 3}`.

```
q = clip(round(w / s) + z, 0, 2^b - 1)
w_hat = s * (q - z)
s = (w_max - w_min) / (2^b - 1)
z = clip(round(-w_min / s), 0, 2^b - 1)
```

Pad a trailing partial group by repeating the edge value only for the range statistic, and record the padding. Metadata per group is identical for Q1/Q2/Q3: FP16 scale (2 bytes) + uint8 zero-point (1 byte) + 1 alignment byte. Payload is little-endian bit-packed, byte-aligned at the end of each tensor. Reconstruct FP16 weights for numerical evaluation. Separately report packed bytes. Do not describe a fake-quantized FP16 matmul as a low-bit kernel.

**Q2 AWQ-style scaling.** Same quantizer. For each eligible projection search a recorded grid `alpha ∈ {0, 0.5, 1}`:

```
m_j = mean_t |X_{t,j}|
c_j(alpha) = m_j^alpha / mean_j(m_j^alpha)
```

Scale weight columns by `c_j`, quantize, and apply the inverse scale to the incoming activation. Pick `alpha` by calibration output MSE (alpha 0 must reproduce Q1). Before rounding, the FP16 rescaling must match the original projection output up to a tight relative tolerance; test this. Fold the inverse scale into a preceding compatible linear or RMSNorm weight when the algebra is exact. For `down_proj`, whose input is not a single linear, store the FP16 channel scales and add their bytes. State in the README that this is a 3-point activation-magnitude grid, not the full published AWQ search.

**Q3 mixed precision.** Same Q2 quantizer. Bits `{2, 3, 4, 8}`. Sensitivity is the Q2 calibration output MSE at INT4. Parameter-free rule: start every projection at 2 bits and upgrade the largest-sensitivity projection by one allowed step while the total packed bytes, including metadata, FP16 uncompressed tensors, folded or serialized scales, and tied-parameter counting, stay within the Q2 uniform-INT4 budget. Build a second allocation against the Q2 uniform-INT3 budget. Record the bit map and the byte check. Do not open test files while building the map. After the map is frozen, measure development-set calibration error as a diagnostic only.

## Resources

For every condition record: achieved sparsity or nominal bits, packed bytes with metadata, the unstructured-sparse estimate (1 mask bit per eligible weight + 16 bits per nonzero + byte padding) next to dense FP16, calibration time, peak allocated GPU memory, prefill latency at 512 and 2048 tokens, and 64-token decode latency. At least 5 warmed-up CUDA-synchronized repetitions via `shared.timing.time_call`. Label dense FP16 execution separately from packed-storage estimates.

## Statistics and plots

Use `shared.bootstrap`. Paired 95% intervals for P3−P2 and Q3−Q2: resample HellaSwag examples; for perplexity resample whole windows, recompute pooled NLL, then exponentiate. Plots: quality against sparsity and against packed bytes, pruning separate from quantization; one sensitivity figure and one sparsity/bit map per adaptive method. Save a few matched HellaSwag items with candidate scores.

## Code layout

- `task1/src/` algorithms and evaluation
- `task1/tests/` CPU tests on synthetic matrices. No weight download.
- `task1/run.py` with stages: `unit`, `smoke`, `baseline`, `prune`, `quantize`, `resources`, `analyze`. `smoke` uses tiny limits and must refuse to write into the final results directory.
- `task1/README.md` stating the AWQ simplification, the allocation rules, and how to reproduce.

## Tests that must pass before you finish

- Magnitude and Wanda masks match a hand-computed 4×6 example.
- P3 integer repair hits the exact zero count and stays inside `[0.10, 0.85]`.
- Quantizer round-trip on a known vector matches the formula, including a partial group.
- FP rescaling before rounding is numerically an identity.
- Packed-byte function matches a hand count.
- 2:4 keeps two weights in every group of four.
- Perplexity helper matches `exp(sum NLL / N)` and excludes the first token.

## Do not

- Download `Qwen/Qwen2.5-1.5B` onto this machine. Disk space is about 8 GB free.
- Run Kaggle or any multi-hour evaluation.
- Copy code from any other student's PA4 repository. The course data scripts may be imported, not rewritten.
- Commit data, checkpoints, or `results/`.
