# Pre-registered prediction — Task 2 B3

Written before any test-set MQAR accuracy is computed. The intervention is fixed. Development accuracy is a description of training, not a reason to change this choice.

## Intervention

Per-head forget gate on causal linear attention, mixed regime, `K=16` only, seeds 6304, 6305, and 6306, 4,500 updates, same optimizer budget as the other runs.

```
g_t = sigmoid(w_h · phi(q_t) + b_h)
S_t = g_t * S_{t-1} + phi(k_t) v_t^T
z_t = g_t * z_{t-1} + phi(k_t)
```

`phi(x) = ELU(x) + 1`. Feature dimension stays 32, so the persistent state `(S, z)` has the same byte size as ungated linear attention. `w_h` is initialized at 0 and `b_h` at 2, so the gate starts at `sigmoid(2) ≈ 0.88` (retain). No other variant is trained.

## Prediction

The forget gate should raise far-regime accuracy relative to ungated linear attention at K=16, and near-regime accuracy should stay close, because the gate reduces overwrite of older associations in a fixed-size state.

Near means source–query distance `< 64`. Far means distance `≥ 128`, using the stored source position. The comparator is the ungated ELU+1 linear model trained at K=16 in the mixed regime, not a dense or window model.
