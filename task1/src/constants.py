"""Locked Task 1 choices. Method grids here follow the task brief."""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2.5-1.5B"

ELIGIBLE_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Projections that read the same RMSNorm. Independent channel scales cannot
# all fold into that one norm weight.
SHARED_RMSNORM_GROUPS = {
    "input_layernorm": ("q_proj", "k_proj", "v_proj"),
    "post_attention_layernorm": ("gate_proj", "up_proj"),
}

# down_proj reads SiLU(gate) * up, which is not a single linear. o_proj reads
# the attention mix. Both always serialize FP16 channel scales.
STORE_SCALE_PROJECTIONS = ("o_proj", "down_proj")

PRUNE_SPARSITIES = (0.25, 0.50, 0.75)
P3_SPARSITIES = (0.50, 0.75)
P3_TAUS = (0.5, 1.0, 2.0)
SPARSITY_CLIP = (0.10, 0.85)

QUANT_BITS = (8, 4, 3)
Q3_BITS = (2, 3, 4, 8)
GROUP_SIZE = 128
AWQ_ALPHAS = (0.0, 0.5, 1.0)

# FP16 scale (2) + uint8 zero-point (1) + alignment byte (1).
GROUP_METADATA_BYTES = 4
FP16_BYTES = 2

HELLASWAG_FULL = 2000
HELLASWAG_DIAGNOSTIC = 512
SCORED_TOKENS = 50_000
CALIBRATION_SEQUENCES = 128
CALIBRATION_LENGTH = 512
DEVELOPMENT_SEQUENCES = 32

TIMING_WARMUP = 2
TIMING_REPEATS = 5
PREFILL_LENGTHS = (512, 2048)
DECODE_TOKENS = 64
DECODE_PROMPT_TOKENS = 16

TEST_FILENAMES = frozenset(
    {
        "wikitext2_test_windows_locked.jsonl.gz",
        "c4_validation_windows_locked.jsonl.gz",
        "hellaswag_val_locked.jsonl.gz",
    }
)

# Full 2,000-example HellaSwag. Every other condition uses the first 512 IDs.
FULL_HELLASWAG_CONDITIONS = frozenset(
    {
        "fp16",
        "p1_s50",
        "p2_s50",
        "p3_s50",
        "q1_b4",
        "q1_b3",
        "q2_b4",
        "q2_b3",
        "q3_b4",
        "q3_b3",
        "q3m_b4",
        "q3m_b3",
    }
)
