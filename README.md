# ATML PA4 — Compression, Attention Memory, and Speculative Decoding

**Course:** EE-5102 / CS-6304, Advanced Topics in Machine Learning, Fall 2026  
**Author:** Muhammad Tayyab Tahir Qureshi (25280024)

This repository contains the code for Programming Assignment 4. The three tasks are independent experiments:

| Path | Task |
|---|---|
| `task1/` | Pruning and weight quantization of `Qwen/Qwen2.5-1.5B` |
| `task2/` | Exact attention, MQAR training, and KV-cache compression |
| `task3/` | Exact speculative sampling and draft-model distillation |

Course data comes from [atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets) at commit `6a0a9ec9da7ef3d313ee3036cfec493886723644`. Clone it under `external/atml-pa4-datasets` (gitignored). Do not commit raw datasets, model weights, or large checkpoints.

```bash
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git external/atml-pa4-datasets
git -C external/atml-pa4-datasets checkout --detach 6a0a9ec9da7ef3d313ee3036cfec493886723644
pip install -r requirements.txt
pip install -r external/atml-pa4-datasets/requirements-public.txt
python external/atml-pa4-datasets/scripts/setup_data.py --public
```

`--prefix-banks` needs a CUDA GPU and is run only after the Task 3 sampler exists. The reference seed is `6304`. Model revisions, tokenizer, library versions, device, and precision are recorded with every run.

Shared timing and bootstrap helpers live in `shared/`. Each task keeps its own algorithms under `taskN/`.
