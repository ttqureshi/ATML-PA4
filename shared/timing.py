"""CUDA-synchronized timing. Warm up first, then record repetitions."""

from __future__ import annotations

import time
from typing import Callable


def cuda_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn: Callable[[], object], *, warmup: int, repeats: int) -> list[float]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    for _ in range(warmup):
        fn()
    cuda_sync()
    samples: list[float] = []
    for _ in range(repeats):
        cuda_sync()
        start = time.perf_counter()
        fn()
        cuda_sync()
        samples.append(time.perf_counter() - start)
    return samples
