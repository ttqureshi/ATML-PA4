"""Reference seed for every PA4 experiment."""

from __future__ import annotations

import random

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
