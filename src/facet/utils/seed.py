"""Deterministic seeding across every RNG we touch."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 1337) -> int:
    """Seed python, numpy and (if present) torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
