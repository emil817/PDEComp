"""Reproducibility helpers shared by benchmark entry points."""

import random

import numpy as np


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch when it is available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
