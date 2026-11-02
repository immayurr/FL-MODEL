"""
utils/seed.py
─────────────
Single function that sets all relevant random seeds for reproducibility.
Call once at the top of every entry-point script (mpi_train.py, prepare_data.py).
"""

import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Seed Python's random, NumPy, and PyTorch (CPU + CUDA).

    Also forces deterministic cuDNN behaviour so runs are reproducible
    at a small performance cost. Acceptable for a research simulation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
