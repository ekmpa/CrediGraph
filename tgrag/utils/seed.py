import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all relevant random number generators for reproducible experiments.

    Parameters:
        seed : int
            The seed value to use for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
