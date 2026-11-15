"""
Baseline seam finder: the straight column with the lowest total energy.

NOT one of the assignment's methods. It exists so the whole pipeline can be
run and tested before DP / Greedy / Graph Cut are finished. It is also a
useful "worst reasonable case" to compare against in the report.
"""

import numpy as np


def find_seam(energy: np.ndarray) -> np.ndarray:
    col = int(np.argmin(energy.sum(axis=0)))
    return np.full(energy.shape[0], col, dtype=np.int64)
