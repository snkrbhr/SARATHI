"""
thresholding.py — Global MAD Binary Search
===========================================
Finds the optimal global threshold such that the fraction of elements
below it equals the target compression ratio.

Uses Median Absolute Deviation (MAD) for robustness to outliers,
as described in the SARATHI paper (Section 3.1, NMF Residual Subspace Probe).
"""

import logging

import numpy as np


def find_mad_threshold(
    flat_scores: np.ndarray,
    compression_ratio: float,
    num_steps: int = 40,
) -> float:
    """
    Binary search for the Global MAD threshold.

    The threshold is expressed as:
        threshold = global_median - std_factor * global_MAD

    We search for std_factor such that the fraction of scores below
    the threshold equals compression_ratio.

    Args:
        flat_scores       : 1-D numpy array of all NMF residual scores.
        compression_ratio : Target fraction of elements to prune (e.g. 0.20).
        num_steps         : Number of binary search iterations.

    Returns:
        Scalar float threshold value.
    """
    if flat_scores.ndim != 1:
        raise ValueError("flat_scores must be a 1-D numpy array.")

    # Down-sample large arrays to speed up binary search without losing precision
    if flat_scores.size > 50_000_000:
        logging.info(f"Down-sampling {flat_scores.size:,} scores to 50M for fast binary search ...")
        sampled = np.random.choice(flat_scores, size=50_000_000, replace=False)
    else:
        sampled = flat_scores

    g_median = float(np.median(sampled))
    g_mad    = float(np.median(np.abs(sampled - g_median)))

    total_weights = sampled.size
    target_pruned = int(compression_ratio * total_weights)

    lo, hi = 0.0, 10.0
    best_threshold = g_median

    for _ in range(num_steps):
        mid = (lo + hi) / 2.0
        threshold = float(
            np.clip(
                g_median - mid * g_mad,
                sampled.min(),
                sampled.max(),
            )
        )
        pruned = int((sampled < threshold).sum())
        if pruned < target_pruned:
            hi = mid
        else:
            lo = mid
        best_threshold = threshold

    std_factor = (lo + hi) / 2.0
    logging.info(
        f"MAD threshold search | std_factor={std_factor:.4f} | "
        f"median={g_median:.6f} | MAD={g_mad:.6f} | threshold={best_threshold:.6f}"
    )
    return best_threshold
