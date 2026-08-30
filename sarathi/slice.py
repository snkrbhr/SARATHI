"""
slice.py — Phase 3: Structured Removal
=======================================
Takes the neuron-level importance scores η_i from Phase 2 and performs
physical structured slicing of the FFN weight matrices.

Removes the bottom neurons by η score to achieve the target structured
compression ratio. Updates model config so the saved model loads cleanly.

Two modes:
  sarathi_structured_slice  — Uniform K neurons kept per layer.
  sarathi_adaptive_slice    — Variable K per layer (Global MAD threshold).
"""

import logging

import torch
from tqdm import tqdm

from .score import LayerScore
from .model_utils import get_intermediate_size, set_intermediate_size, slice_mlp_layer


def sarathi_structured_slice(
    layer_scores     : list,
    model,
    model_name       : str,
    compression_ratio: float,
) -> dict:
    """
    Phase 3: Keep the top-K neurons per layer based on their SARATHI scores.

    Uniform structured slicing: the same K is applied to every layer.

    Args:
        layer_scores      : List of LayerScore from Phase 2.
        model             : Loaded HuggingFace causal LM.
        model_name        : Architecture identifier string.
        compression_ratio : Fraction of neurons to remove (e.g. 0.25).

    Returns:
        Stats dictionary with keys: original_size, new_size, compression_ratio, variant.
    """
    logging.info("=== Phase 3: SARATHI Structured Slicing ===")

    orig_size = get_intermediate_size(model_name, model)
    K         = int(orig_size * (1.0 - compression_ratio))

    logging.info(
        f"Target: keep Top-{K}/{orig_size} neurons per layer "
        f"({compression_ratio * 100:.1f}% removed)"
    )

    for lns in tqdm(layer_scores, desc="Slicing layers"):
        top_k = torch.topk(lns.scores, K, largest=True)
        keep_indices, _ = torch.sort(top_k.indices)
        slice_mlp_layer(model_name, lns.layer, keep_indices)

    set_intermediate_size(model_name, model, K)

    logging.info(
        f"SARATHI structured slicing complete. "
        f"New intermediate_size = {K} "
        f"(FLOPs reduced by ~{compression_ratio * 100:.1f}%)"
    )

    variant = layer_scores[0].variant if layer_scores else "Unknown"
    return {
        "original_size"    : orig_size,
        "new_size"         : K,
        "compression_ratio": compression_ratio,
        "variant"          : variant,
    }


def sarathi_adaptive_slice(
    layer_scores     : list,
    model,
    model_name       : str,
    compression_ratio: float,
    min_keep_ratio   : float = 0.50,
) -> dict:
    """
    Phase 3 (Adaptive): Global MAD threshold → variable neurons kept per layer.

    Computes a single global threshold over all layer scores and applies it
    independently to each layer, allowing layers with naturally cleaner weight
    geometry to be pruned more aggressively.

    Args:
        layer_scores      : List of LayerScore from Phase 2.
        model             : Loaded HuggingFace causal LM.
        model_name        : Architecture identifier string.
        compression_ratio : Target average fraction of neurons to remove.
        min_keep_ratio    : Floor: never keep fewer than this fraction per layer.

    Returns:
        Stats dictionary including adaptive_sizes list.
    """
    from .thresholding import find_mad_threshold
    import numpy as np

    logging.info("=== Phase 3: SARATHI Adaptive Structured Slicing ===")

    orig_size = get_intermediate_size(model_name, model)

    # 1. Gather all scores
    all_scores = np.concatenate([lns.scores.cpu().numpy() for lns in layer_scores])

    # 2. Find global threshold
    logging.info("Searching for Global MAD threshold for adaptive sparsity ...")
    threshold = find_mad_threshold(all_scores, compression_ratio)
    logging.info(f"Global Adaptive Threshold: {threshold:.6f}")

    # 3. Apply threshold per layer
    adaptive_sizes = []
    abs_min_keep   = max(int(orig_size * min_keep_ratio), 1)

    for lns in tqdm(layer_scores, desc="Adaptive Slicing"):
        scores      = lns.scores
        mad_indices = (scores >= threshold).nonzero(as_tuple=True)[0]

        n_survive = len(mad_indices)
        if n_survive < abs_min_keep:
            _, keep_indices = torch.topk(scores, abs_min_keep, largest=True)
            n_survive = abs_min_keep
        else:
            keep_indices = mad_indices

        keep_indices, _ = torch.sort(keep_indices)  # preserve weight order
        adaptive_sizes.append(len(keep_indices))
        slice_mlp_layer(model_name, lns.layer, keep_indices)

    total_kept  = sum(adaptive_sizes)
    total_orig  = orig_size * len(layer_scores)
    actual_ratio = 1.0 - (total_kept / total_orig)

    logging.info(
        f"SARATHI adaptive slicing complete.\n"
        f"Adaptive sizes (min/max/avg): "
        f"{min(adaptive_sizes)} / {max(adaptive_sizes)} / "
        f"{sum(adaptive_sizes) // len(adaptive_sizes)}\n"
        f"Actual FLOPs reduced by ~{actual_ratio * 100:.2f}%"
    )

    variant = layer_scores[0].variant if layer_scores else "Unknown"
    return {
        "original_size"    : orig_size,
        "new_size"         : "adaptive",
        "compression_ratio": actual_ratio,
        "variant"          : variant,
        "adaptive_sizes"   : adaptive_sizes,
    }
