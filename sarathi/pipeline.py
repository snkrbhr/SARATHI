"""
pipeline.py — SARATHI Pruning Pipeline Orchestrator
=====================================================
High-level function to apply the complete SARATHI method to a model.

Three core phases:
  Phase 1: PROBE  — NMF Residual Subspace Probe (data-free, weight-geometry-based)
  Phase 2: SCORE  — Magnitude-weighted neuron importance scoring (SwiGLU-aware)
  Phase 3: SLICE  — Physical structured removal of low-importance neurons

Optional Phase 4:
  Phase 4: OBS    — Adaptive OBS weight reconstruction (O(d²) memory per layer)
                    + Bias Shift Compensation for post-LayerNorm correction

The pipeline ALWAYS ends with a physically STRUCTURED slice:
  intermediate_size is permanently reduced in model.config.
  Real FLOPs are removed regardless of variant.

Variants:
  E (default) — SARATHI: data-free NMF Residual Subspace Probe + OBS
  B           — SARATHI-Wanda: Wanda-weighted probe (128 samples) for gated
  A           — Magnitude probe baseline (data-free)
  C           — NMF scoring on Wanda-masked weights
"""

import gc
import logging
import time

import torch

from .probe import magnitude_probe, wanda_probe, nmf_residual_probe
from .score import compute_sarathi_scores
from .slice import sarathi_structured_slice, sarathi_adaptive_slice


def apply_sarathi_pruning(
    model,
    model_name          : str,
    variant             : str,
    structured_ratio    : float,
    probe_sigma         : float,
    n_calib             : int   = 128,
    nmf_rank            : int   = 7,
    nmf_iters           : int   = 100,
    calib_dataset       : str   = "wikitext",
    tokenizer                   = None,
    is_adaptive         : bool  = False,
    adaptive_min_keep   : float = 0.50,
    do_obs_reconstruct  : bool  = False,
    calib_tokens                = None,
    obs_damping         : float = 1e-6,
    multi_gpu           : bool  = False,
) -> dict:
    """
    Run the end-to-end SARATHI pipeline on a loaded model.

    Args:
        model            : Loaded HuggingFace causal LM.
        model_name       : Architecture identifier string (e.g. 'meta-llama/Llama-3-8B').
        variant          : Pruning variant:
                           'E' — SARATHI (NMF Residual Subspace Probe, data-free) [default]
                           'B' — SARATHI-Wanda (Wanda probe, 128 samples)
                           'A' — Magnitude probe baseline
                           'C' — NMF scoring on Wanda-masked weights
        structured_ratio : Target fraction of FFN neurons to permanently remove.
        probe_sigma      : Unstructured probe sparsity σ applied in Phase 1.
        n_calib          : Calibration samples (Variants B and C only).
        nmf_rank         : NMF factorization rank r (default 7).
        nmf_iters        : NMF multiplicative update iterations (default 100).
        calib_dataset    : 'wikitext' | 'c4' | 'alpaca' (Variants B/C + OBS).
        tokenizer        : HuggingFace tokenizer (Variants B, C; OBS reconstruction).
        is_adaptive      : If True, uses adaptive structured slicing (variable K/layer).
        adaptive_min_keep: Floor fraction: never keep fewer than this per layer.
        do_obs_reconstruct: If True, runs Adaptive OBS Weight Reconstruction (Phase 4).
        calib_tokens     : [n_samples, seq_len] Long tensor (required for OBS).
        obs_damping      : Tikhonov regularisation λ for OBS Cholesky solve.
        multi_gpu        : Whether model is distributed across GPUs (13B+ models).

    Returns:
        Stats dict: original_size, new_size, compression_ratio, variant,
                    and optionally adaptive_sizes.
    """
    logging.info("=" * 60)
    logging.info(f"[SARATHI] Pipeline — Variant {variant}")
    logging.info(f"  structured ratio = {structured_ratio:.2%}  (permanent FFN slice)")
    logging.info(f"  probe σ          = {probe_sigma:.2%}")
    if variant in ("B", "C"):
        logging.info(f"  calib dataset    = {calib_dataset}")
        logging.info(f"  n_calib          = {n_calib}")
    if variant in ("C", "E"):
        logging.info(f"  nmf_rank         = {nmf_rank}")
        logging.info(f"  nmf_iters        = {nmf_iters}")
    logging.info(f"  OBS reconstruct  = {do_obs_reconstruct}")
    logging.info("=" * 60)

    act_rms = None
    masks   = []

    _t_total = time.time()

    # ── Phase 1: PROBE ────────────────────────────────────────────────────────
    _t0 = time.time()

    if variant == "A":
        masks = magnitude_probe(model, model_name, sigma=probe_sigma)

    elif variant == "E":
        # SARATHI: data-free NMF Residual Subspace Probe
        masks = nmf_residual_probe(
            model, model_name,
            sigma     = probe_sigma,
            nmf_rank  = nmf_rank,
            nmf_iters = nmf_iters,
        )

    else:
        # Variants B and C: Wanda activation-weighted probe
        if tokenizer is None:
            raise ValueError(
                f"Variant {variant} requires a tokenizer for Wanda calibration."
            )
        masks, act_rms = wanda_probe(
            model, model_name, tokenizer,
            sigma        = probe_sigma,
            n_calib      = n_calib,
            calib_dataset= calib_dataset,
        )

    logging.info(f"[TIMING] Phase 1 (Probe):               {time.time() - _t0:.2f}s")

    # ── Phase 2: SCORE ────────────────────────────────────────────────────────
    _t0 = time.time()
    layer_scores = compute_sarathi_scores(
        masks      = masks,
        model      = model,
        model_name = model_name,
        variant    = variant,
        act_rms    = act_rms,
        nmf_rank   = nmf_rank,
        nmf_iters  = nmf_iters,
    )
    logging.info(f"[TIMING] Phase 2 (Score):               {time.time() - _t0:.2f}s")

    # ── Phase 3: SLICE ────────────────────────────────────────────────────────
    _t0 = time.time()
    if not do_obs_reconstruct:
        if is_adaptive:
            stats = sarathi_adaptive_slice(
                layer_scores      = layer_scores,
                model             = model,
                model_name        = model_name,
                compression_ratio = structured_ratio,
                min_keep_ratio    = adaptive_min_keep,
            )
        else:
            stats = sarathi_structured_slice(
                layer_scores      = layer_scores,
                model             = model,
                model_name        = model_name,
                compression_ratio = structured_ratio,
            )
    else:
        # OBS mode: defer physical slicing — OBS needs pre-pruning outputs as targets
        from .model_utils import get_intermediate_size
        orig_size = get_intermediate_size(model_name, model)
        K         = int(orig_size * (1.0 - structured_ratio))
        stats     = {
            "original_size"    : orig_size,
            "new_size"         : "adaptive" if is_adaptive else K,
            "compression_ratio": structured_ratio,
        }
    logging.info(f"[TIMING] Phase 3 (Slice):               {time.time() - _t0:.2f}s")

    # ── Free Phase 1 probe data ───────────────────────────────────────────────
    masks   = None
    act_rms = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 4: ADAPTIVE OBS RECONSTRUCTION (optional) ──────────────────────
    if do_obs_reconstruct:
        if calib_tokens is None:
            raise ValueError(
                "OBS Reconstruction requires calib_tokens. "
                "Pass --obs-reconstruct together with --calib-dataset and --n-calib."
            )

        from .obs_reconstruction import obs_reconstruct
        import numpy as np
        from .thresholding import find_mad_threshold

        _t0 = time.time()

        per_layer_keep_indices = None
        if is_adaptive:
            all_scores = np.concatenate([lns.scores.cpu().numpy() for lns in layer_scores])
            threshold  = find_mad_threshold(all_scores, structured_ratio)
            orig_size  = stats["original_size"]
            min_keep   = max(int(orig_size * adaptive_min_keep), 1)

            per_layer_keep_indices = []
            for lns in layer_scores:
                mask = lns.scores >= threshold
                idx  = mask.nonzero(as_tuple=True)[0]
                if len(idx) < min_keep:
                    _, idx = torch.topk(lns.scores, min_keep, largest=True)
                keep_indices, _ = torch.sort(idx)
                per_layer_keep_indices.append(keep_indices.cpu())

            avg_K = sum(len(x) for x in per_layer_keep_indices) // len(per_layer_keep_indices)
            logging.info(
                f"[SARATHI OBS] Adaptive mode: avg K={avg_K} "
                f"(min={min(len(x) for x in per_layer_keep_indices)}, "
                f"max={max(len(x) for x in per_layer_keep_indices)})"
            )
            stats["adaptive_sizes"] = [len(x) for x in per_layer_keep_indices]
            
        else:
            # Uniform mode: seed OBS with variant-specific top-K ordering.
            # Without this, OBS ignores Wanda/NMF scores and produces identical models!
            orig_size = layer_scores[0].scores.shape[0]
            K = int(orig_size * (1.0 - structured_ratio))
            per_layer_keep_indices = []
            for lns in layer_scores:
                _, top_idx = torch.topk(lns.scores, K, largest=True)
                keep_indices, _ = torch.sort(top_idx)
                per_layer_keep_indices.append(keep_indices.cpu())
            logging.info(f"[Pipeline] Uniform OBS: K={K} neurons/layer kept (variant-score-seeded)")

        device = next(model.parameters()).device
        obs_reconstruct(
            model                  = model,
            model_name             = model_name,
            calib_tokens           = calib_tokens,
            layer_scores           = layer_scores,
            compression_ratio      = structured_ratio,
            device                 = str(device),
            damping                = obs_damping,
            multi_gpu              = multi_gpu,
            per_layer_keep_indices = per_layer_keep_indices,
        )

        logging.info(f"[TIMING] Phase 4 (OBS Reconstruct):    {time.time() - _t0:.2f}s")

    logging.info(f"[TIMING] ─────────────────────────────────────────────────")
    logging.info(f"[TIMING] Total Pipeline Time:           {time.time() - _t_total:.2f}s")
    logging.info("=" * 60)
    logging.info(
        f"[SARATHI] Pipeline Complete — Variant {variant}  |  "
        f"{stats['original_size']} → {stats['new_size']} neurons  |  "
        f"{stats['compression_ratio'] * 100:.1f}% FLOPs reduced"
    )
    logging.info("=" * 60)

    return stats
