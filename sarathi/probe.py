"""
probe.py — Phase 1: Neuron Selection Probes
============================================
SARATHI Phase 1: Apply a probe to obtain a binary mask
M ∈ {0,1}^(rows × cols) per MLP weight matrix.

The mask is NOT applied to the model weights — it is only used downstream in
Phase 2 (score.py) to compute the row-level neuron importance scores.

Three probe methods:

  magnitude_probe      — Data-free, uses |W| directly.
                         (Variant A: SARATHI-Magnitude)

  wanda_probe          — Activation-weighted, uses ||act_i||₂ × |W_ij|.
                         (Variant B: SARATHI-Wanda)
                         Requires ~128 calibration forward passes.

  nmf_residual_probe   — Data-free NMF Residual Subspace Probe.
                         (Variant E: SARATHI main — core contribution)
                         Computes S = |W| − V@H (NMF residual), masks
                         weights NOT explained by the low-rank structure.
                         Grounded in Robust PCA theory.

Public API
----------
  masks = magnitude_probe(model, model_name, sigma)
  masks, act_rms = wanda_probe(model, model_name, tokenizer, sigma, n_calib)
  masks = nmf_residual_probe(model, model_name, sigma, nmf_rank, nmf_iters)

All return a list of MaskResult named-tuples (one per MLP weight matrix per layer).
"""

import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import NamedTuple

from .model_utils import get_all_layers, get_mlp_modules, load_calib_tokens
from .nmf import pytorch_nmf
from .thresholding import find_mad_threshold
import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

class MaskResult(NamedTuple):
    """One mask per MLP weight matrix."""
    layer_idx  : int
    matrix_name: str           # 'gate', 'up', or 'down'
    weight     : torch.Tensor  # reference to the live parameter (NOT copied)
    mask       : torch.Tensor  # bool tensor, same shape as weight. True = kept.


# ── Helpers ───────────────────────────────────────────────────────────────────

def _topk_mask(scores: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Return a bool mask where the bottom (sigma * numel) entries are False
    (the entries that would be pruned).

    Args:
        scores : Any-shape float tensor (higher = more important).
        sigma  : Probe sparsity fraction in (0, 1).

    Returns:
        mask : bool tensor of same shape. True = keep, False = prune.
    """
    flat    = scores.view(-1)
    n_prune = max(int(sigma * flat.numel()), 0)
    if n_prune == 0:
        return torch.ones_like(flat, dtype=torch.bool).view(scores.shape)
    threshold = flat.kthvalue(n_prune).values
    mask_flat = flat > threshold
    return mask_flat.view(scores.shape)


# ── Probe A: Magnitude (data-free) ───────────────────────────────────────────

def magnitude_probe(
    model     : nn.Module,
    model_name: str,
    sigma     : float = 0.10,
) -> list:
    """
    Data-free unstructured magnitude probe.

    Score = |W_ij|  (larger = more important).
    Mask  = top (1 - σ) entries by magnitude.

    Args:
        model      : Loaded HuggingFace causal LM (eval mode).
        model_name : Architecture identifier string.
        sigma      : Probe sparsity; fraction of weights to probe-prune per matrix.
                     Recommended: 0.10 – 0.15.

    Returns:
        List of MaskResult, one per (layer, matrix).
    """
    logging.info(f"[SARATHI] Magnitude probe  |  σ = {sigma:.2%}  (data-free)")
    layers  = get_all_layers(model_name, model)
    results = []

    for idx, layer in enumerate(tqdm(layers, desc="Magnitude probe")):
        modules = get_mlp_modules(model_name, layer)
        for name, linear in modules.items():
            if linear is None:
                continue
            with torch.no_grad():
                scores = linear.weight.data.abs().cpu().float()
                mask   = _topk_mask(scores, sigma)
            results.append(MaskResult(
                layer_idx   = idx,
                matrix_name = name,
                weight      = linear.weight,
                mask        = mask,
            ))

    logging.info(
        f"[SARATHI] Magnitude probe done — {len(results)} matrices masked  "
        f"(target σ = {sigma:.2%}  |  actual avg survival = "
        f"{sum(r.mask.float().mean().item() for r in results) / len(results):.3f})"
    )
    return results


# ── Probe B: Wanda activation-weighted ───────────────────────────────────────

class _DownProjPreHook:
    """
    Forward pre-hook on down_proj (or fc2) to capture neuron activations
    (the tensor flowing INTO the down projection).
    Accumulates squared norms for RMS computation.
    """
    def __init__(self, n_neurons: int):
        self.sq_sum = torch.zeros(n_neurons)
        self.count  = 0

    def __call__(self, module, inp):
        act = inp[0].detach().float().cpu()   # [B, T, N]
        self.sq_sum += act.pow(2).sum(dim=(0, 1))
        self.count  += act.shape[0] * act.shape[1]


def wanda_probe(
    model        : nn.Module,
    model_name   : str,
    tokenizer,
    sigma        : float = 0.10,
    n_calib      : int   = 128,
    seq_len      : int   = 512,
    calib_dataset: str   = "wikitext",
) -> tuple:
    """
    Wanda-style activation-weighted unstructured probe (SARATHI-Wanda variant).

    Score_ij = ||act_i||_rms × |W_ij|
    Mask     = top (1 - σ) entries by this score per matrix.

    Used in SARATHI-Wanda for gated architectures where 128-sample calibration
    further improves zero-shot accuracy (Section 4.4 of paper).

    Args:
        model      : Loaded HuggingFace causal LM (eval mode).
        model_name : Architecture identifier string.
        tokenizer  : Corresponding tokenizer.
        sigma      : Probe sparsity fraction.
        n_calib    : Number of calibration samples.
        seq_len    : Sequence length per calibration chunk.
        calib_dataset : 'wikitext' | 'c4' | 'alpaca'.

    Returns:
        Tuple of:
          - List of MaskResult, one per (layer, matrix).
          - dict {layer_idx → [N] activation RMS tensor} for downstream scoring.
    """
    logging.info(f"[SARATHI] Wanda probe  |  σ = {sigma:.2%}  |  n_calib = {n_calib}")
    device = next(model.parameters()).device
    layers = get_all_layers(model_name, model)

    # Step 1: Register forward pre-hooks on down_proj to capture activations
    hook_registry: dict = {}
    handles = []

    for idx, layer in enumerate(layers):
        modules = get_mlp_modules(model_name, layer)
        n_inter = modules["up"].out_features
        hook    = _DownProjPreHook(n_inter)
        handle  = modules["down"].register_forward_pre_hook(hook)
        hook_registry[idx] = hook
        handles.append(handle)

    # Step 2: Run calibration forward passes
    logging.info(f"[SARATHI] Running {n_calib} calibration forward passes (dataset={calib_dataset}) ...")
    calib = load_calib_tokens(tokenizer, dataset=calib_dataset, n_samples=n_calib, seq_len=seq_len)
    calib = calib.to(device)

    model.eval()
    with torch.no_grad():
        for chunk in tqdm(calib, desc="Calibration forward passes"):
            model(input_ids=chunk.unsqueeze(0))

    for h in handles:
        h.remove()

    # Step 3: Build activation RMS per neuron per layer
    act_rms: dict = {}
    for idx, hook in hook_registry.items():
        act_rms[idx] = (hook.sq_sum / max(hook.count, 1)).sqrt()

    # Step 4: Compute per-weight Wanda score and mask
    results = []
    for idx, layer in enumerate(tqdm(layers, desc="Wanda scoring")):
        modules = get_mlp_modules(model_name, layer)
        rms     = act_rms[idx]  # [N]

        for name, linear in modules.items():
            if linear is None:
                continue
            with torch.no_grad():
                W_abs = linear.weight.data.abs().cpu().float()  # [out, in]

                if name in ("gate", "up"):
                    score = W_abs * rms.unsqueeze(1)            # [N, in]
                else:
                    score = W_abs * rms.unsqueeze(0)            # [out, N]

                mask = _topk_mask(score, sigma)

            results.append(MaskResult(
                layer_idx   = idx,
                matrix_name = name,
                weight      = linear.weight,
                mask        = mask,
            ))

    logging.info(
        f"[SARATHI] Wanda probe done — {len(results)} matrices masked  "
        f"(avg survival = "
        f"{sum(r.mask.float().mean().item() for r in results) / len(results):.3f})"
    )
    return results, act_rms


# ── Probe E: NMF Residual Subspace Probe (core SARATHI contribution) ──────────

class _WeightScore:
    """Internal holder for per-matrix NMF residual scores."""
    __slots__ = ("name", "weight", "score")

    def __init__(self, name: str, weight: torch.Tensor, score: torch.Tensor):
        self.name   = name
        self.weight = weight
        self.score  = score


def _compute_nmf_residual_scores(
    model      : nn.Module,
    model_name : str,
    nmf_rank   : int,
    nmf_iters  : int,
) -> list:
    """
    Compute NMF Residual score S = |W| − V@H for every MLP weight matrix.

    The residual captures weight components NOT explained by the low-rank
    structure V@H. Neurons with high residual scores are geometrically
    irreplaceable (Robust PCA grounding, Section 3.1 of paper).

    Args:
        model      : Loaded HuggingFace causal LM.
        model_name : Architecture identifier string.
        nmf_rank   : NMF factorization rank r (default 7).
        nmf_iters  : NMF multiplicative update iterations (default 100).

    Returns:
        List of _WeightScore, one per (layer, matrix_name).
    """
    device = next(model.parameters()).device
    layers = get_all_layers(model_name, model)
    results = []

    for idx, layer in enumerate(tqdm(layers, desc="[SARATHI] NMF residual scoring")):
        modules = get_mlp_modules(model_name, layer)
        for name, linear in modules.items():
            if linear is None:
                continue
            with torch.no_grad():
                W_abs = linear.weight.abs().to(device).float()
                V, H  = pytorch_nmf(W_abs, r=nmf_rank, n_iter=nmf_iters)
                S     = (W_abs - torch.matmul(V, H)).cpu()
            results.append(_WeightScore(
                name   = f"layer_{idx}_{name}",
                weight = linear.weight,
                score  = S,
            ))

    logging.info(f"[SARATHI] NMF residual scoring done — {len(results)} matrices scored.")
    return results


def nmf_residual_probe(
    model      : nn.Module,
    model_name : str,
    sigma      : float = 0.10,
    nmf_rank   : int   = 7,
    nmf_iters  : int   = 100,
) -> list:
    """
    NMF Residual Subspace Probe — data-free, calibration-free (SARATHI core).

    Three-pass pipeline:
      Pass 1 — Score every MLP weight matrix: S = |W| − V@H  (NMF residual)
      Pass 2 — Flatten all scores; find Global MAD threshold for target σ
      Pass 3 — Build boolean mask (True = keep, S >= threshold)

    Weights with HIGH residual score are NOT explained by the low-rank
    structure → they are geometrically irreplaceable → marked as important.
    This is grounded in Robust PCA: the residual isolates the sparse,
    structurally-significant component from the low-rank background.

    Args:
        model      : Loaded HuggingFace causal LM (eval mode, weights unmodified).
        model_name : Architecture identifier string.
        sigma      : Target probe sparsity fraction (recommended: 0.10).
        nmf_rank   : NMF factorization rank r (default 7).
        nmf_iters  : NMF multiplicative update iterations (default 100).

    Returns:
        List of MaskResult, one per (layer, matrix). True = kept (high residual).
    """
    logging.info("=" * 60)
    logging.info("[SARATHI] NMF Residual Subspace Probe (data-free)")
    logging.info(f"  sigma      = {sigma:.2%}  (probe sparsity)")
    logging.info(f"  nmf_rank   = {nmf_rank}")
    logging.info(f"  nmf_iters  = {nmf_iters}")
    logging.info("=" * 60)

    # Pass 1: NMF residual scoring
    logging.info("[SARATHI Probe] Pass 1: Computing per-weight NMF residual scores ...")
    weight_scores = _compute_nmf_residual_scores(model, model_name, nmf_rank, nmf_iters)

    # Pass 2: Global MAD threshold
    logging.info("[SARATHI Probe] Pass 2: Searching for Global MAD threshold ...")
    flat_scores = np.concatenate(
        [ws.score.numpy().flatten() for ws in weight_scores]
    )
    threshold = find_mad_threshold(flat_scores, sigma)

    # Pass 3: Build masks (no weight modification)
    logging.info(f"[SARATHI Probe] Pass 3: Building masks (threshold={threshold:.6f}) ...")
    results = []
    total_w    = 0
    total_kept = 0

    for ws in tqdm(weight_scores, desc="[SARATHI] Building masks"):
        S_np      = ws.score.numpy()
        keep_mask = torch.tensor(S_np >= threshold, dtype=torch.bool)

        parts       = ws.name.split("_", maxsplit=2)   # ['layer', idx, matrix_name]
        layer_idx   = int(parts[1])
        matrix_name = parts[2]

        results.append(MaskResult(
            layer_idx   = layer_idx,
            matrix_name = matrix_name,
            weight      = ws.weight,
            mask        = keep_mask,
        ))
        total_kept += int(keep_mask.sum())
        total_w    += keep_mask.numel()

    actual_sparsity = 1.0 - (total_kept / total_w) if total_w > 0 else 0.0
    logging.info(
        f"[SARATHI Probe] Done — kept {total_kept:,}/{total_w:,} weights "
        f"(effective probe sparsity = {actual_sparsity * 100:.2f}%)"
    )
    return results
