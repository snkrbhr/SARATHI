"""
score.py — Phase 2: Neuron-Level Importance Scoring
=====================================================
Converts the unstructured probe mask M ∈ {0,1}^(rows × cols) produced by
probe.py into a per-neuron importance score vector η ∈ ℝ^(intermediate_size).

Four scoring strategies:

  Variant A — SARATHI-Magnitude
    η_i = (1/d_in) Σ_j M_ij          ← row survival rate of the UP projection
    Pure mask aggregation; data-free.

  Variant B — SARATHI-Wanda
    η_i = η_i^(A) × ||act_i||_rms    ← activation-scaled survival rate
    Weights mask signal by the neuron's actual firing magnitude.

  Variant C — SARATHI-NMF (scoring variant)
    Apply NMF to the masked weight matrix W ⊙ M.
    η_i = ||U_i||_F  where W ⊙ M ≈ U @ V^T
    NMF column importance ≈ row-norm of the basis factor U.

  Variant E — SARATHI (main contribution) ★
    Uses NMF Residual Subspace Probe masks for magnitude-weighted scoring.

    Phase 1 (Probe): nmf_residual_probe — data-free NMF residual masks.
      S = |W| − V@H. High residual → NOT explained by low-rank → important.
      Mask: True if S[i,j] >= MAD_threshold.

    Phase 2 (Score): Magnitude-Weighted Survival (SwiGLU-aware).
      η_up_i   = Σ_j (|W_up[i,j]|   × M_up[i,j])
      η_gate_i = Σ_j (|W_gate[i,j]|  × M_gate[i,j])
      η_down_i = Σ_j (|W_down[j,i]|  × M_down[j,i])
      SwiGLU (LLaMA/Mistral): η_i = (η_gate_i + η_up_i + η_down_i) / 3
      GELU/ReLU (OPT):        η_i = (η_up_i + η_down_i) / 2

      SwiGLU-awareness is critical: a neuron fires strongly only when BOTH
      gate AND up have large, high-residual weights. Ignoring gate_proj causes
      post-pruning perplexity collapse on LLaMA architectures.

η_i interpretation (all variants):
  • High η_i → neuron i retained large/important weights → KEEP.
  • Low η_i  → neuron i has mostly low-residual weights → PRUNE.
"""

import logging
from typing import NamedTuple, Literal

import torch
from tqdm import tqdm

from .probe import MaskResult
from .model_utils import get_all_layers, get_mlp_modules
from .nmf import pytorch_nmf


# ── Data structures ───────────────────────────────────────────────────────────

class LayerScore(NamedTuple):
    """Per-layer neuron importance vector produced by Phase 2."""
    layer_idx : int
    layer     : object           # nn.Module reference (decoder layer)
    scores    : torch.Tensor     # [intermediate_size], higher = more important = keep
    variant   : str              # 'A', 'B', 'C', or 'E'


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_by_layer(masks: list) -> dict:
    """Reorganise flat MaskResult list into {layer_idx: {name: MaskResult}}."""
    grouped: dict = {}
    for mr in masks:
        grouped.setdefault(mr.layer_idx, {})[mr.matrix_name] = mr
    return grouped


def _row_survival(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute row-level survival rate η_i = mean(M[i, :]).
    Args:
        mask : [out, in] bool tensor.
    Returns:
        [out] float tensor in [0, 1].
    """
    return mask.float().mean(dim=1)


# ── Variant A: Magnitude probe → row survival ─────────────────────────────────

def score_variant_a(
    masks     : list,
    model,
    model_name: str,
) -> list:
    """
    Variant A — Row survival rate (data-free).

    η_i = mean(M_up[i, :])

    When both gate and up are available (SwiGLU), we average both survival rates.

    Returns:
        List of LayerScore (one per decoder layer).
    """
    logging.info("[SARATHI-A] Computing row-survival scores from magnitude probe ...")
    grouped = _group_by_layer(masks)
    layers  = get_all_layers(model_name, model)
    results = []

    for idx, layer in enumerate(layers):
        lm      = grouped.get(idx, {})
        mr_up   = lm.get("up")
        mr_gate = lm.get("gate")

        if mr_up is None:
            logging.warning(f"[SARATHI-A] Layer {idx}: no 'up' mask — skipping.")
            continue

        eta = _row_survival(mr_up.mask)

        if mr_gate is not None:
            eta = (eta + _row_survival(mr_gate.mask)) / 2.0

        results.append(LayerScore(layer_idx=idx, layer=layer, scores=eta, variant="A"))

    logging.info(f"[SARATHI-A] Scored {len(results)} layers.")
    return results


# ── Variant B: Wanda probe → activation-scaled row survival ──────────────────

def score_variant_b(
    masks     : list,
    model,
    model_name: str,
    act_rms   : dict,
) -> list:
    """
    Variant B — Activation-scaled row survival rate (SARATHI-Wanda).

    η_i^+ = η_i × ||act_i||_rms

    Returns:
        List of LayerScore (one per decoder layer).
    """
    logging.info("[SARATHI-B] Computing activation-weighted row-survival scores ...")
    grouped = _group_by_layer(masks)
    layers  = get_all_layers(model_name, model)
    results = []

    for idx, layer in enumerate(layers):
        lm      = grouped.get(idx, {})
        mr_up   = lm.get("up")
        mr_gate = lm.get("gate")

        if mr_up is None:
            logging.warning(f"[SARATHI-B] Layer {idx}: no 'up' mask — skipping.")
            continue

        eta = _row_survival(mr_up.mask)

        if mr_gate is not None:
            eta = (eta + _row_survival(mr_gate.mask)) / 2.0

        if idx in act_rms:
            rms = act_rms[idx].cpu().float()
            rms = rms / (rms.mean() + 1e-8)
            eta = eta * rms

        eta = eta / (eta.mean() + 1e-8)

        results.append(LayerScore(layer_idx=idx, layer=layer, scores=eta, variant="B"))

    logging.info(f"[SARATHI-B] Scored {len(results)} layers.")
    return results


# ── Variant C: NMF scoring on masked weights ──────────────────────────────────

def score_variant_c(
    masks     : list,
    model,
    model_name: str,
    nmf_rank  : int = 7,
    nmf_iters : int = 100,
) -> list:
    """
    Variant C — NMF on masked weight matrix.

    Factorize W ⊙ M ≈ U @ V^T  (NMF, non-negative)
    η_i = ||U_i||_F  (row norm of the basis factor)

    Returns:
        List of LayerScore (one per decoder layer).
    """
    logging.info(
        f"[SARATHI-C] NMF scoring on masked weights  |  rank={nmf_rank}  iters={nmf_iters}"
    )
    grouped = _group_by_layer(masks)
    layers  = get_all_layers(model_name, model)
    device  = next(model.parameters()).device
    results = []

    for idx, layer in enumerate(tqdm(layers, desc="SARATHI-C NMF scoring")):
        lm      = grouped.get(idx, {})
        mr_up   = lm.get("up")
        mr_down = lm.get("down")

        if mr_up is None or mr_down is None:
            logging.warning(f"[SARATHI-C] Layer {idx}: missing up/down masks — skipping.")
            continue

        with torch.no_grad():
            W_up   = mr_up.weight.data.abs().to(device).float()
            M_up   = mr_up.mask.to(device).float()
            WM_up  = W_up * M_up

            U_up, _ = pytorch_nmf(WM_up, r=nmf_rank, n_iter=nmf_iters)
            scores_up = U_up.norm(dim=1).cpu()

            W_down  = mr_down.weight.data.abs().to(device).float()
            M_down  = mr_down.mask.to(device).float()
            WM_down = (W_down * M_down).T

            U_down, _ = pytorch_nmf(WM_down, r=nmf_rank, n_iter=nmf_iters)
            scores_down = U_down.norm(dim=1).cpu()

        eta = (scores_up + scores_down) / 2.0
        eta = eta / (eta.mean() + 1e-8)

        results.append(LayerScore(layer_idx=idx, layer=layer, scores=eta, variant="C"))

    logging.info(f"[SARATHI-C] Scored {len(results)} layers.")
    return results


# ── Variant E: SARATHI — Magnitude-Weighted Survival, SwiGLU-aware ───────────

def _mag_weighted_survival(mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Magnitude-weighted row survival for input projections (rows = neurons).

    η_i = Σ_j (|W[i,j]| × M[i,j])

    A large weight that survived the mask contributes MORE to the score
    than a tiny weight that survived. This captures true structural importance.

    Args:
        mask   : [out, in] bool tensor — True = kept by NMF residual probe.
        weight : [out, in] float tensor — actual weight matrix.
    Returns:
        [out] float tensor.
    """
    W_abs = weight.detach().float().abs().cpu()
    return (W_abs * mask.float()).sum(dim=1)


def _mag_weighted_survival_col(mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Magnitude-weighted COLUMN survival for the output (down) projection.
    Columns of down_proj correspond to intermediate neurons.

    η_i = Σ_j (|W[j,i]| × M[j,i])

    Args:
        mask   : [out, in] bool tensor — True = kept. Shape is [d_out, N].
        weight : [d_out, N] float tensor.
    Returns:
        [N] float tensor.
    """
    W_abs = weight.detach().float().abs().cpu()
    return (W_abs * mask.float()).sum(dim=0)


def score_variant_e(
    masks     : list,
    model,
    model_name: str,
    nmf_rank  : int = 7,    # accepted for API compat, not used in scoring
    nmf_iters : int = 100,  # accepted for API compat, not used in scoring
) -> list:
    """
    Variant E — SARATHI: Magnitude-Weighted Survival Scoring (SwiGLU-aware).

    Core design choices:

    1. MAGNITUDE WEIGHTING: score = Σ(|W| × M)
       A large weight that survived the NMF residual mask contributes more
       than a tiny weight. This captures structural rather than binary importance.

    2. SWIGLU-AWARENESS (LLaMA-3, LLaMA-2, Mistral):
       FFN neuron i computes: silu(gate[i]) × up[i]
       Both gate AND up must be important for the neuron to fire strongly.
       Score = (η_gate + η_up + η_down) / 3
       For OPT (no gate): Score = (η_up + η_down) / 2

       Ignoring gate_proj causes post-pruning perplexity collapse on LLaMA
       (neurons whose up_proj is important but gate_proj ≈ 0 → output ≈ 0).

    Args:
        masks      : Output of nmf_residual_probe() — gate/up/down masks.
        model      : Loaded HuggingFace causal LM.
        model_name : Architecture identifier string.
        nmf_rank   : Unused (API compat).
        nmf_iters  : Unused (API compat).

    Returns:
        List of LayerScore (variant='E'), one per decoder layer.
    """
    logging.info(
        "[SARATHI-E] Magnitude-Weighted Survival Scoring "
        "(gate+up+down combined, SwiGLU-aware)"
    )

    grouped = _group_by_layer(masks)
    layers  = get_all_layers(model_name, model)
    results = []

    for idx, layer in enumerate(layers):
        lm = grouped.get(idx, {})

        mr_up   = lm.get("up")
        mr_down = lm.get("down")
        mr_gate = lm.get("gate")

        if mr_up is None or mr_down is None:
            logging.warning(f"[SARATHI-E] Layer {idx}: missing up or down mask — skipping.")
            continue

        η_up   = _mag_weighted_survival(mr_up.mask, mr_up.weight)
        η_down = _mag_weighted_survival_col(mr_down.mask, mr_down.weight)

        if mr_gate is not None:
            # SwiGLU architecture (LLaMA, Mistral)
            η_gate = _mag_weighted_survival(mr_gate.mask, mr_gate.weight)
            # Use additive coupling for gate and up as defined in the paper
            eta = (η_gate + η_up + η_down) / 3.0
        else:
            # Non-gated (OPT)
            eta = (η_up + η_down) / 2.0

        eta = eta / (eta.mean() + 1e-8)

        results.append(LayerScore(layer_idx=idx, layer=layer, scores=eta, variant="E"))

    logging.info(f"[SARATHI-E] Scored {len(results)} layers.")
    return results


# ── Unified Phase 2 dispatcher ────────────────────────────────────────────────

def compute_sarathi_scores(
    masks     : list,
    model,
    model_name: str,
    variant   : str = "E",
    act_rms   : dict | None = None,
    nmf_rank  : int = 7,
    nmf_iters : int = 100,
) -> list:
    """
    Unified Phase 2 dispatcher for SARATHI scoring.

    Args:
        masks      : Output of probe.py. For Variant E: nmf_residual_probe().
        model      : Loaded HuggingFace causal LM.
        model_name : Architecture string.
        variant    : 'A' | 'B' | 'C' | 'E'.
                       A = Magnitude probe, row survival (data-free).
                       B = Wanda probe, activation-scaled survival.
                       C = NMF on masked weight matrix.
                       E = SARATHI: magnitude-weighted survival of NMF residual masks,
                           SwiGLU-aware. Data-free. (Recommended)
        act_rms    : Required for variant 'B': dict {layer_idx → [N] act-RMS tensor}.
        nmf_rank   : NMF rank (variants C, pass-through).
        nmf_iters  : NMF iterations (variants C, pass-through).

    Returns:
        List of LayerScore objects, one per decoder layer.
    """
    if variant == "A":
        return score_variant_a(masks, model, model_name)
    elif variant == "B":
        if act_rms is None:
            logging.warning(
                "[SARATHI-B] act_rms not supplied; falling back to Variant A scoring."
            )
            return score_variant_a(masks, model, model_name)
        return score_variant_b(masks, model, model_name, act_rms)
    elif variant == "C":
        return score_variant_c(masks, model, model_name, nmf_rank, nmf_iters)
    elif variant == "E":
        return score_variant_e(masks, model, model_name, nmf_rank, nmf_iters)
    else:
        raise ValueError(f"Unknown SARATHI variant: '{variant}'. Choose A, B, C, or E.")
