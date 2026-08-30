"""
SARATHI: Structured Pruning of LLMs via NMF Residual Subspace Probing
======================================================================
EMNLP 2026.

Method overview:
  1. NMF Residual Subspace Probe  — data-free, weight-geometry-based neuron
     selection grounded in Robust PCA (identifies irreplaceable neurons).
  2. Adaptive OBS Reconstruction  — reconstructs pruned weights layer-locally
     in O(d²) memory, eliminating calibration-data coupling.
  3. Bias Shift Compensation      — zero-overhead correction for post-LayerNorm
     activation shifts introduced by structured removal.

Variants:
  SARATHI      : Data-free NMF Residual Subspace Probe → Adaptive OBS
  SARATHI-Wanda: Wanda-weighted probe (128 samples) for gated architectures

All variants output a PHYSICALLY STRUCTURED model:
  intermediate_size permanently reduced in config,
  weight matrices physically sliced, real FLOPs removed.
"""

from .probe import magnitude_probe, wanda_probe, nmf_residual_probe
from .score import compute_sarathi_scores
from .slice import sarathi_structured_slice

__all__ = [
    "magnitude_probe",
    "wanda_probe",
    "nmf_residual_probe",
    "compute_sarathi_scores",
    "sarathi_structured_slice",
]
