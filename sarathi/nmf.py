"""
nmf.py — Pure PyTorch Non-negative Matrix Factorization
========================================================
Multiplicative Update (MU) rules for NMF.
  V ≈ W @ H,  where V, W, H ≥ 0

Used in:
  - NMF Residual Subspace Probe (Phase 1): S = |W| - V@H
  - NMF-based neuron scoring (Variant C): factorize W ⊙ M
"""

import torch


def pytorch_nmf(
    V: torch.Tensor,
    r: int,
    n_iter: int = 100,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Nonnegative Matrix Factorization via Multiplicative Update rules.

    Args:
        V      : Non-negative input matrix of shape (n, m). Should be |W| (abs weights).
        r      : Factorization rank.
        n_iter : Number of update iterations.
        eps    : Small constant for numerical stability.

    Returns:
        W : Factor matrix of shape (n, r).
        H : Factor matrix of shape (r, m).
    """
    if V.ndim != 2:
        raise ValueError(f"pytorch_nmf expects a 2-D tensor, got shape {V.shape}")

    n, m = V.shape
    device, orig_dtype = V.device, V.dtype

    # Work in float32 for numerical stability
    V_f32 = V.to(torch.float32)
    W = torch.rand(n, r, device=device, dtype=torch.float32)
    H = torch.rand(r, m, device=device, dtype=torch.float32)

    for _ in range(n_iter):
        # Update H
        num_H = torch.matmul(W.t(), V_f32)
        den_H = torch.matmul(W.t(), torch.matmul(W, H)) + eps
        H = H * (num_H / den_H)

        # Update W
        num_W = torch.matmul(V_f32, H.t())
        den_W = torch.matmul(W, torch.matmul(H, H.t())) + eps
        W = W * (num_W / den_W)

    return W.to(orig_dtype), H.to(orig_dtype)
