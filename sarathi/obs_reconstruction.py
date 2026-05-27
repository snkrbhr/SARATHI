"""
obs_reconstruction.py — Phase 4: Adaptive OBS Weight Reconstruction
====================================================================
Implements the Adaptive OBS (Optimal Brain Surgeon) weight reconstruction
from SARATHI (Section 3.2 of the paper).

Key properties vs. coupled baselines:
  • O(d²) memory per layer — processes one layer at a time using
    sequential hidden-state propagation (no global Hessian cache).
  • Decoupled from neuron selection — OBS only adjusts surviving weight
    values; the selection is already done by Phase 1+2.
  • Zero calibration-data coupling — OBS calibration is independent of
    the NMF Residual Subspace Probe (no shared data bias).

Bias Shift Compensation (Section 3.3):
  For non-gated architectures (OPT), the FFN has a bias term. After slicing
  neurons, the expected activation shifts. We correct this with a zero-overhead
  Bias Shift Compensation that adjusts the bias to maintain the pre-pruning
  mean output, preventing post-LayerNorm perplexity collapse.

  This reduces PPL from 2,082 → 18.98 on OPT-2.7B at 40% sparsity
  (see Table 3 of the paper).
"""

import logging
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .model_utils import (
    get_all_layers,
    get_intermediate_size,
    set_intermediate_size,
    slice_mlp_layer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_ffn_weights(model_name: str, layer: nn.Module):
    """Return (gate_proj, up_proj, down_proj) for the given layer."""
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral")):
        mlp = layer.mlp
        return mlp.gate_proj, mlp.up_proj, mlp.down_proj
    elif "opt" in mn:
        return layer.fc1, None, layer.fc2
    elif "phi" in mn:
        return layer.mlp.fc1, None, layer.mlp.fc2
    else:
        raise ValueError(f"Architecture not supported by OBS: {model_name}")


def _has_gated_ffn(model_name: str) -> bool:
    mn = model_name.lower()
    return any(k in mn for k in ("llama", "mistral"))


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive OBS: Cholesky least-squares (O(d²) memory per layer)
# ─────────────────────────────────────────────────────────────────────────────

def _cholesky_obs_update(
    X      : torch.Tensor,
    Y      : torch.Tensor,
    K      : int,
    scores : torch.Tensor,
    damping: float = 1e-6,
    device : str   = "cuda",
) -> tuple:
    """
    Adaptive OBS weight reconstruction via Cholesky least-squares.

    Selects the top-K neurons by score, then solves:
        W_new[:, keep] = argmin ||X[:, keep] @ W_new[:, keep]^T - Y||_F²

    Memory: O(d²) for the Hessian of the kept columns only (not full Hessian).

    Args:
        X       : [T×B, d_in]  input activations (intermediate representations).
        Y       : [T×B, d_out] target outputs (original down_proj outputs).
        K       : Number of neurons to keep.
        scores  : [d_in] neuron importance scores from Phase 2.
        damping : Tikhonov regularisation λ for numerical stability.
        device  : CUDA device string.

    Returns:
        (W_new, keep_indices) — reconstructed weight slice and kept neuron indices.
    """
    # Select keep_indices from scores
    _, top_idx   = torch.topk(scores, K, largest=True)
    keep_indices = torch.sort(top_idx).values.to(device)

    X_keep = X[:, keep_indices].to(device, dtype=torch.float32)
    Y_d    = Y.to(device, dtype=torch.float32)

    # Build Hessian H = X_keep^T @ X_keep  (O(K²) — small)
    H = X_keep.t().mm(X_keep)
    damp_val = damping * H.diagonal().mean().clamp(min=1e-8)
    H.diagonal().add_(damp_val)

    XtY = X_keep.t().mm(Y_d)    # [K, d_out]

    try:
        L = torch.linalg.cholesky(H)
        W_new = torch.cholesky_solve(XtY, L).t()   # [d_out, K]
    except torch.linalg.LinAlgError:
        logging.warning("[OBS] Cholesky failed — falling back to lstsq.")
        W_new = torch.linalg.lstsq(X_keep, Y_d).solution.t()

    return W_new.to(dtype=Y.dtype), keep_indices.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def obs_reconstruct(
    model                  : nn.Module,
    model_name             : str,
    calib_tokens           : torch.Tensor,
    layer_scores           : list,
    compression_ratio      : float,
    device                 : str  = "cuda",
    damping                : float = 1e-6,
    batch_size             : int   = 4,
    multi_gpu              : bool  = False,
    per_layer_keep_indices : list  = None,
) -> None:
    """
    Adaptive OBS Weight Reconstruction (Phase 4 of SARATHI).

    Processes one layer at a time using sequential hidden-state propagation.
    This gives O(d²) peak memory per layer rather than O(L×d²) for
    methods that cache all layers' Hessians simultaneously.

    For non-gated architectures (OPT), applies Bias Shift Compensation
    to correct post-LayerNorm activation shifts (Section 3.3 of paper).

    Args:
        model                  : Loaded HuggingFace causal LM (eval mode).
        model_name             : Architecture identifier string.
        calib_tokens           : [n_samples, seq_len] Long tensor of calibration tokens.
        layer_scores           : List of LayerScore from Phase 2.
        compression_ratio      : Fraction of neurons removed (used when
                                 per_layer_keep_indices is None).
        device                 : Primary CUDA device.
        damping                : Tikhonov regularisation λ for OBS Cholesky solve.
        batch_size             : Mini-batch size for forward pass collection.
        multi_gpu              : Whether the model is distributed across GPUs.
        per_layer_keep_indices : Optional list of per-layer keep-index tensors
                                 (for adaptive slicing). If None, uniform K used.
    """
    logging.info("=" * 60)
    logging.info("[SARATHI OBS] Phase 4: Adaptive OBS Weight Reconstruction")
    logging.info(f"[SARATHI OBS]   Calibration tokens: {calib_tokens.shape}")
    logging.info(f"[SARATHI OBS]   Damping λ = {damping}")
    logging.info("=" * 60)

    _t_start        = time.time()
    _t_bias_total   = 0.0
    _t_hessian_total = 0.0
    _t_obs_total    = 0.0

    model.eval()
    layers   = get_all_layers(model_name, model)
    n_layers = len(layers)
    gated    = _has_gated_ffn(model_name)
    mn       = model_name.lower()

    # Determine embedding device for multi-GPU
    if multi_gpu:
        if "opt" in mn:
            embed_device = next(model.model.decoder.embed_tokens.parameters()).device
        else:
            embed_device = next(model.model.embed_tokens.parameters()).device
    else:
        embed_device = torch.device(device)

    # ── Pre-compute embeddings ─────────────────────────────────────────────
    calib_tokens = calib_tokens.to(embed_device)
    hiddens = []
    for i in range(0, len(calib_tokens), batch_size):
        batch = calib_tokens[i : i + batch_size]
        with torch.no_grad():
            if "opt" in mn:
                attn_mask = torch.ones_like(batch)
                tok_emb   = model.model.decoder.embed_tokens(batch)
                pos_emb   = model.model.decoder.embed_positions(attn_mask)
                h         = (tok_emb + pos_emb).detach().cpu()
            else:
                h = model.model.embed_tokens(batch).detach().cpu()
        hiddens.append(h)
    hiddens = torch.cat(hiddens, dim=0)

    # Build score lookup {layer_idx → scores tensor}
    score_map = {lns.layer_idx: lns.scores for lns in layer_scores}

    # ── Layer-by-layer reconstruction ─────────────────────────────────────
    for layer_idx, layer in enumerate(tqdm(layers, desc="[SARATHI OBS] Reconstructing layers")):
        if multi_gpu:
            layer_device = next(layer.parameters()).device
        else:
            layer = layer.to(device)
            layer_device = torch.device(device)
        layer.eval()

        # Determine K for this layer
        if per_layer_keep_indices is not None:
            K            = len(per_layer_keep_indices[layer_idx])
            layer_scores_ = None  # will use precomputed indices
        else:
            orig_size = get_intermediate_size(model_name, model)
            K         = int(orig_size * (1.0 - compression_ratio))
            layer_scores_ = score_map.get(layer_idx)

        # Collect post-attention hidden states for this layer
        post_attn_list = []
        for i in range(0, len(hiddens), batch_size):
            h_batch = hiddens[i : i + batch_size].to(layer_device)
            seq_len = h_batch.shape[1]
            bsz     = h_batch.shape[0]

            with torch.no_grad():
                if "opt" in mn:
                    mask = torch.full(
                        (seq_len, seq_len), float("-inf"),
                        device=layer_device, dtype=h_batch.dtype
                    )
                    mask = torch.triu(mask, diagonal=1)
                    causal_mask = mask.view(1, 1, seq_len, seq_len).expand(bsz, -1, -1, -1)
                    if layer.do_layer_norm_before:
                        normed_h = layer.self_attn_layer_norm(h_batch)
                    else:
                        normed_h = h_batch
                    attn_out = layer.self_attn(normed_h, attention_mask=causal_mask)[0]
                    post_attn = h_batch + attn_out
                    if not layer.do_layer_norm_before:
                        post_attn = layer.self_attn_layer_norm(post_attn)
                else:
                    normed    = layer.input_layernorm(h_batch)
                    pos_ids   = torch.arange(seq_len, dtype=torch.long, device=layer_device).unsqueeze(0).expand(bsz, -1)
                    attn_out  = layer.self_attn(normed, attention_mask=None, position_ids=pos_ids)[0]
                    post_attn = h_batch + attn_out
            post_attn_list.append(post_attn.cpu())

        post_attn_all = torch.cat(post_attn_list, dim=0)
        gate_proj, up_proj, down_proj = _get_ffn_weights(model_name, layer)

        if gated:
            # ── Gated FFN (LLaMA, Mistral) ─────────────────────────────
            down_out_list = []
            X_ffn_list   = []
            _t0 = time.time()

            for i in range(0, len(hiddens), batch_size):
                x_b = post_attn_all[i : i + batch_size].to(layer_device)
                if "opt" not in mn:
                    x_b = layer.post_attention_layernorm(x_b)
                gate_out     = nn.functional.silu(gate_proj(x_b))
                up_out       = up_proj(x_b)
                intermediate = gate_out * up_out
                out          = down_proj(intermediate)
                down_out_list.append(out.detach().cpu())
                X_ffn_list.append(intermediate.detach().cpu())

            _t_hessian_total += time.time() - _t0

            Y_down_flat = torch.cat(down_out_list, dim=0).reshape(-1, down_proj.out_features)
            X_ffn_flat  = torch.cat(X_ffn_list,   dim=0).reshape(-1, gate_proj.out_features)

            _t0 = time.time()
            if per_layer_keep_indices is not None:
                keep_indices = per_layer_keep_indices[layer_idx].to(layer_device)
                X_keep  = X_ffn_flat[:, keep_indices.cpu()].to(layer_device, dtype=torch.float32)
                Y_d     = Y_down_flat.to(layer_device, dtype=torch.float32)
                H       = X_keep.t().mm(X_keep)
                damp_v  = damping * H.diagonal().mean().clamp(min=1e-8)
                H.diagonal().add_(damp_v)
                XtY     = X_keep.t().mm(Y_d)
                try:
                    L       = torch.linalg.cholesky(H)
                    W_new   = torch.cholesky_solve(XtY, L).t().to(dtype=Y_d.dtype)
                except torch.linalg.LinAlgError:
                    W_new = torch.linalg.lstsq(X_keep, Y_d).solution.t().to(dtype=Y_d.dtype)
                keep_indices = keep_indices.cpu()
            else:
                W_new, keep_indices = _cholesky_obs_update(
                    X_ffn_flat, Y_down_flat, K,
                    layer_scores_.cpu(), damping, str(layer_device)
                )
            _t_obs_total += time.time() - _t0

            down_proj.weight.data.copy_(W_new)
            slice_mlp_layer(model_name, layer, keep_indices)

        else:
            # ── Non-gated FFN (OPT) ────────────────────────────────────
            fc1, _, fc2 = gate_proj, up_proj, down_proj

            fc2_out_list = []
            X_ffn_list   = []
            _t0 = time.time()

            for i in range(0, len(hiddens), batch_size):
                x_b = post_attn_all[i : i + batch_size].to(layer_device)
                if layer.do_layer_norm_before:
                    x_b = layer.final_layer_norm(x_b)
                intermediate = nn.functional.relu(fc1(x_b))
                out          = nn.functional.linear(intermediate, fc2.weight)
                fc2_out_list.append(out.detach().cpu())
                X_ffn_list.append(intermediate.detach().cpu())

            _t_hessian_total += time.time() - _t0

            Y_fc2_flat  = torch.cat(fc2_out_list, dim=0).reshape(-1, fc2.out_features)
            X_ffn_flat  = torch.cat(X_ffn_list,   dim=0).reshape(-1, fc1.out_features)

            _t0 = time.time()
            if per_layer_keep_indices is not None:
                keep_indices = per_layer_keep_indices[layer_idx].to(layer_device)
                X_keep  = X_ffn_flat[:, keep_indices.cpu()].to(layer_device, dtype=torch.float32)
                Y_d     = Y_fc2_flat.to(layer_device, dtype=torch.float32)
                H       = X_keep.t().mm(X_keep)
                damp_v  = damping * H.diagonal().mean().clamp(min=1e-8)
                H.diagonal().add_(damp_v)
                XtY     = X_keep.t().mm(Y_d)
                try:
                    L       = torch.linalg.cholesky(H)
                    W_new   = torch.cholesky_solve(XtY, L).t().to(dtype=Y_d.dtype)
                except torch.linalg.LinAlgError:
                    W_new = torch.linalg.lstsq(X_keep, Y_d).solution.t().to(dtype=Y_d.dtype)
                keep_indices = keep_indices.cpu()
            else:
                W_new, keep_indices = _cholesky_obs_update(
                    X_ffn_flat, Y_fc2_flat, K,
                    layer_scores_.cpu(), damping, str(layer_device)
                )
            _t_obs_total += time.time() - _t0

            fc2.weight.data.copy_(W_new)

            # ── Bias Shift Compensation (Section 3.3) ──────────────────
            # Corrects post-LayerNorm activation shift from neuron removal.
            # Zero overhead: only adjusts fc2.bias (one vector add).
            if fc2.bias is not None:
                _t_bias = time.time()
                old_mean  = Y_fc2_flat.to(layer_device).mean(dim=0) + fc2.bias.data
                X_sliced  = X_ffn_flat[:, keep_indices].to(layer_device)
                W_sliced  = W_new[:, keep_indices] if W_new.shape[1] > len(keep_indices) else W_new
                new_mean  = (X_sliced @ W_sliced.t()).mean(dim=0) + fc2.bias.data
                shift     = old_mean - new_mean
                _t_bias_total += time.time() - _t_bias

            slice_mlp_layer(model_name, layer, keep_indices)

            if fc2.bias is not None:
                _, _, fc2_sliced = _get_ffn_weights(model_name, layer)
                fc2_sliced.bias.data.add_(shift)
                if layer_idx == 0 or layer_idx == n_layers - 1:
                    logging.info(
                        f"    [Bias Shift Compensation] Layer {layer_idx} "
                        f"shifted. Norm: {shift.norm().item():.4f}"
                    )

        # ── Propagate hidden states through this reconstructed layer ───
        new_hiddens_list = []
        for i in range(0, len(hiddens), batch_size):
            h_batch = hiddens[i : i + batch_size].to(layer_device)
            seq_len = h_batch.shape[1]
            bsz     = h_batch.shape[0]
            with torch.no_grad():
                if "opt" in mn:
                    mask = torch.full(
                        (seq_len, seq_len), float("-inf"),
                        device=layer_device, dtype=h_batch.dtype
                    )
                    mask = torch.triu(mask, diagonal=1)
                    causal_mask = mask.view(1, 1, seq_len, seq_len).expand(bsz, -1, -1, -1)
                    out = layer(h_batch, attention_mask=causal_mask)[0]
                else:
                    pos_ids = torch.arange(seq_len, dtype=torch.long, device=layer_device).unsqueeze(0).expand(bsz, -1)
                    out = layer(h_batch, attention_mask=None, position_ids=pos_ids)[0]
            new_hiddens_list.append(out.detach().cpu())

        hiddens = torch.cat(new_hiddens_list, dim=0)
        if not multi_gpu:
            layer = layer.cpu()
        torch.cuda.empty_cache()

    # ── Update config ──────────────────────────────────────────────────────
    orig_size = get_intermediate_size(model_name, model)
    if per_layer_keep_indices is not None:
        final_K = min(len(idx) for idx in per_layer_keep_indices)
        logging.info(f"[SARATHI OBS] Adaptive mode: config.intermediate_size set to floor={final_K}")
    else:
        final_K = int(orig_size * (1.0 - compression_ratio))
    set_intermediate_size(model_name, model, final_K)

    _t_end = time.time()
    logging.info(f"[SARATHI OBS] Reconstruction complete. New intermediate_size = {final_K}")
    logging.info(f"[TIMING] OBS Hessian Collection:           {_t_hessian_total:.2f}s")
    logging.info(f"[TIMING] OBS Cholesky Solve (all layers):  {_t_obs_total:.2f}s")
    logging.info(f"[TIMING] Bias Shift Compensation:          {_t_bias_total:.2f}s")
    logging.info(f"[TIMING] OBS Total (incl. forward pass):   {_t_end - _t_start:.2f}s")
