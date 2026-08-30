"""
obs_reconstruction.py — Phase 4: Adaptive OBS Weight Reconstruction
=====================================================================
Implements the Adaptive Optimal Brain Surgeon (OBS) weight reconstruction
step of the SARATHI pipeline.

For each decoder layer, given the SARATHI-selected keep_indices:
  1. Collects calibration activations X (input to FFN) and Y (output of FFN)
     via a forward pass through the original (pre-slice) weights.
  2. Solves the least-squares reconstruction: W_new = argmin ||XW^T - Y||_F
     constrained to the selected neuron subset, using Hessian regularisation.
  3. Physically slices the layer weights to keep_indices.
  4. Optionally applies Bias Shift Compensation to correct post-LayerNorm
     activation mean shifts introduced by structured removal (OPT / Phi-2).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .model_utils import get_all_layers, get_mlp_modules, get_intermediate_size, set_intermediate_size, slice_mlp_layer

def _get_ffn_weights(model_name: str, layer):
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral", "llama3")):
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


# ---------------------------------------------------------------------------
# OLD Iterative Greedy Refinement (ORIGINAL — replaced by Cholesky solver)
# ---------------------------------------------------------------------------

def _iterative_greedy_update(
    X: torch.Tensor,
    Y: torch.Tensor,
    K: int,
    damping: float = 1e-6,
    device: str = "cuda",
    forced_keep_indices: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Implements Local Greedy Refinement (Iterative OBS).
    Starts with full dense weights W, computes exact inverse Hessian H^{-1},
    and iteratively prunes columns of W that result in the smallest reconstruction error.
    Returns:
        W_pruned: Dense weight matrix with pruned columns set to 0.
        keep_indices: 1D tensor of indices that were kept.

    WARNING: This is extremely slow for large FFN dimensions (e.g. OPT-13B d_ffn=20480).
    It will hang/deadlock due to CPU-GPU sync bottleneck in the greedy loop.
    """
    d_in = X.shape[1]
    d_out = Y.shape[1]

    H = torch.zeros((d_in, d_in), device=device, dtype=torch.float32)
    XtY = torch.zeros((d_in, d_out), device=device, dtype=torch.float32)

    chunk_size = 16384
    for i in range(0, X.shape[0], chunk_size):
        X_chunk = X[i:i+chunk_size].to(device=device, dtype=torch.float32)
        Y_chunk = Y[i:i+chunk_size].to(device=device, dtype=torch.float32)
        H += X_chunk.t() @ X_chunk
        XtY += X_chunk.t() @ Y_chunk

    damp_val = damping * torch.diag(H).mean().clamp(min=1e-8)
    H.diagonal().add_(damp_val)

    try:
        H_inv = torch.linalg.inv(H)
    except torch.linalg.LinAlgError:
        logging.warning("Dense Hessian inversion failed. Falling back to pinv.")
        H_inv = torch.linalg.pinv(H)

    W = (H_inv @ XtY).t()  # [d_out, d_in]

    alive_mask = torch.ones(d_in, dtype=torch.bool, device=device)
    num_to_prune = d_in - K

    if forced_keep_indices is not None:
        keep_mask = torch.zeros(d_in, dtype=torch.bool, device=device)
        keep_mask[forced_keep_indices] = True
        prune_order = (~keep_mask).nonzero(as_tuple=True)[0]
    else:
        # === Pre-rank pruning order ONCE (1 sync) instead of argmin per step ===
        diag_H_init = H_inv.diagonal().clone()
        W_norms_init = (W ** 2).sum(dim=0)
        E_init = W_norms_init / diag_H_init.clamp(min=1e-8)
        E_init[~alive_mask] = float('inf')
        prune_order = torch.argsort(E_init)[:num_to_prune]  # ONE sync total

    for idx, k in enumerate(prune_order):
        k_int = k.item()
        if not alive_mask[k_int]:
            continue
        alive_mask[k_int] = False

        h_k = H_inv[:, k_int].clone()
        h_kk = h_k[k_int].clone()
        w_k = W[:, k_int].clone()

        W.addr_(w_k, h_k, alpha=-1.0 / h_kk)
        H_inv.addr_(h_k, h_k, alpha=-1.0 / h_kk)

        W[:, k_int] = 0.0
        H_inv[:, k_int] = 0.0
        H_inv[k_int, :] = 0.0
        
        # Prevent CUDA launch queue deadlock
        if idx % 50 == 0:
            torch.cuda.synchronize(device)

    keep_indices = alive_mask.nonzero(as_tuple=True)[0]
    return W.to(dtype=Y.dtype), keep_indices


@torch.no_grad()
def obs_reconstruct(
    model,
    model_name: str,
    calib_tokens: torch.Tensor,
    layer_scores: list,
    compression_ratio: float,
    device: str = "cuda",
    damping: float = 1e-6,
    batch_size: int = 4,
    multi_gpu: bool = False,
    per_layer_keep_indices: list = None,
) -> None:
    logging.info("=" * 60)
    logging.info("[OBS] Phase 4: Iterative Greedy Weight Reconstruction")
    logging.info(f"[OBS]   Calibration tokens: {calib_tokens.shape}")
    logging.info(f"[OBS]   Damping λ = {damping}")
    logging.info("=" * 60)

    _t_obs_global_start = time.time()
    _t_bias_total = 0.0
    _t_hessian_total = 0.0
    _t_greedy_total = 0.0

    model.eval()
    layers = get_all_layers(model_name, model)
    n_layers = len(layers)
    gated = _has_gated_ffn(model_name)
    mn = model_name.lower()

    if multi_gpu:
        if "opt" in mn:
            embed_device = next(model.model.decoder.embed_tokens.parameters()).device
        else:
            embed_device = next(model.model.embed_tokens.parameters()).device
    else:
        embed_device = torch.device(device)

    calib_tokens = calib_tokens.to(embed_device)
    hiddens = []
    for i in range(0, len(calib_tokens), batch_size):
        batch = calib_tokens[i : i + batch_size]
        with torch.no_grad():
            if "opt" in mn:
                attn_mask = torch.ones_like(batch)
                tok_emb = model.model.decoder.embed_tokens(batch)
                pos_emb = model.model.decoder.embed_positions(attn_mask)
                h = (tok_emb + pos_emb).detach().cpu()
            else:
                h = model.model.embed_tokens(batch).detach().cpu()
        hiddens.append(h)
    hiddens = torch.cat(hiddens, dim=0)

    for layer_idx, layer in enumerate(tqdm(layers, desc="[OBS] Reconstructing layers")):
        if multi_gpu:
            layer_device = next(layer.parameters()).device
        else:
            layer = layer.to(device)
            layer_device = torch.device(device)
        layer.eval()

        post_attn_list = []
        for i in range(0, len(hiddens), batch_size):
            h_batch = hiddens[i : i + batch_size].to(layer_device)
            seq_len = h_batch.shape[1]
            bsz = h_batch.shape[0]

            with torch.no_grad():
                if "opt" in mn:
                    mask = torch.full((seq_len, seq_len), float("-inf"), device=layer_device, dtype=h_batch.dtype)
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
                    normed = layer.input_layernorm(h_batch)
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=layer_device).unsqueeze(0).expand(bsz, -1)
                    if hasattr(layer.self_attn, "forward"):
                        attn_out = layer.self_attn(normed, attention_mask=None, position_ids=position_ids)[0]
                    post_attn = h_batch + attn_out
            post_attn_list.append(post_attn.cpu())

        post_attn_all = torch.cat(post_attn_list, dim=0)

        if per_layer_keep_indices is not None:
            forced_indices = per_layer_keep_indices[layer_idx].to(layer_device)
            K = len(forced_indices)
        else:
            forced_indices = None
            orig_size = get_intermediate_size(model_name, model)
            K = int(orig_size * (1.0 - compression_ratio))

        gate_proj, up_proj, down_proj = _get_ffn_weights(model_name, layer)

        if gated:
            down_out_list = []
            X_ffn_list = []
            _t_hessian_layer = time.time()
            for i in range(0, len(hiddens), batch_size):
                x_b = post_attn_all[i : i + batch_size].to(layer_device)
                if "opt" not in mn:
                    x_b = layer.post_attention_layernorm(x_b)
                gate_out = nn.functional.silu(gate_proj(x_b))
                up_out   = up_proj(x_b)
                intermediate = gate_out * up_out
                out = down_proj(intermediate)
                down_out_list.append(out.detach().cpu())
                X_ffn_list.append(intermediate.detach().cpu())

            Y_down_flat = torch.cat(down_out_list, dim=0).reshape(-1, down_proj.out_features)
            X_ffn_flat = torch.cat(X_ffn_list, dim=0).reshape(-1, gate_proj.out_features)

            _t_hessian_total += time.time() - _t_hessian_layer
            _t_greedy_layer = time.time()
            W_down_new, keep_indices = _iterative_greedy_update(X_ffn_flat, Y_down_flat, K, damping, str(layer_device), forced_keep_indices=forced_indices)
            _t_greedy_total += time.time() - _t_greedy_layer
            keep_indices = keep_indices.cpu()
            down_proj.weight.data.copy_(W_down_new)
            slice_mlp_layer(model_name, layer, keep_indices)

        else:
            fc1, _, fc2 = gate_proj, up_proj, down_proj

            fc2_out_list = []
            X_ffn_list = []
            _t_hessian_layer = time.time()
            for i in range(0, len(hiddens), batch_size):
                x_b = post_attn_all[i : i + batch_size].to(layer_device)
                do_prenorm = getattr(layer, "do_layer_norm_before", True)
                if do_prenorm:
                    ln = getattr(layer, "final_layer_norm", None) or getattr(layer, "layer_norm", None)
                    if ln is not None:
                        x_b = ln(x_b)
                intermediate = nn.functional.relu(fc1(x_b))
                out = nn.functional.linear(intermediate, fc2.weight)
                fc2_out_list.append(out.detach().cpu())
                X_ffn_list.append(intermediate.detach().cpu())

            Y_fc2_flat = torch.cat(fc2_out_list, dim=0).reshape(-1, fc2.out_features)
            X_ffn_flat = torch.cat(X_ffn_list, dim=0).reshape(-1, fc1.out_features)

            _t_hessian_total += time.time() - _t_hessian_layer
            _t_greedy_layer = time.time()
            W_fc2_new, keep_indices = _iterative_greedy_update(X_ffn_flat, Y_fc2_flat, K, damping, str(layer_device), forced_keep_indices=forced_indices)
            _t_greedy_total += time.time() - _t_greedy_layer
            keep_indices = keep_indices.cpu()
            fc2.weight.data.copy_(W_fc2_new)

            if fc2.bias is not None:
                _t_bias_layer = time.time()
                old_mean = Y_fc2_flat.to(layer_device).mean(dim=0) + fc2.bias.data
                X_sliced = X_ffn_flat[:, keep_indices].to(layer_device)
                W_sliced = W_fc2_new[:, keep_indices]
                new_mean = (X_sliced @ W_sliced.t()).mean(dim=0) + fc2.bias.data
                shift = old_mean - new_mean
                _t_bias_total += time.time() - _t_bias_layer

            slice_mlp_layer(model_name, layer, keep_indices)

            if fc2.bias is not None:
                _, _, fc2_sliced = _get_ffn_weights(model_name, layer)
                fc2_sliced.bias.data.add_(shift)
                if layer_idx == 0 or layer_idx == n_layers - 1:
                    logging.info(f"    [Bias Shift] Layer {layer_idx} shifted. Norm: {shift.norm().item():.4f}")

        new_hiddens_list = []
        for i in range(0, len(hiddens), batch_size):
            h_batch = hiddens[i : i + batch_size].to(layer_device)
            seq_len = h_batch.shape[1]
            bsz = h_batch.shape[0]
            with torch.no_grad():
                if "opt" in mn:
                    mask = torch.full((seq_len, seq_len), float("-inf"), device=layer_device, dtype=h_batch.dtype)
                    mask = torch.triu(mask, diagonal=1)
                    causal_mask = mask.view(1, 1, seq_len, seq_len).expand(bsz, -1, -1, -1)
                    out = layer(h_batch, attention_mask=causal_mask)[0]
                else:
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=layer_device).unsqueeze(0).expand(bsz, -1)
                    out = layer(h_batch, attention_mask=None, position_ids=position_ids)[0]
            new_hiddens_list.append(out.detach().cpu())

        hiddens = torch.cat(new_hiddens_list, dim=0)
        if not multi_gpu:
            layer = layer.cpu()
        torch.cuda.empty_cache()

    orig_size = get_intermediate_size(model_name, model)
    if per_layer_keep_indices is not None:
        final_K = min(len(idx) for idx in per_layer_keep_indices)
        logging.info(f"[OBS] Adaptive mode: config.intermediate_size set to floor={final_K}")
    else:
        final_K = int(orig_size * (1.0 - compression_ratio))
    set_intermediate_size(model_name, model, final_K)

    _t_obs_global_end = time.time()
    logging.info(f"[OBS] Weight reconstruction complete. New intermediate_size = {final_K}")
    logging.info(f"[TIMING] OBS Hessian Collection:       {_t_hessian_total:.2f}s")
    logging.info(f"[TIMING] OBS Greedy Update (all layers):{_t_greedy_total:.2f}s")
    logging.info(f"[TIMING] Bias Shift Invariance:        {_t_bias_total:.2f}s")
    logging.info(f"[TIMING] OBS Total (incl. forward pass):{_t_obs_global_end - _t_obs_global_start:.2f}s")
