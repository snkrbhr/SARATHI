"""
sarathi_main.py — CLI Entry Point for SARATHI Pruning
======================================================
SARATHI: Structured Pruning of LLMs via NMF Residual Subspace Probing
EMNLP 2026 Anonymous Submission.

Usage:

  # SARATHI (data-free, recommended):
  python sarathi_main.py \\
      --model meta-llama/Llama-3-8B \\
      --variant E \\
      --structured-ratio 0.25 \\
      --obs-reconstruct \\
      --save-dir ./pruned_models

  # SARATHI-Wanda (128-sample calibration, for gated architectures):
  python sarathi_main.py \\
      --model meta-llama/Llama-3-8B \\
      --variant B \\
      --structured-ratio 0.25 \\
      --n-calib 128 \\
      --calib-dataset wikitext \\
      --obs-reconstruct \\
      --save-dir ./pruned_models

  # OPT-2.7B at 40% sparsity (only SARATHI succeeds — baselines OOM):
  python sarathi_main.py \\
      --model facebook/opt-2.7b \\
      --variant E \\
      --structured-ratio 0.40 \\
      --obs-reconstruct \\
      --calib-dataset c4 \\
      --save-dir ./pruned_models

NOTE: All variants produce a STRUCTURED output — intermediate_size is
permanently reduced in model.config → real FLOPs removed.

Environment variables for dataset paths (set once on your cluster):
  SARATHI_WIKITEXT_PATH  — path to wikitext-train.arrow or .parquet
  SARATHI_ALPACA_PATH    — path to alpaca_data.json
  SARATHI_C4_PATH        — path to c4_calibration_subset.json
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import os
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch

from sarathi.model_utils import load_model_and_tokenizer, load_calib_tokens
from sarathi.pipeline import apply_sarathi_pruning


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description     = "SARATHI: Structured Pruning of LLMs via NMF Residual Subspace Probing",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or local path")

    # ── Variant ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "E"],
        default="E",
        help=(
            "SARATHI variant:\n"
            "  E = SARATHI        : NMF Residual Subspace Probe (data-free) [default]\n"
            "  B = SARATHI-Wanda  : Wanda probe (128 samples, gated architectures)\n"
            "  A = Magnitude      : Data-free magnitude probe baseline\n"
            "  C = NMF-Scoring    : NMF on Wanda-masked weights"
        ),
    )

    # ── Pruning config ─────────────────────────────────────────────────────────
    parser.add_argument("--structured-ratio", type=float, default=0.25,
                        help="Fraction of FFN neurons to permanently remove (e.g. 0.25 = 25%%)")
    parser.add_argument("--probe-sigma", type=float, default=0.10,
                        help="Unstructured probe sparsity σ for Phase 1")
    parser.add_argument("--nmf-rank", type=int, default=7,
                        help="NMF factorization rank r (Variants C and E)")
    parser.add_argument("--nmf-iters", type=int, default=100,
                        help="NMF multiplicative update iterations (Variants C and E)")

    # ── Calibration ────────────────────────────────────────────────────────────
    parser.add_argument("--n-calib", type=int, default=128,
                        help="Calibration samples (Variants B, C, and OBS)")
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "c4", "alpaca"],
        help="Dataset for calibration (Variants B/C and OBS Reconstruction).",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=2048,
        help="Sequence length for calibration data (default=2048, matches SoBP protocol).",
    )

    # ── OBS Reconstruction ────────────────────────────────────────────────────
    parser.add_argument(
        "--obs-reconstruct",
        action="store_true",
        help="Enable Adaptive OBS Weight Reconstruction after pruning.",
    )
    parser.add_argument(
        "--obs-damping",
        type=float,
        default=1e-6,
        help="Tikhonov regularisation λ for OBS Cholesky solve.",
    )

    # ── Adaptive slicing ───────────────────────────────────────────────────────
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use Adaptive Structured Slicing (Global MAD threshold, variable K/layer).",
    )
    parser.add_argument(
        "--min-keep", type=float, default=0.5,
        help="Minimum fraction of neurons to keep per layer in adaptive mode.",
    )

    # ── Infrastructure ─────────────────────────────────────────────────────────
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save the pruned model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Load model with device_map=auto across all visible GPUs (13B+ models). "
             "Set CUDA_VISIBLE_DEVICES before launching.",
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[SARATHI] Using device: {device}")
    if args.multi_gpu:
        logging.info("[SARATHI] Multi-GPU: loading with device_map=auto.")

    # Load model
    need_tokenizer = args.variant in ("B", "C") or args.obs_reconstruct
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device     = device,
        device_map = "auto" if args.multi_gpu else None,
    )

    # Pre-load calibration tokens for OBS Reconstruction
    calib_tokens = None
    if args.obs_reconstruct:
        if tokenizer is None:
            raise ValueError("OBS Reconstruction requires a tokenizer.")
        logging.info(
            f"[SARATHI] Loading {args.n_calib} calibration samples "
            f"from '{args.calib_dataset}' (seq_len={args.calib_seq_len}) ..."
        )
        calib_tokens = load_calib_tokens(
            tokenizer,
            dataset   = args.calib_dataset,
            n_samples = args.n_calib,
            seq_len   = args.calib_seq_len,
        )
        logging.info(f"[SARATHI] Calibration tokens ready: {calib_tokens.shape}")

    # Run SARATHI pipeline
    stats = apply_sarathi_pruning(
        model             = model,
        model_name        = args.model,
        tokenizer         = tokenizer if need_tokenizer else None,
        variant           = args.variant,
        structured_ratio  = args.structured_ratio,
        probe_sigma       = args.probe_sigma,
        n_calib           = args.n_calib,
        calib_dataset     = args.calib_dataset,
        nmf_rank          = args.nmf_rank,
        nmf_iters         = args.nmf_iters,
        is_adaptive       = args.adaptive,
        adaptive_min_keep = args.min_keep,
        do_obs_reconstruct= args.obs_reconstruct,
        calib_tokens      = calib_tokens,
        obs_damping       = args.obs_damping,
        multi_gpu         = args.multi_gpu,
    )

    # Save pruned model
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[SARATHI] Saving pruned model to {out_dir}")
        model.save_pretrained(str(out_dir))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out_dir))

        if "adaptive_sizes" in stats:
            with open(out_dir / "adaptive_sizes.json", "w") as f:
                json.dump(stats["adaptive_sizes"], f)
            logging.info("[SARATHI] Saved adaptive_sizes.json for custom loader.")

        logging.info("[SARATHI] Save complete.")


if __name__ == "__main__":
    main()
