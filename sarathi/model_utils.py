"""
model_utils.py — Architecture-agnostic model helpers for SARATHI
================================================================
Provides model loading, layer access, and calibration data utilities.

Supported architectures: LLaMA-2, LLaMA-3, Mistral, OPT-2.7B/6.7B, LLaMA-2-13B
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── DynamicCache patch: prevent multi-GPU device mismatch during generate() ──
try:
    import transformers.cache_utils
    _orig_dynamic_cache_update = transformers.cache_utils.DynamicCache.update

    def _patched_dynamic_cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        dev = key_states.device
        if len(self.key_cache) > layer_idx:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(dev)
        if len(self.value_cache) > layer_idx:
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(dev)
        return _orig_dynamic_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)

    transformers.cache_utils.DynamicCache.update = _patched_dynamic_cache_update
except Exception as e:
    logging.warning(f"Failed to patch DynamicCache: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_all_layers(model_name: str, model: nn.Module) -> list:
    """Return the list of transformer decoder layers for a given model."""
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral")):
        return list(model.model.layers)
    elif "opt" in mn:
        return list(model.model.decoder.layers)
    elif "phi" in mn:
        return list(model.model.layers)
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")


def get_mlp_modules(model_name: str, layer: nn.Module) -> dict:
    """
    Return a dict mapping 'gate', 'up', 'down' to the corresponding Linear modules.
    'gate' is None for non-gated FFNs (OPT, Phi-2).
    """
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral")):
        return {
            "gate": layer.mlp.gate_proj,
            "up":   layer.mlp.up_proj,
            "down": layer.mlp.down_proj,
        }
    elif "opt" in mn:
        return {
            "gate": None,
            "up":   layer.fc1,
            "down": layer.fc2,
        }
    elif "phi" in mn:
        return {
            "gate": None,
            "up":   layer.mlp.fc1,
            "down": layer.mlp.fc2,
        }
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")


def get_intermediate_size(model_name: str, model: nn.Module) -> int:
    """Return the current FFN intermediate_size from model config."""
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral")):
        return model.config.intermediate_size
    elif "opt" in mn:
        return model.config.ffn_dim
    elif "phi" in mn:
        return model.config.intermediate_size
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")


def set_intermediate_size(model_name: str, model: nn.Module, new_size: int) -> None:
    """Update the FFN intermediate_size in model config after pruning."""
    mn = model_name.lower()
    if any(k in mn for k in ("llama", "mistral")):
        model.config.intermediate_size = new_size
    elif "opt" in mn:
        model.config.ffn_dim = new_size
    elif "phi" in mn:
        model.config.intermediate_size = new_size
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")


def slice_mlp_layer(model_name: str, layer: nn.Module, keep_indices: torch.Tensor) -> None:
    """
    Physically slice MLP weight matrices to retain only the neurons in keep_indices.
    This permanently reduces intermediate_size for this layer.

    Args:
        model_name   : Architecture identifier string.
        layer        : A single transformer decoder layer (nn.Module).
        keep_indices : 1-D LongTensor of neuron indices to KEEP (sorted).
    """
    mn = model_name.lower()
    idx = keep_indices.to("cpu")

    if any(k in mn for k in ("llama", "mistral")):
        mlp = layer.mlp
        # gate_proj, up_proj: rows = intermediate neurons  [N, d_model]
        mlp.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.data[idx])
        mlp.up_proj.weight   = nn.Parameter(mlp.up_proj.weight.data[idx])
        # down_proj: cols = intermediate neurons  [d_model, N]
        mlp.down_proj.weight = nn.Parameter(mlp.down_proj.weight.data[:, idx])
        # Update in-module size tracking
        mlp.gate_proj.out_features = len(idx)
        mlp.up_proj.out_features   = len(idx)
        mlp.down_proj.in_features  = len(idx)

    elif "opt" in mn:
        # fc1: rows = intermediate [N, d_model]
        layer.fc1.weight = nn.Parameter(layer.fc1.weight.data[idx])
        if layer.fc1.bias is not None:
            layer.fc1.bias = nn.Parameter(layer.fc1.bias.data[idx])
        # fc2: cols = intermediate [d_model, N]
        layer.fc2.weight = nn.Parameter(layer.fc2.weight.data[:, idx])
        layer.fc1.out_features = len(idx)
        layer.fc2.in_features  = len(idx)

    elif "phi" in mn:
        layer.mlp.fc1.weight = nn.Parameter(layer.mlp.fc1.weight.data[idx])
        if layer.mlp.fc1.bias is not None:
            layer.mlp.fc1.bias = nn.Parameter(layer.mlp.fc1.bias.data[idx])
        layer.mlp.fc2.weight = nn.Parameter(layer.mlp.fc2.weight.data[:, idx])
        layer.mlp.fc1.out_features = len(idx)
        layer.mlp.fc2.in_features  = len(idx)

    else:
        raise ValueError(f"Unsupported architecture: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
    device=None,
    torch_dtype: str = "auto",
    device_map=None,
):
    """
    Load a HuggingFace causal LM and tokenizer from a local path or HF hub.

    If device_map='auto', the model is distributed across all visible GPUs
    (for 13B+ models) and .to(device) is NOT called.

    Args:
        model_path  : HuggingFace model name or absolute local path.
        device      : torch.device (used only when device_map is None).
        torch_dtype : Dtype string passed to from_pretrained (default 'auto').
        device_map  : 'auto' for multi-GPU, None for single device.

    Returns:
        (model, tokenizer) tuple.
    """
    import glob

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve local HF cache snapshot if model_path is a hub ID
    cache_pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}/snapshots/*"
    )
    snapshots = glob.glob(cache_pattern)
    if snapshots:
        model_path = snapshots[0]

    logging.info(f"[SARATHI] Loading model from {model_path} (device_map={device_map}) ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        if "model.safetensors" in str(e) or "NameResolutionError" in str(e):
            load_kwargs["use_safetensors"] = False
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        else:
            raise

    if device_map is None:
        model.to(device)

    model.eval()
    if hasattr(model.config, "max_position_embeddings"):
        model.seqlen = model.config.max_position_embeddings
    elif hasattr(model.config, "max_sequence_length"):
        model.seqlen = model.config.max_sequence_length
    else:
        model.seqlen = 2048

    logging.info("[SARATHI] Model loaded successfully.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_wikitext_calib_tokens(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 512,
) -> torch.Tensor:
    """
    Load WikiText-2 calibration tokens from the local HuggingFace cache.

    Looks for the dataset in:
      1. Standard HF parquet snapshot  (~/.cache/huggingface/hub/...)
      2. Path set via SARATHI_WIKITEXT_PATH environment variable

    Returns:
        Tensor of shape [n_samples, seq_len] (Long).
    """
    from datasets import load_dataset, Dataset

    # 1. Standard HF hub parquet path
    import glob
    wiki_parquet = ""
    for pattern in [
        "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/train-00000-of-00001.parquet",
    ]:
        matches = glob.glob(os.path.expanduser(pattern))
        if matches:
            wiki_parquet = matches[0]
            break

    # 2. Override via environment variable (set this on your cluster)
    env_path = os.environ.get("SARATHI_WIKITEXT_PATH", "")

    if wiki_parquet and os.path.exists(wiki_parquet):
        logging.info(f"[SARATHI] Loading WikiText-2 from parquet: {wiki_parquet}")
        raw = load_dataset("parquet", data_files=wiki_parquet, split="train")
    elif env_path and os.path.exists(env_path):
        logging.info(f"[SARATHI] Loading WikiText-2 from SARATHI_WIKITEXT_PATH: {env_path}")
        if env_path.endswith(".arrow"):
            raw = Dataset.from_file(env_path)
        else:
            raw = load_dataset("parquet", data_files=env_path, split="train")
    else:
        raise FileNotFoundError(
            f"WikiText-2 not found.\n"
            "Either:\n"
            "  1. Run: huggingface-cli download wikitext --repo-type dataset\n"
            "  2. Set SARATHI_WIKITEXT_PATH=/path/to/wikitext-train.arrow (or .parquet)"
        )

    text   = "\n\n".join([t for t in raw["text"] if t.strip()])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunks.append(tokens[i : i + seq_len])
        if len(chunks) >= n_samples:
            break

    logging.info(f"[SARATHI] WikiText-2 calibration: {len(chunks)} chunks × {seq_len} tokens")
    return torch.stack(chunks)  # [n_samples, seq_len]


def load_alpaca_calib_tokens(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 512,
) -> torch.Tensor:
    """
    Load Alpaca calibration tokens.

    Reads the JSON file from SARATHI_ALPACA_PATH environment variable.
    Format: each record has 'instruction', 'input', 'output' fields.

    Returns:
        Tensor of shape [n_samples, seq_len] (Long).
    """
    import json

    alpaca_path = os.environ.get("SARATHI_ALPACA_PATH", "")
    if not alpaca_path or not os.path.exists(alpaca_path):
        raise FileNotFoundError(
            "Alpaca calibration data not found.\n"
            "Set SARATHI_ALPACA_PATH=/path/to/alpaca_data.json"
        )

    logging.info(f"[SARATHI] Loading Alpaca calibration data from {alpaca_path} ...")
    with open(alpaca_path) as f:
        data = json.load(f)

    texts = []
    for rec in data:
        instr = rec.get("instruction", "").strip()
        inp   = rec.get("input", "").strip()
        out   = rec.get("output", "").strip()
        if inp:
            text = f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        else:
            text = f"Instruction: {instr}\nOutput: {out}"
        texts.append(text)

    full_text = "\n\n".join(texts)
    tokens    = tokenizer(full_text, return_tensors="pt").input_ids[0]

    chunks = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunks.append(tokens[i : i + seq_len])
        if len(chunks) >= n_samples:
            break

    logging.info(f"[SARATHI] Alpaca calibration: {len(chunks)} chunks × {seq_len} tokens")
    return torch.stack(chunks)  # [n_samples, seq_len]


def load_c4_calib_tokens(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
) -> torch.Tensor:
    """
    Load C4 calibration tokens from the locally cached subset.
    Used for OPT models as per the SoBP (EMNLP 2024) protocol.

    Returns:
        Tensor of shape [n_samples, seq_len] (Long).
    """
    import json

    c4_path = os.environ.get("SARATHI_C4_PATH", "")
    if not c4_path or not os.path.exists(c4_path):
        raise FileNotFoundError(
            "C4 calibration data not found.\n"
            "Set SARATHI_C4_PATH=/path/to/c4_calibration_subset.json\n"
            "Generate with: python scripts/prepare_c4_cache.py"
        )

    logging.info("[SARATHI] Loading C4 calibration data (128 samples, seq_len=2048, SoBP protocol) ...")
    with open(c4_path, "r") as f:
        texts = json.load(f)

    full_text = "\n\n".join(texts)
    tokens    = tokenizer(full_text, return_tensors="pt").input_ids[0]

    chunks = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunks.append(tokens[i : i + seq_len])
        if len(chunks) == n_samples:
            break

    logging.info(f"[SARATHI] C4 calibration: {len(chunks)} chunks × {seq_len} tokens")
    return torch.stack(chunks)  # [n_samples, seq_len]


def load_calib_tokens(
    tokenizer,
    dataset: str = "wikitext",
    n_samples: int = 128,
    seq_len: int = 2048,
) -> torch.Tensor:
    """
    Unified calibration token loader.

    Args:
        dataset  : 'wikitext' | 'c4' | 'alpaca'
        n_samples: Number of calibration chunks.
        seq_len  : Token length per chunk.

    Returns:
        Tensor of shape [n_samples, seq_len] (Long).
    """
    if dataset == "wikitext":
        return load_wikitext_calib_tokens(tokenizer, n_samples=n_samples, seq_len=seq_len)
    elif dataset == "c4":
        return load_c4_calib_tokens(tokenizer, n_samples=n_samples, seq_len=seq_len)
    elif dataset == "alpaca":
        return load_alpaca_calib_tokens(tokenizer, n_samples=n_samples, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown calib dataset: '{dataset}'. Choose 'wikitext', 'c4', or 'alpaca'.")


__all__ = [
    "load_model_and_tokenizer",
    "get_all_layers",
    "get_mlp_modules",
    "get_intermediate_size",
    "set_intermediate_size",
    "slice_mlp_layer",
    "load_calib_tokens",
    "load_wikitext_calib_tokens",
    "load_alpaca_calib_tokens",
    "load_c4_calib_tokens",
]
