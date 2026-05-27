# SARATHI — Environment Setup Guide
# EMNLP 2026 Anonymous Submission

This document covers everything needed to reproduce SARATHI experiments from scratch —
from environment creation to running pruning and evaluation.

---

## ⚠️ Python Version Requirement

> **Python 3.10 or higher is required.**

The `sarathi/` codebase uses Python 3.10+ syntax:
- PEP 585 built-in generic types: `list[MaskResult]`, `dict[int, torch.Tensor]`
- PEP 604 union type hints: `dict | None`

These will raise a `TypeError` on Python 3.9 or earlier. Do **not** use the system Python on older clusters.

---

## Step 1 — Create a Conda Environment

```bash
# Create a fresh environment with Python 3.10
conda create -n sarathi python=3.10 -y
conda activate sarathi
```

Verify:
```bash
python --version   # must print Python 3.10.x or higher
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Install PyTorch (GPU)

If `pip install torch` installs a CPU-only build, install the CUDA version explicitly:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU access:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Step 3 — Download Models (offline / cluster)

SARATHI sets `HF_HUB_OFFLINE=1` at runtime. Download models **before** running:

```bash
# On a login node (internet access):
huggingface-cli download meta-llama/Meta-Llama-3-8B
huggingface-cli download mistralai/Mistral-7B-v0.1
huggingface-cli download facebook/opt-2.7b
huggingface-cli download facebook/opt-6.7b
huggingface-cli download meta-llama/Llama-2-13b-hf
```

Models are cached to `~/.cache/huggingface/hub/`. SARATHI auto-resolves snapshots from there.

---

## Step 4 — Dataset Paths

Set these environment variables **once** (add to `~/.bashrc` or your job script):

```bash
# WikiText-2 (auto-resolved from HF cache if not set)
export SARATHI_WIKITEXT_PATH=/path/to/wikitext-train.arrow    # or .parquet

# Alpaca (only needed for SARATHI-Wanda with --calib-dataset alpaca)
export SARATHI_ALPACA_PATH=/path/to/alpaca_data.json

# C4 subset (for OPT models — SoBP protocol)
export SARATHI_C4_PATH=/path/to/c4_calibration_subset.json
```

### Downloading WikiText-2

```bash
huggingface-cli download wikitext --repo-type dataset
```

This populates `~/.cache/huggingface/hub/datasets--wikitext/`. SARATHI finds it automatically.

### Generating a C4 Calibration Subset (for OPT)

```python
# Run once on a machine with internet access:
from datasets import load_dataset
import json

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
texts = [sample["text"] for _, sample in zip(range(500), ds)]
with open("c4_calibration_subset.json", "w") as f:
    json.dump(texts, f)
```

---

## Step 5 — Quick Smoke Test

Run a tiny sanity check to confirm the environment is working:

```bash
python - <<'EOF'
import torch
from sarathi.model_utils import get_all_layers, get_mlp_modules
from sarathi.nmf import pytorch_nmf
from sarathi.thresholding import find_mad_threshold
import numpy as np

# NMF smoke test
V = torch.rand(64, 128).abs()
W, H = pytorch_nmf(V, r=4, n_iter=10)
assert W.shape == (64, 4) and H.shape == (4, 128), "NMF shape mismatch"

# MAD threshold smoke test
scores = np.random.rand(10000)
t = find_mad_threshold(scores, 0.25)
assert isinstance(t, float), "Threshold type error"

print("✅ Environment OK — SARATHI sarathi/ package imports correctly.")
EOF
```

Expected output:
```
✅ Environment OK — SARATHI sarathi/ package imports correctly.
```

---

## Step 6 — Run SARATHI

### Single GPU (LLaMA-3-8B, 25% sparsity)

```bash
python sarathi_main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --variant E \
    --structured-ratio 0.25 \
    --obs-reconstruct \
    --calib-dataset wikitext \
    --n-calib 128 \
    --save-dir ./pruned/sarathi_llama3_8b_25
```

### Multi-GPU (LLaMA-2-13B)

```bash
CUDA_VISIBLE_DEVICES=0,1 python sarathi_main.py \
    --model meta-llama/Llama-2-13b-hf \
    --variant E \
    --structured-ratio 0.25 \
    --obs-reconstruct \
    --multi-gpu \
    --save-dir ./pruned/sarathi_llama2_13b_25
```

### OPT-2.7B at 40% (Bias Shift Compensation active)

```bash
python sarathi_main.py \
    --model facebook/opt-2.7b \
    --variant E \
    --structured-ratio 0.40 \
    --obs-reconstruct \
    --calib-dataset c4 \
    --save-dir ./pruned/sarathi_opt2.7b_40
```

### SARATHI-Wanda (gated architectures)

```bash
python sarathi_main.py \
    --model mistralai/Mistral-7B-v0.1 \
    --variant B \
    --structured-ratio 0.25 \
    --n-calib 128 \
    --calib-dataset wikitext \
    --obs-reconstruct \
    --save-dir ./pruned/sarathi_wanda_mistral_25
```

---

## Step 7 — Evaluate

### WikiText-2 Perplexity

```bash
python run_eval.py \
    --model meta-llama/Meta-Llama-3-8B \
    --model-path ./pruned/sarathi_llama3_8b_25 \
    --batch-size 8
```

### Zero-Shot Accuracy (5 tasks)

```bash
lm_eval --model hf \
    --model_args pretrained=./pruned/sarathi_llama3_8b_25 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --batch_size 8
```

---

## Common Issues

| Error                                           | Cause                          | Fix                                                     |
| :---------------------------------------------- | :----------------------------- | :------------------------------------------------------ |
| `TypeError: 'type' object is not subscriptable` | Python < 3.10                  | `conda activate sarathi` (Python 3.10)                  |
| `FileNotFoundError: WikiText-2 not found`       | Dataset not cached             | `huggingface-cli download wikitext --repo-type dataset` |
| `FileNotFoundError: C4 cache not found`         | `SARATHI_C4_PATH` not set      | Set `export SARATHI_C4_PATH=...`                        |
| `CUDA out of memory`                            | Model too large for single GPU | Add `--multi-gpu` flag                                  |
| `HF hub connection error`                       | Offline mode enabled           | Pre-download models on login node                       |
| `transformers.modeling_utils: ... safetensors`  | Missing safetensors            | `pip install safetensors`                               |

---

## Reproducibility Checklist

- [ ] Python 3.10+
- [ ] `pip install -r requirements.txt`
- [ ] Models downloaded to HF cache
- [ ] `SARATHI_WIKITEXT_PATH` or HF cache populated
- [ ] `SARATHI_C4_PATH` set (for OPT models)
- [ ] `--seed 42` (default)
- [ ] `--nmf-rank 7 --nmf-iters 100 --probe-sigma 0.10` (defaults match paper)
- [ ] `--n-calib 128 --calib-seq-len 2048` (SoBP protocol)
