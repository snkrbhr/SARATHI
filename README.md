# SARATHI: Structured Pruning of LLMs via NMF Residual Subspace Probing

> **EMNLP 2026 Anonymous Submission**

---

## Method Overview

```
Dense LLM
    |
    |-- Phase 1: NMF RESIDUAL SUBSPACE PROBE  (data-free)
    |       S = |W| - V@H   (NMF residual per weight matrix)
    |       MAD threshold -> binary mask M (True = geometrically irreplaceable)
    |
    |-- Phase 2: MAGNITUDE-WEIGHTED SCORING  (SwiGLU-aware)
    |       eta_i = sum_j (|W[i,j]| * M[i,j])   for gate, up, down projections
    |       SwiGLU: eta = (eta_gate + eta_up + eta_down) / 3
    |
    |-- Phase 3: STRUCTURED SLICE
    |       Keep Top-K neurons per layer -> permanently reduced intermediate_size
    |
    +-- Phase 4: ADAPTIVE OBS RECONSTRUCTION  (optional, O(d^2) per layer)
            Cholesky least-squares solve per layer
            + Bias Shift Compensation (post-LayerNorm correction)
```

---

## Variants

| Variant | Name          |    Data-Free?    | Probe                       | Notes                           |
| :-----: | :------------ | :--------------: | :-------------------------- | :------------------------------ |
|  **E**  | SARATHI       |       Yes        | NMF Residual Subspace Probe | Recommended. Core contribution. |
|  **B**  | SARATHI-Wanda | No (128 samples) | Wanda activation-weighted   | Better on gated (LLaMA/Mistral) |
|  **A**  | Magnitude     |       Yes        | Magnitude at sigma          | Baseline                        |
|  **C**  | NMF-Scoring   | No (128 samples) | Wanda + NMF on masked W     | Ablation                        |

---

## Installation

**Requires Python >= 3.10**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers datasets scipy tqdm tabulate matplotlib lm-eval accelerate safetensors
```

### Dataset Paths (set once on your cluster)

SARATHI uses environment variables instead of hardcoded paths:

```bash
export SARATHI_WIKITEXT_PATH=/path/to/wikitext-train.arrow   # or .parquet
export SARATHI_ALPACA_PATH=/path/to/alpaca_data.json
export SARATHI_C4_PATH=/path/to/c4_calibration_subset.json
```

WikiText-2 is also auto-resolved from the standard HuggingFace cache
(`~/.cache/huggingface/hub/datasets--wikitext/...`) if the env var is not set.

---

## Usage

### Pruning

```bash
# SARATHI -- data-free, with OBS reconstruction (recommended)
python sarathi_main.py \
    --model meta-llama/Llama-3-8B \
    --variant E \
    --structured-ratio 0.25 \
    --obs-reconstruct \
    --save-dir ./pruned_models/sarathi_llama3_8b_25

# SARATHI-Wanda -- 128-sample calibration (better on gated architectures)
python sarathi_main.py \
    --model meta-llama/Llama-3-8B \
    --variant B \
    --structured-ratio 0.25 \
    --n-calib 128 \
    --calib-dataset wikitext \
    --obs-reconstruct \
    --save-dir ./pruned_models/sarathi_wanda_llama3_8b_25

# OPT-2.7B at 40% sparsity (SARATHI is the only method that does not OOM)
python sarathi_main.py \
    --model facebook/opt-2.7b \
    --variant E \
    --structured-ratio 0.40 \
    --obs-reconstruct \
    --calib-dataset c4 \
    --save-dir ./pruned_models/sarathi_opt2.7b_40

# LLaMA-2-13B with multi-GPU (2x A100)
CUDA_VISIBLE_DEVICES=0,1 python sarathi_main.py \
    --model meta-llama/Llama-2-13b-hf \
    --variant E \
    --structured-ratio 0.25 \
    --obs-reconstruct \
    --multi-gpu \
    --save-dir ./pruned_models/sarathi_llama2_13b_25
```

### Evaluation (WikiText-2 PPL)

```bash
python sarathi_eval.py \
    --model meta-llama/Llama-3-8B \
    --model-path ./pruned_models/sarathi_llama3_8b_25 \
    --batch-size 8
```

### Zero-Shot Accuracy (SoBP-compatible 7-task)

```bash
python sarathi_eval.py \
    --model meta-llama/Llama-3-8B \
    --model-path ./pruned_models/sarathi_llama3_8b_25 \
    --sobp-tasks \
    --batch-size 8
```

---

## Key Arguments

| Argument             | Default | Description                                                               |
| :------------------- | :-----: | :------------------------------------------------------------------------ |
| `--variant`          |   `E`   | SARATHI variant: E (data-free), B (Wanda), A (magnitude), C (NMF-scoring) |
| `--structured-ratio` | `0.25`  | Fraction of FFN neurons to permanently remove                             |
| `--probe-sigma`      | `0.10`  | Unstructured probe sparsity sigma (Phase 1)                               |
| `--nmf-rank`         |   `7`   | NMF factorization rank r                                                  |
| `--nmf-iters`        |  `100`  | NMF multiplicative update iterations                                      |
| `--n-calib`          |  `128`  | Calibration samples (Variants B, C; OBS)                                  |
| `--obs-reconstruct`  | `False` | Enable Adaptive OBS Weight Reconstruction                                 |
| `--obs-damping`      | `1e-6`  | Tikhonov regularisation for Cholesky solve                                |
| `--adaptive`         | `False` | Adaptive slicing (Global MAD threshold, variable K/layer)                 |
| `--multi-gpu`        | `False` | Multi-GPU loading for 13B+ models                                         |
| `--seed`             |  `42`   | Random seed                                                               |

---

## Supported Models

| Model       | Architecture |  Gated FFN?  | Notes                          |
| :---------- | :----------- | :----------: | :----------------------------- |
| LLaMA-3-8B  | LLaMA        | Yes (SwiGLU) | Main experiment                |
| Mistral-7B  | Mistral      | Yes (SwiGLU) | Main experiment                |
| OPT-2.7B    | OPT          |  No (ReLU)   | Bias Shift Compensation active |
| OPT-6.7B    | OPT          |  No (ReLU)   | Only SARATHI succeeds at 40%   |
| LLaMA-2-13B | LLaMA        | Yes (SwiGLU) | Multi-GPU experiment           |

---

## Codebase Structure

```
sarathi_submission/
|
|-- sarathi_main.py              # CLI entry point (pruning)
|-- sarathi_eval.py              # Evaluation entry point
|-- requirements.txt
|-- ENVIRONMENT_SETUP.md        # Detailed setup instructions
|
+-- sarathi/                     # Core SARATHI package
    |-- __init__.py
    |-- model_utils.py           # Model loading + architecture helpers
    |-- nmf.py                   # Pure PyTorch NMF (Multiplicative Update)
    |-- thresholding.py          # Global MAD binary search
    |-- probe.py                 # Phase 1: NMF residual, Wanda, magnitude probes
    |-- score.py                 # Phase 2: Neuron importance scoring
    |-- slice.py                 # Phase 3: Structured slicing (uniform + adaptive)
    |-- obs_reconstruction.py    # Phase 4: Adaptive OBS + Bias Shift Compensation
    +-- pipeline.py              # End-to-end orchestrator
```

---

## Main Results

### WikiText-2 Perplexity (lower is better)

| Method             | LLaMA-3-8B (25%) | Mistral-7B (25%) | OPT-2.7B (40%) | OPT-6.7B (40%) |
| :----------------- | :--------------: | :--------------: | :------------: | :------------: |
| Dense              |       6.14       |       5.25       |     12.47      |     10.86      |
| SliceGPT           |       8.92       |       7.41       |     28.73      |      OOM       |
| SoBP               |       8.11       |       6.89       |     22.34      |      OOM       |
| Dynamic Slicing    |       8.44       |       7.12       |     25.61      |      OOM       |
| **SARATHI (ours)** |     **7.53**     |     **6.31**     |   **18.98**    |   **16.42**    |

OPT-6.7B at 40% sparsity: SARATHI is the only method that completes without OOM.
All baselines crash due to coupled O(Ld^2) Hessian caching.

### Zero-Shot Accuracy (5-task average, higher is better)

| Method                   | LLaMA-3-8B (25%) | Mistral-7B (25%) |
| :----------------------- | :--------------: | :--------------: |
| Dense                    |       72.8       |       71.3       |
| SliceGPT                 |       62.1       |       60.4       |
| SoBP                     |       64.7       |       62.9       |
| **SARATHI (ours)**       |     **67.4**     |     **65.8**     |
| **SARATHI-Wanda (ours)** |     **68.9**     |     **67.1**     |

---

## Ablation: Bias Shift Compensation (OPT-2.7B, 40% sparsity)

| Configuration                            | WikiText-2 PPL |
| :--------------------------------------- | :------------: |
| SARATHI without Bias Shift Compensation  |    2,082.4     |
| **SARATHI with Bias Shift Compensation** |   **18.98**    |

---

## Reproducibility

All experiments use:
- `--seed 42`
- `--n-calib 128` (when calibration is used)
- `--calib-seq-len 2048`
- `--nmf-rank 7`, `--nmf-iters 100`
- `--probe-sigma 0.10`
