# SARATHI: Decoupled Data-Free Probing and Adaptive Reconstruction for Structured LLM Pruning

## Abstract
Structured pruning of LLMs enables hardware-agnostic speedup by shrinking FFN dimensions, yet existing methods couple neuron selection and weight reconstruction through shared calibration data, causing domain bias and out-of-memory (OOM) failures at scale. We introduce SARATHI: a data-free NMF Residual Subspace Probe identifies irreplaceable neurons via weight geometry grounded in Robust PCA, while Adaptive OBS reconstructs pruned weights using only $\mathcal{O}(d^2)$ memory per layer compared to $\mathcal{O}(Ld^2)$ in coupled baselines. In addition, a zero-overhead Bias Shift Compensation corrects post-LayerNorm activation shifts, reducing perplexity from 2082 to 18.98 on OPT-2.7B at 40% sparsity relative to the uncorrected pruned model. Across five models (LLaMA-3-8B, Mistral-7B, OPT-2.7B/6.7B, and LLaMA-2-13B) at 15-40% sparsity, SARATHI achieves state-of-the-art WikiText-2 perplexity, consistently outperforms SoBP, SliceGPT, and Dynamic Slicing on zero-shot accuracy, and is the only method capable of compressing OPT-6.7B at 40% sparsity where all baselines fail with OOM errors. A lightweight variant, SARATHI-Wanda, using only a 128-sample calibration batch, further improves zero-shot accuracy on gated architectures.

### Method Overview
```text
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

This repository implements the SARATHI pruning method for Large Language Models. (EMNLP 2026 Anonymous Submission)

---

## Setup and Installation

**Requires Python >= 3.10**

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch transformers datasets scipy tqdm tabulate matplotlib lm-eval accelerate safetensors
```

2. **Dataset Paths (set once on your cluster):**

SARATHI uses environment variables instead of hardcoded paths:
```bash
export SARATHI_WIKITEXT_PATH=/path/to/wikitext-train.arrow   # or .parquet
export SARATHI_ALPACA_PATH=/path/to/alpaca_data.json
export SARATHI_C4_PATH=/path/to/c4_calibration_subset.json
```
WikiText-2 is also auto-resolved from the standard HuggingFace cache (`~/.cache/huggingface/hub/datasets--wikitext/...`) if the env var is not set.

---

## Usage

### 1. Pruning
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

### 2. Evaluation (WikiText-2 PPL)
```bash
python sarathi_eval.py \
    --model meta-llama/Llama-3-8B \
    --model-path ./pruned_models/sarathi_llama3_8b_25 \
    --batch-size 8
```

### 3. Zero-Shot Accuracy (SoBP-compatible 7-task)
```bash
python sarathi_eval.py \
    --model meta-llama/Llama-3-8B \
    --model-path ./pruned_models/sarathi_llama3_8b_25 \
    --sobp-tasks \
    --batch-size 8
```

---

## Project Structure

```text
sarathi_submission/
|
|-- sarathi_main.py              # CLI entry point (pruning)
|-- sarathi_eval.py              # Evaluation entry point
|-- requirements.txt
|-- ENVIRONMENT_SETUP.md         # Detailed setup instructions
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

## Configuration & Details

### Variants
| Variant | Name | Data-Free? | Probe | Notes |
|:---:|:---|:---:|:---|:---|
| **E** | SARATHI | Yes | NMF Residual Subspace Probe | Recommended. Core contribution. |
| **B** | SARATHI-Wanda | No (128 samples) | Wanda activation-weighted | Better on gated (LLaMA/Mistral) |
| **A** | Magnitude | Yes | Magnitude at sigma | Baseline |
| **C** | NMF-Scoring | No (128 samples) | Wanda + NMF on masked W | Ablation |

### Key Arguments
| Argument | Default | Description |
|:---|:---:|:---|
| `--variant` | `E` | SARATHI variant: E (data-free), B (Wanda), A (magnitude), C (NMF-scoring) |
| `--structured-ratio` | `0.25` | Fraction of FFN neurons to permanently remove |
| `--probe-sigma` | `0.10` | Unstructured probe sparsity sigma (Phase 1) |
| `--nmf-rank` | `7` | NMF factorization rank r |
| `--nmf-iters` | `100` | NMF multiplicative update iterations |
| `--n-calib` | `128` | Calibration samples (Variants B, C; OBS) |
| `--obs-reconstruct` | `False` | Enable Adaptive OBS Weight Reconstruction |
| `--obs-damping` | `1e-6` | Tikhonov regularisation for Cholesky solve |
| `--adaptive` | `False` | Adaptive slicing (Global MAD threshold, variable K/layer) |
| `--multi-gpu` | `False` | Multi-GPU loading for 13B+ models |
| `--seed` | `42` | Random seed |

### Supported Models
| Model | Architecture | Gated FFN? | Notes |
|:---|:---|:---:|:---|
| LLaMA-3-8B | LLaMA | Yes (SwiGLU) | Main experiment |
| Mistral-7B | Mistral | Yes (SwiGLU) | Main experiment |
| OPT-2.7B | OPT | No (ReLU) | Bias Shift Compensation active |
| OPT-6.7B | OPT | No (ReLU) | Only SARATHI succeeds at 40% |
| LLaMA-2-13B | LLaMA | Yes (SwiGLU) | Multi-GPU experiment |

---

## Main Results

### WikiText-2 Perplexity (lower is better)
| Method | LLaMA-3-8B (25%) | Mistral-7B (25%) | OPT-2.7B (40%) | OPT-6.7B (40%) |
|:---|:---:|:---:|:---:|:---:|
| Dense | 6.14 | 5.25 | 12.47 | 10.86 |
| SliceGPT | 8.92 | 7.41 | 28.73 | OOM |
| SoBP | 8.11 | 6.89 | 22.34 | OOM |
| Dynamic Slicing | 8.44 | 7.12 | 25.61 | OOM |
| **SARATHI (ours)** | **7.53** | **6.31** | **18.98** | **16.42** |

*OPT-6.7B at 40% sparsity: SARATHI is the only method that completes without OOM. All baselines crash due to coupled O(Ld^2) Hessian caching.*

### Zero-Shot Accuracy (5-task average, higher is better)
| Method | LLaMA-3-8B (25%) | Mistral-7B (25%) |
|:---|:---:|:---:|
| Dense | 72.8 | 71.3 |
| SliceGPT | 62.1 | 60.4 |
| SoBP | 64.7 | 62.9 |
| **SARATHI (ours)** | **67.4** | **65.8** |
| **SARATHI-Wanda (ours)** | **68.9** | **67.1** |

### Ablation: Bias Shift Compensation (OPT-2.7B, 40% sparsity)
| Configuration | WikiText-2 PPL |
|:---|:---:|
| SARATHI without Bias Shift Compensation | 2,082.4 |
| **SARATHI with Bias Shift Compensation** | **18.98** |

### Reproducibility
All experiments use:
- `--seed 42`
- `--n-calib 128` (when calibration is used)
- `--calib-seq-len 2048`
- `--nmf-rank 7`, `--nmf-iters 100`
- `--probe-sigma 0.10`

---

## Citation

If you find this code useful for your research, please consider citing our paper. This citation will be updated following the double-blind review process.

```bibtex
@inproceedings{anonymous2026sarathi,
  title={SARATHI: Structured Pruning of LLMs via NMF Residual Subspace Probing},
  author={Anonymous Authors},
  booktitle={Under Review (EMNLP 2026)},
  year={2026}
}
```
