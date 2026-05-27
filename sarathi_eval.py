"""
sarathi_eval.py — Zero-Shot Evaluation for SARATHI Pruned LLMs
===============================================================
Wraps the EleutherAI lm-evaluation-harness to evaluate a SARATHI-pruned
model on zero-shot reasoning benchmarks.

Two task sets:
  5-task (default):
    PIQA, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande

  7-task (SoBP-compatible, use --sobp-tasks):
    PIQA, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, BoolQ, OpenBookQA
    — matches the exact evaluation protocol of SoBP (EMNLP 2024, Table 1)

Usage:

  # Standard 5-task eval:
  python sarathi_eval.py \\
      --model meta-llama/Meta-Llama-3-8B \\
      --model-path ./pruned/sarathi_llama3_8b_25 \\
      --batch-size 8

  # SoBP-compatible 7-task eval:
  python sarathi_eval.py \\
      --model meta-llama/Meta-Llama-3-8B \\
      --model-path ./pruned/sarathi_llama3_8b_25 \\
      --sobp-tasks --batch-size 4

  # Dense baseline:
  python sarathi_eval.py --model meta-llama/Meta-Llama-3-8B --sobp-tasks
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

import torch

# ── Task sets ─────────────────────────────────────────────────────────────────
DEFAULT_TASKS = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"]
SOBP_TASKS    = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq", "openbookqa"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description     = "Zero-shot evaluation for SARATHI pruned LLMs.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="HuggingFace model name or path to the base model.")
    parser.add_argument("--model-path", default=None, metavar="PATH",
                        help="Path to the pruned model directory (saved by sarathi_main.py).")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="lm-eval task names to evaluate. Overrides --sobp-tasks.")
    parser.add_argument("--sobp-tasks", action="store_true", default=False,
                        help="Use SoBP 7-task eval set (adds BoolQ, OpenBookQA). "
                             "Required for fair comparison against SoBP EMNLP 2024 Table 1.")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--output-file", default=None,
                        help="Explicit output path for JSON results. Overrides --output-dir/--output-name.")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use device_map='auto' for multi-GPU inference.")
    return parser


def load_pruned_model(model_path: str, device=None, device_map=None, **kwargs):
    """
    Load a SARATHI-pruned model.

    If adaptive_sizes.json exists, the model was pruned with Adaptive Structured
    Slicing (variable intermediate_size per layer). We load with the custom
    per-layer resizing logic. Otherwise, standard from_pretrained is used.
    """
    import json as _json
    from transformers import AutoModelForCausalLM, AutoConfig

    adaptive_json = os.path.join(model_path, "adaptive_sizes.json")

    if not os.path.exists(adaptive_json):
        # Uniform pruning — standard HF load
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, **kwargs
        )
        return model.to(device) if device else model

    # Adaptive pruning — manual per-layer resize before loading state dict
    logging.info(f"[SARATHI] Loading adaptive-pruned model from {model_path} ...")
    with open(adaptive_json) as f:
        adaptive_sizes = _json.load(f)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    _device_map = kwargs.pop("device_map", device_map)

    from sarathi.model_utils import get_mlp_modules

    if _device_map is not None:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        with init_empty_weights():
            dense_model = AutoModelForCausalLM.from_config(config, **kwargs)
    else:
        dense_model = AutoModelForCausalLM.from_config(config, **kwargs)

    # Find decoder layers
    if hasattr(dense_model, "model") and hasattr(dense_model.model, "layers"):
        layers = dense_model.model.layers
    elif hasattr(dense_model, "model") and hasattr(dense_model.model, "decoder"):
        layers = dense_model.model.decoder.layers
    else:
        raise ValueError("Cannot find decoder layers in model.")

    target_device = device or "cpu"

    def _new_linear(in_f, out_f, bias, dtype, dev):
        return torch.nn.Linear(in_f, out_f, bias=bias, dtype=dtype,
                               device=dev if _device_map is None else "meta")

    mn = config.model_type
    for layer, new_size in zip(layers, adaptive_sizes):
        mods = get_mlp_modules(mn, layer)
        if mods.get("gate") is not None:
            in_dim  = mods["gate"].in_features
            out_dim = mods["down"].out_features
            dtype   = mods["gate"].weight.dtype
            layer.mlp.gate_proj = _new_linear(in_dim, new_size, False, dtype, target_device)
            layer.mlp.up_proj   = _new_linear(in_dim, new_size, False, dtype, target_device)
            layer.mlp.down_proj = _new_linear(new_size, out_dim, False, dtype, target_device)
        else:
            in_dim  = mods["up"].in_features
            out_dim = mods["down"].out_features
            dtype   = mods["up"].weight.dtype
            has_b1  = mods["up"].bias is not None
            has_b2  = mods["down"].bias is not None
            if mn.startswith("phi"):
                layer.mlp.fc1 = _new_linear(in_dim, new_size, has_b1, dtype, target_device)
                layer.mlp.fc2 = _new_linear(new_size, out_dim, has_b2, dtype, target_device)
            elif mn == "opt":
                layer.fc1 = _new_linear(in_dim, new_size, has_b1, dtype, target_device)
                layer.fc2 = _new_linear(new_size, out_dim, has_b2, dtype, target_device)

    if _device_map is not None:
        from accelerate import load_checkpoint_and_dispatch
        dense_model.tie_weights()
        dense_model = load_checkpoint_and_dispatch(
            dense_model, checkpoint=model_path, device_map=_device_map,
            no_split_module_classes=dense_model._no_split_modules,
        )
        return dense_model

    # Load state dict
    import safetensors.torch, glob as _glob
    st_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(st_path):
        sd = safetensors.torch.load_file(st_path, device=target_device)
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location=target_device, weights_only=False)
    else:
        # Sharded safetensors
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(idx_path) as f:
            index = _json.load(f)
        weight_files = set(index["weight_map"].values())
        sd = {}
        for wf in weight_files:
            sd.update(safetensors.torch.load_file(os.path.join(model_path, wf), device=target_device))

    dense_model.load_state_dict(sd, strict=False)
    logging.info("[SARATHI] Adaptive model loaded successfully.")
    return dense_model


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= [logging.StreamHandler(sys.stdout)],
    )

    parser = build_arg_parser()
    args   = parser.parse_args()

    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve task list
    if args.tasks is not None:
        task_list = []
        for t in args.tasks:
            task_list.extend(t.split(","))
    elif args.sobp_tasks:
        task_list = SOBP_TASKS
        logging.info("[SARATHI] Using SoBP 7-task evaluation set.")
    else:
        task_list = DEFAULT_TASKS
    args.tasks = task_list

    load_path = str(Path(args.model_path if args.model_path else args.model).resolve())
    logging.info(f"[SARATHI] Loading model from: {load_path}")

    try:
        from lm_eval import evaluator, tasks as lm_tasks
        from lm_eval.models.huggingface import HFLM
        if hasattr(lm_tasks, "initialize_tasks"):
            lm_tasks.initialize_tasks()
    except ImportError:
        logging.error("lm-evaluation-harness not installed. Run: pip install lm-eval")
        sys.exit(1)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)

    device_arg = None if args.multi_gpu else str(device)
    device_map = "auto" if args.multi_gpu else None

    model = load_pruned_model(
        load_path,
        device     = device_arg,
        device_map = device_map,
        torch_dtype= torch.float16,
        trust_remote_code=True,
    ).eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    logging.info(f"[SARATHI] Evaluating on tasks: {args.tasks}")
    results = evaluator.simple_evaluate(
        model=lm, tasks=args.tasks, num_fewshot=args.num_fewshot, log_samples=False,
    )

    print("\n" + "=" * 60)
    print("SARATHI — Zero-Shot Results")
    print("=" * 60)
    task_accs = {}
    for task_name, task_result in results["results"].items():
        acc = task_result.get("acc_norm,none", task_result.get("acc,none", None))
        if acc is not None:
            task_accs[task_name] = round(acc * 100, 2)
            print(f"  {task_name:<22} {acc * 100:.2f}%")

    if task_accs:
        avg = sum(task_accs.values()) / len(task_accs)
        print(f"  {'Average':<22} {avg:.2f}%")
    print("=" * 60 + "\n")

    # Save results
    if args.output_file:
        out_file = Path(args.output_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir  = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem     = args.output_name or f"eval_{Path(load_path).name}"
        out_file = out_dir / f"{stem}.json"

    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "results": results["results"]}, f, indent=2)
    logging.info(f"[SARATHI] Results saved to {out_file}")


if __name__ == "__main__":
    main()
