import os
import subprocess
import time
from queue import Queue
from threading import Thread

# ==============================================================================
# OPT Sweep Script (Distributed across GPUs)
# ==============================================================================
# This script orchestrates 8 tasks (Wanda/NMF x 20/40% x c4/wikitext)
# across available GPUs. Ensure you run this on a compute node, not the login node.

# Define GPUs to use on THIS node (Change this if running on node 2)
# E.g., if you have 4 GPUs on this node:
GPUS = [0, 1, 2, 3]

# The base command components
PYTHON_EXEC = "python"  # Ensure this points to the correct virtual environment
MAIN_SCRIPT = "sarathi_main.py"
EVAL_SCRIPT = "sarathi_eval.py"

MODEL = "facebook/opt-6.7b"
SPARSITIES = [0.20, 0.40]
VARIANTS = {"Wanda": "B", "NMF": "E"}
DATASETS = ["c4", "wikitext"]

# Create permutations
tasks = []
for sparsity in SPARSITIES:
    for v_name, v_code in VARIANTS.items():
        for dataset in DATASETS:
            tasks.append({
                "sparsity": sparsity,
                "variant_name": v_name,
                "variant_code": v_code,
                "dataset": dataset
            })

def run_task(gpu_id, task):
    alias = "opt_6.7b"
    label = f"v{task['variant_code']}_{task['dataset']}"
    model_path = f"pruned_models/{alias}/{label}_sp{task['sparsity']}"
    prune_log = f"prune_{label}_sp{task['sparsity']}.log"
    eval_log = f"eval_{label}_sp{task['sparsity']}.log"

    os.makedirs(model_path, exist_ok=True)
    
    # 1. PRUNING COMMAND
    prune_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} {PYTHON_EXEC} {MAIN_SCRIPT} "
        f"--model {MODEL} "
        f"--variant {task['variant_code']} "
        f"--structured-ratio {task['sparsity']} "
        f"--adaptive --min-keep 0.5 "
        f"--obs-reconstruct "
        f"--n-calib 128 --calib-seq-len 2048 "
        f"--calib-dataset {task['dataset']} "
        f"--save-dir {model_path} > {prune_log} 2>&1"
    )
    
    # Add YOPO rank if variant E
    if task['variant_code'] == "E":
        prune_cmd = prune_cmd.replace("--save-dir", "--yopo-rank 16 --yopo-iters 100 --save-dir")

    print(f"[GPU {gpu_id}] Starting Pruning: {task['variant_name']} {task['sparsity']*100}% on {task['dataset']}")
    subprocess.run(prune_cmd, shell=True)
    
    # 2. EVALUATION COMMAND
    eval_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} {PYTHON_EXEC} {EVAL_SCRIPT} "
        f"--model {model_path} "
        f"--batch-size 16 > {eval_log} 2>&1"
    )
    
    print(f"[GPU {gpu_id}] Starting Evaluation: {task['variant_name']} {task['sparsity']*100}% on {task['dataset']}")
    subprocess.run(eval_cmd, shell=True)
    print(f"[GPU {gpu_id}] Completed: {task['variant_name']} {task['sparsity']*100}% on {task['dataset']}")

def worker(gpu_id, queue):
    while not queue.empty():
        task = queue.get()
        try:
            run_task(gpu_id, task)
        finally:
            queue.task_done()

if __name__ == "__main__":
    print(f"Total tasks: {len(tasks)}")
    print(f"GPUs allocated: {GPUS}")
    
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)
        
    threads = []
    for gpu_id in GPUS:
        t = Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("All tasks completed successfully!")
