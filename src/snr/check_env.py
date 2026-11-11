"""Print the environment, GPU state and GPU budget. Run before any GPU job.

  PYTHONPATH=src python -m snr.check_env
"""
from __future__ import annotations

import os
import subprocess

import torch
import transformers

from .common import ledger_total_hours


def main():
    print(f"torch {torch.__version__}  transformers {transformers.__version__}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU-hours spent so far (logs/gpu_ledger.csv): {ledger_total_hours():.2f}")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"HF token present: {bool(tok)} (needed for gated Llama and Gemma weights)")
    if not torch.cuda.is_available():
        print("No CUDA device visible. Laptop phase: data building, counts, tests and CPU smoke only.")
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    try:
        print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv"], capture_output=True, text=True, timeout=20).stdout)
    except Exception as e:  # pragma: no cover
        print(f"nvidia-smi unavailable: {e}")
    print("Processes you did not start belong to other people. Never kill, signal or renice them.")


if __name__ == "__main__":
    main()
