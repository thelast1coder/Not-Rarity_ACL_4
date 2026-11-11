"""Shared helpers: I/O, seeding, GPU etiquette on a shared card, model loading, prompts.

This file is identical across the lab's ACL 2027 repositories (SNR, ABS, VOICE).
If you change it in one repo, copy the change to the others.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import random
import re
import socket
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- I/O
def write_jsonl(path, rows, gz: bool | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gz = path.suffix == ".gz" if gz is None else gz
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- GPU etiquette
def gpu_free_gib() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def require_gpu_memory(need_gib: float, headroom_gib: float = 8.0):
    """The GPU may be shared with other people's experiments. Never kill their
    processes; refuse to start if there is not enough memory plus headroom."""
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. Check CUDA_VISIBLE_DEVICES or use --device cpu for smoke tests.")
    free, total = gpu_free_gib()
    if free < need_gib + headroom_gib:
        raise SystemExit(
            f"Only {free:.1f} GiB free of {total:.1f} GiB; this job needs about {need_gib:.1f} GiB "
            f"plus {headroom_gib:.0f} GiB headroom. Other jobs may be using this GPU. Do not kill them. "
            "Wait, lower --batch-size, or use a smaller model."
        )
    print(f"[gpu] {free:.1f} GiB free of {total:.1f} GiB; this job needs about {need_gib:.1f} GiB")


def estimate_need_gib(hf_id: str, batch_size: int) -> float:
    """Rough: bf16 weights + activations/logits buffer + slack."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB]\b", hf_id.replace("-", " ").replace("_", " "))
    params_b = float(m.group(1)) if m else 8.0
    return params_b * 2.1 + 0.08 * batch_size + 3.0


def append_gpu_ledger(script: str, model: str, seconds: float, note: str = ""):
    """Every GPU job appends one row. The ledger is how we keep the budget honest."""
    path = ROOT / "logs" / "gpu_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc_time", "host", "device", "script", "model", "gpu_hours", "note"])
        hours = seconds / 3600 if name != "cpu" else 0.0
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(),
                    name, script, model, f"{hours:.4f}", note])


def ledger_total_hours() -> float:
    path = ROOT / "logs" / "gpu_ledger.csv"
    if not path.exists():
        return 0.0
    with open(path, encoding="utf-8") as f:
        return sum(float(r["gpu_hours"]) for r in csv.DictReader(f))


# ---------------------------------------------------------------- models
def load_model(hf_id: str, device: str = "cuda", revision: str | None = None, dtype: str = "auto"):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    major = int(transformers.__version__.split(".")[0])
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    kw = {"revision": revision}
    kw["dtype" if major >= 5 else "torch_dtype"] = torch_dtype
    if "gemma-2" in hf_id.lower():
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kw).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def build_prompt(tok, text: str, chat: bool, system: str | None = None) -> str:
    """Base models get the raw text. Chat models get their own chat template with a
    generation prompt, so the assistant turn starts right after the returned string."""
    if not chat:
        return text
    if not getattr(tok, "chat_template", None):
        raise ValueError("chat=True but the tokenizer has no chat_template")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_meta(model, hf_id: str, extra: dict | None = None) -> dict:
    import transformers

    meta = {
        "model": hf_id,
        "commit": getattr(model.config, "_commit_hash", None),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        meta.update(extra)
    return meta


def model_tag(hf_id: str) -> str:
    return hf_id.rstrip("/").split("/")[-1]
