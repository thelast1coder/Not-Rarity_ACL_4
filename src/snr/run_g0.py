"""G0 model pass (GPU phase): score every candidate value for real entities, ghosts and the
generic baseline, plus a free-generation agreement check.

  PYTHONPATH=src python -m snr.run_g0 --model allenai/OLMo-2-1124-7B-Instruct --chat --data data/g0 --out results/G0
  PYTHONPATH=src python -m snr.run_g0 --model allenai/OLMo-2-1124-7B --no-chat --data data/g0 --out results/G0

Outputs in results/G0/<model_tag>/: scores.jsonl.gz, freegen.jsonl, meta.json.
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from pathlib import Path

from .common import (append_gpu_ledger, build_prompt, estimate_need_gib, load_model, model_tag, read_json,
                     read_jsonl, require_gpu_memory, run_meta, seed_all, write_json, write_jsonl)
from .prompts import candidate_continuation, slot_text
from .scoring import score_pairs


def build_jobs(items, cells, ghosts, max_ghosts: int | None = None):
    jobs = []
    for it in items:
        jobs.append({"kind": "real", "slot": it["slot"], "cell": it["cell"], "ref": it["item_id"],
                     "name": it["name"], "profession": it["profession"], "decade": it["decade"]})
    for slot, cs in cells.items():
        for cell, info in cs.items():
            gl = ghosts.get(info["origin"], [])
            if max_ghosts is not None:
                gl = gl[:max_ghosts]
            for gi, g in enumerate(gl):
                jobs.append({"kind": "ghost", "slot": slot, "cell": cell, "ref": gi, "name": g["name"],
                             "profession": info["profession"], "decade": info["decade"]})
            jobs.append({"kind": "generic", "slot": slot, "cell": cell, "ref": "generic", "name": None,
                         "profession": info["profession"], "decade": info["decade"]})
    return jobs


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def map_answer(text: str, cands: list[dict]) -> int | None:
    """Map a free-text answer to a candidate index (longest surface/label that matches)."""
    t = " " + norm(text) + " "
    best, best_len = None, 0
    for k, c in enumerate(cands):
        for form in {c["surface"], c["label"]}:
            f = norm(form)
            if f and (" " + f + " ") in t and len(f) > best_len:
                best, best_len = k, len(f)
    return best


def run(model, tok, hf_id: str, chat: bool, data_dir: Path, out_dir: Path, batch_size: int = 32,
        limit_items: int | None = None, max_ghosts: int | None = None, freegen: bool = True,
        progress: bool = True) -> Path:
    items = read_jsonl(data_dir / "items.jsonl")
    if limit_items:
        items = items[:limit_items]
    cells = read_json(data_dir / "cells.json")
    used = {(it["slot"], it["cell"]) for it in items}
    cells = {s: {c: v for c, v in cs.items() if (s, c) in used} for s, cs in cells.items()}
    ghosts = read_json(data_dir / "ghosts.json")
    cands = read_json(data_dir / "candidates.json")
    jobs = build_jobs(items, cells, ghosts, max_ghosts)
    prompts = [build_prompt(tok, slot_text(j["slot"], j["name"], j["profession"], j["decade"], chat), chat)
               for j in jobs]
    cand_strs = [[candidate_continuation(c["surface"], chat) for c in cands[j["slot"]]] for j in jobs]
    logp, ntok, bok = score_pairs(model, tok, prompts, cand_strs, add_special_tokens=not chat,
                                  batch_size=batch_size, progress=progress)
    tag = model_tag(hf_id)
    od = out_dir / tag
    rows = []
    for j, lp, nt, ok in zip(jobs, logp, ntok, bok):
        rows.append({**j, "logp": [round(float(x), 5) for x in lp], "ntok": [int(x) for x in nt],
                     "boundary_ok": [bool(x) for x in ok]})
    write_jsonl(od / "scores.jsonl.gz", rows)
    if freegen:
        from .generation import generate_hf

        real = [(j, p) for j, p in zip(jobs, prompts) if j["kind"] == "real"]
        outs = generate_hf(model, tok, [p for _, p in real], max_new_tokens=10, do_sample=False,
                           batch_size=batch_size, stops=["\n"], add_special_tokens=not chat, progress=progress)
        fg = []
        for (j, _), o in zip(real, outs):
            k = map_answer(o[0], cands[j["slot"]])
            fg.append({"ref": j["ref"], "slot": j["slot"], "text": o[0], "mapped": k})
        write_jsonl(od / "freegen.jsonl", fg)
    write_json(od / "meta.json", run_meta(model, hf_id, {"chat": chat, "n_jobs": len(jobs),
                                                          "n_items": len(items)}))
    return od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chat", dest="chat", action="store_true")
    ap.add_argument("--no-chat", dest="chat", action="store_false")
    ap.set_defaults(chat=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--out", default="results/G0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--max-ghosts", type=int, default=None)
    ap.add_argument("--no-freegen", action="store_true")
    args = ap.parse_args()
    if args.chat is None:
        raise SystemExit("Say --chat or --no-chat explicitly (base models must not get a chat template).")
    seed_all(0)
    if args.device == "cuda":
        require_gpu_memory(estimate_need_gib(args.model, args.batch_size))
    t0 = time.time()
    model, tok = load_model(args.model, device=args.device, revision=args.revision)
    od = run(model, tok, args.model, args.chat, Path(args.data), Path(args.out), args.batch_size,
             args.limit_items, args.max_ghosts, not args.no_freegen)
    append_gpu_ledger("snr.run_g0", args.model, time.time() - t0, note=str(od))
    print(f"wrote {od}")


if __name__ == "__main__":
    main()
