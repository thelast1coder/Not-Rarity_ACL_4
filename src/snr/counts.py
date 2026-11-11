"""Exposure counts per item from the OLMo-2 pretraining corpus (laptop phase).

  PYTHONPATH=src python -m snr.counts --data data/g0 --config configs/g0.yaml [--method name_side|cnf]

n  = exact count of the full name (infini-gram `find`, never approximate).
c  = number of name occurrences with a surface form of a true value within the window.

Two methods:
- name_side (default, decided 2026-11-11): enumerate the name's occurrences from the rare side.
  If n <= K, fetch every occurrence's window: c is exact (for this window and these forms).
  If n > K, fetch a seeded uniform sample of K occurrences: c = n * hits / K, an unbiased estimate
  (c_approx = True). Window: max_disp_len = 2 * (window_tokens + L) tokens centred on the match
  (L = name length in tokens), so about window_tokens + L tokens either side, clipped at document edges.
  Value forms are matched as whole words, case-sensitive.
- cnf: infini-gram CNF count `name AND (forms)`, max_diff_tokens apart. Not usable for common values:
  a clause above max_clause_freq is subsampled, which drives c to 0 for rare names (see notes/decisions.md).
Counts are lower bounds on true exposure (paraphrases and pronouns are missed).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

from .common import load_yaml, read_json, read_jsonl, write_jsonl
from .infinigram import InfiniGram, InfiniGramError, cnf


def value_forms(label: str, surface: str, slot: str, demonyms: dict) -> list[str]:
    forms = [surface, label]
    if slot == "citizenship":
        forms += demonyms.get(label, [])
    else:
        forms += [surface.lower(), surface[:1].upper() + surface[1:]]
    return list(dict.fromkeys(f for f in forms if f))


def item_forms(it: dict, surf: dict, demonyms: dict) -> list[str]:
    forms = []
    for t in it["truth"]:
        forms += value_forms(t, surf[it["slot"]][t], it["slot"], demonyms)
    return list(dict.fromkeys(forms))


def build_queries(items: list[dict], candidates: dict, demonyms: dict) -> tuple[list[str], list[str]]:
    surf = {s: {c["label"]: c["surface"] for c in cs} for s, cs in candidates.items()}
    qn, qc = [], []
    for it in items:
        qn.append(it["name"])
        qc.append(cnf([it["name"]], item_forms(it, surf, demonyms)))
    return qn, qc


def forms_regex(forms: list[str]) -> re.Pattern:
    alts = sorted({f for f in forms if f}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(f) for f in alts) + r")(?!\w)")


def sample_occurrences(segments: list[list[int]], n: int, k: int, seed_key: str) -> list[tuple[int, int]]:
    """All (shard, rank) pairs if n <= k, else a uniform sample of k without replacement (seeded by item)."""
    total = sum(b - a for _, a, b in segments)
    if total != n:
        raise InfiniGramError(f"segments cover {total} occurrences, cnt says {n}")
    if n <= k:
        picks = range(n)
    else:
        seed = int(hashlib.sha1(seed_key.encode()).hexdigest()[:12], 16)
        picks = sorted(random.Random(seed).sample(range(n), k))
    out, it, base = [], iter(picks), 0
    nxt = next(it, None)
    for s, a, b in segments:
        while nxt is not None and nxt < base + (b - a):
            out.append((s, a + nxt - base))
            nxt = next(it, None)
        base += b - a
    return out


def name_side_count(ig: InfiniGram, name: str, forms: list[str], k: int, window: int, seed_key: str) -> dict:
    f = ig.find(name)
    n = f["cnt"]
    if n == 0:
        return {"n": 0, "n_approx": False, "c": 0, "c_approx": False, "c_hits": 0, "c_sampled": 0}
    occ = sample_occurrences(f["segments"], n, k, seed_key)
    disp = 2 * (window + len(f["token_ids"]))
    rx = forms_regex(forms)
    hits = sum(bool(rx.search(ig.doc_window(s, r, disp, f["token_ids"])["context"])) for s, r in occ)
    exact = n <= k
    c = hits if exact else n * hits / len(occ)
    return {"n": n, "n_approx": False, "c": c, "c_approx": not exact, "c_hits": hits, "c_sampled": len(occ)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--method", default=None, choices=["name_side", "cnf"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=1.0, help="seconds between API calls (shared rate limit)")
    ap.add_argument("--retries", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ccfg = cfg["counts"]
    method = args.method or ccfg.get("method", "name_side")
    data = Path(args.data)
    items = read_jsonl(data / "items.jsonl")
    cands = read_json(data / "candidates.json")
    demonyms = ccfg.get("demonyms", {})
    ig = InfiniGram(index=ccfg["index"], cache_path=data / "infinigram_cache.jsonl",
                    max_diff_tokens=ccfg["max_diff_tokens"], min_interval=args.min_interval,
                    retries=args.retries)
    ig.ping()
    rows = []
    if method == "cnf":
        qn, qc = build_queries(items, cands, demonyms)
        rn = ig.count_many(qn, workers=args.workers)
        rc = ig.count_many(qc, workers=args.workers)
        for it, a, b, q1, q2 in zip(items, rn, rc, qn, qc):
            rows.append({"item_id": it["item_id"], "n": None if a is None else a["count"],
                         "n_approx": None if a is None else a["approx"],
                         "c": None if b is None else b["count"], "c_approx": None if b is None else b["approx"],
                         "method": "cnf", "query_n": q1, "query_c": q2})
    else:
        from concurrent.futures import ThreadPoolExecutor

        surf = {s: {c["label"]: c["surface"] for c in cs} for s, cs in cands.items()}
        k, window = int(ccfg.get("k_occurrences", 50)), int(ccfg["max_diff_tokens"])
        seed = str(ccfg.get("seed", 0))

        def job(it):
            forms = item_forms(it, surf, demonyms)
            base = {"item_id": it["item_id"], "method": "name_side", "k": k, "window": window, "forms": forms}
            try:
                return {**base, **name_side_count(ig, it["name"], forms, k, window, f"{seed}:{it['item_id']}")}
            except InfiniGramError as e:
                print(f"[counts] failed {it['item_id']}: {e}", flush=True)
                return {**base, "n": None, "n_approx": None, "c": None, "c_approx": None}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(job, items)):
                rows.append(r)
                if (i + 1) % 25 == 0:
                    print(f"[counts] {i + 1}/{len(items)}  refused so far: {ig.n_refused}", flush=True)
    write_jsonl(data / "counts.jsonl", rows)
    ok = sum(r["c"] is not None for r in rows)
    print(f"counts ({method}): {ok}/{len(rows)} items have c; approx c: {sum(bool(r['c_approx']) for r in rows)}; "
          f"c > 0: {sum(bool(r['c']) for r in rows if r['c'] is not None)}")


if __name__ == "__main__":
    main()
