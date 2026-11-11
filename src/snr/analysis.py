"""G0 analysis and gate decision (CPU).

  PYTHONPATH=src python -m snr.analysis --data data/g0 --results results/G0 --config configs/g0.yaml

Definitions (docs/02_theory_and_definitions.md):
  p(a|x)   softmax over candidate log-probabilities for prompt x
  F(a|d)   type prior of cell d = mean of p(a|ghost) over the cell's ghosts
  G(a|d)   generic-subject baseline ('This person is a ...'), WikiMem-style
  s_m      model surprise   = -log2 sum_{a in truth} F(a|d)
  s_w      world surprise   = -log2 sum_{a in truth} W_loo(a|d)   (Wikidata, leave-one-out)
  mode     argmax_a F(a|d)
Gate G0 (pre-registered in configs/g0.yaml):
  GO     mode-match among wrong answers >= 0.50, chance < 0.10, and adding s_m to
         [log c, log n] raises held-out AUROC by >= 0.03
  KILL   mode-match < 0.30 (headline dead)
  PIVOT  delta-AUROC < 0.01 (no surprise effect: measurement paper only)
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import load_yaml, read_json, read_jsonl, write_json, write_jsonl
from .scoring import softmax_rows


def _probs(row):
    return softmax_rows(np.asarray(row["logp"], dtype=np.float64)[None, :])[0]


def assemble(items, cells, cands, rows, counts, alpha: float = 0.5):
    labels = {s: [c["label"] for c in cs] for s, cs in cands.items()}
    real, ghosts, generic = {}, defaultdict(list), {}
    for r in rows:
        if r["kind"] == "real":
            real[r["ref"]] = _probs(r)
        elif r["kind"] == "ghost":
            ghosts[(r["slot"], r["cell"])].append((r["ref"], _probs(r)))
        else:
            generic[(r["slot"], r["cell"])] = _probs(r)
    F, Fa, Fb = {}, {}, {}
    for key, lst in ghosts.items():
        lst.sort(key=lambda t: t[0])
        arr = np.stack([p for _, p in lst])
        F[key] = arr.mean(0)
        Fa[key] = arr[0::2].mean(0)
        Fb[key] = arr[1::2].mean(0) if len(arr) > 1 else arr.mean(0)
    cmap = {c["item_id"]: c for c in counts} if counts else {}
    # global value frequency per slot (for the frequency-matched chance baseline)
    gfreq = {}
    for s in labels:
        tot = defaultdict(float)
        for cell in cells.get(s, {}).values():
            for v, w in cell["w_counts"].items():
                tot[v] += w
        gfreq[s] = np.array([tot.get(a, 0.0) + 1e-9 for a in labels[s]])
    table = []
    for it in items:
        key = (it["slot"], it["cell"])
        if it["item_id"] not in real or key not in F:
            continue
        lab = labels[it["slot"]]
        K = len(lab)
        tidx = [lab.index(t) for t in it["truth"]]
        p = real[it["item_id"]]
        pred = int(np.argmax(p))
        mode = int(np.argmax(F[key]))
        mode_g = int(np.argmax(generic[key])) if key in generic else None
        cell = cells[it["slot"]][it["cell"]]
        w = cell["w_counts"]
        n_others = cell["pool_size"] - 1
        w_true = sum(max(w.get(t, 0.0) - 1.0 / len(it["truth"]), 0.0) + alpha for t in it["truth"])
        W_true = w_true / (n_others + alpha * K)
        F_true = float(sum(F[key][i] for i in tidx))
        c = cmap.get(it["item_id"], {})
        table.append({
            "item_id": it["item_id"], "slot": it["slot"], "cell": it["cell"], "K": K, "n_truth": len(tidx),
            "pred": lab[pred], "correct": pred in tidx, "mode": lab[mode], "mode_in_truth": mode in tidx,
            "mode_generic": None if mode_g is None else lab[mode_g],
            "p_true": float(sum(p[i] for i in tidx)), "F_true": F_true,
            "s_m": -math.log2(max(F_true, 1e-12)), "s_w": -math.log2(max(min(W_true, 1.0), 1e-12)),
            "c": c.get("c"), "n": c.get("n"), "sitelinks": it.get("sitelinks"),
            "freq_mode_share": float(gfreq[it["slot"]][mode] / gfreq[it["slot"]][[i for i in range(K) if i not in tidx]].sum())
            if mode not in tidx else 0.0,
        })
    rel = None
    if Fa:
        keys = sorted(Fa)
        x = np.concatenate([Fa[k] for k in keys])
        y = np.concatenate([Fb[k] for k in keys])
        if x.std() > 0 and y.std() > 0:
            rel = float(np.corrcoef(x, y)[0, 1])
    return table, rel


def mode_match_stats(table, n_boot: int = 1000, seed: int = 0):
    wrong = [r for r in table if not r["correct"]]
    if not wrong:
        return {"n_wrong": 0, "n_items": len(table),
                "accuracy": float(np.mean([r["correct"] for r in table])) if table else None}
    mm = np.array([r["pred"] == r["mode"] for r in wrong], dtype=float)
    cond = [r for r in wrong if not r["mode_in_truth"]]
    chance_u = np.array([0.0 if r["mode_in_truth"] else 1.0 / (r["K"] - r["n_truth"]) for r in wrong])
    chance_f = np.array([r["freq_mode_share"] for r in wrong])
    gen = [r for r in wrong if r["mode_generic"] is not None]
    rng = np.random.default_rng(seed)
    by_cell = defaultdict(list)
    for i, r in enumerate(wrong):
        by_cell[r["cell"]].append(i)
    cells = list(by_cell)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(cells), len(cells), replace=True)
        idx = [i for k in pick for i in by_cell[cells[k]]]
        boots.append(mm[idx].mean())
    return {
        "n_wrong": len(wrong), "n_items": len(table), "accuracy": float(np.mean([r["correct"] for r in table])),
        "mode_match": float(mm.mean()), "mode_match_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "mode_match_given_mode_wrong": float(np.mean([r["pred"] == r["mode"] for r in cond])) if cond else None,
        "chance_uniform": float(chance_u.mean()), "chance_frequency": float(chance_f.mean()),
        "generic_mode_match": float(np.mean([r["pred"] == r["mode_generic"] for r in gen])) if gen else None,
    }


def _design(table, extra: str | None, use_counts: bool):
    slots = sorted({r["slot"] for r in table})
    X = []
    for r in table:
        row = [1.0 if r["slot"] == s else 0.0 for s in slots[1:]]
        if use_counts:
            row += [math.log1p(r["c"]), math.log1p(r["n"])]
        if extra:
            row.append(r[extra])
        X.append(row)
    return np.asarray(X, dtype=float)


def auroc_gain(table, extra: str, use_counts: bool, n_splits: int = 5, n_boot: int = 1000, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([r["correct"] for r in table], dtype=int)
    groups = np.array([r["cell"] for r in table])
    if y.min() == y.max() or len(set(groups)) < n_splits:
        return {"note": "not enough variation for cross-validated AUROC"}
    oof = {}
    for name, ex in (("base", None), ("full", extra)):
        X = _design(table, ex, use_counts)
        pred = np.zeros(len(y))
        if X.shape[1] == 0:
            pred[:] = 0.5
        else:
            for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
                if y[tr].min() == y[tr].max():
                    pred[te] = y[tr].mean()
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                clf.fit(X[tr], y[tr])
                pred[te] = clf.predict_proba(X[te])[:, 1]
        oof[name] = pred
    base, full = roc_auc_score(y, oof["base"]), roc_auc_score(y, oof["full"])
    rng = np.random.default_rng(seed)
    by_cell = defaultdict(list)
    for i, g in enumerate(groups):
        by_cell[g].append(i)
    cl = list(by_cell)
    diffs = []
    for _ in range(n_boot):
        idx = np.array([i for k in rng.choice(len(cl), len(cl), replace=True) for i in by_cell[cl[k]]])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], oof["full"][idx]) - roc_auc_score(y[idx], oof["base"][idx]))
    return {"auroc_base": float(base), "auroc_full": float(full), "delta": float(full - base),
            "delta_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None}


def decide(mm: dict, gain_m: dict | None, gain_w: dict | None, counts_ok: bool, rel, ghosts_verified: float,
           fg_agree, gate: dict) -> dict:
    v = {"checks": {}, "notes": []}
    v["checks"]["V1_split_half_reliability>=%.2f" % gate["min_reliability"]] = rel is not None and rel >= gate["min_reliability"]
    v["checks"]["V2_ghosts_zero_count_verified"] = ghosts_verified >= 0.999
    v["checks"]["V3_freegen_agreement>=%.2f" % gate["min_freegen_agreement"]] = fg_agree is not None and fg_agree >= gate["min_freegen_agreement"]
    v["checks"]["V4_wrong_answers>=%d" % gate["min_wrong"]] = mm.get("n_wrong", 0) >= gate["min_wrong"]
    v["checks"]["V5_counts_available"] = counts_ok
    if mm.get("n_wrong", 0) == 0:
        v["verdict"] = "BLOCKED (no wrong answers)"
        return v
    m = mm["mode_match"]
    if m < gate["kill_mode_match"]:
        v["verdict"] = "KILL (errors are not prior modes)"
        return v
    if not counts_ok or not gain_m or "delta" not in gain_m:
        v["verdict"] = "BLOCKED (surprise criterion needs exact counts c and n)"
        return v
    d = gain_m["delta"]
    if d < gate["pivot_delta_auroc"]:
        v["verdict"] = "PIVOT (errors are defaults but surprise adds nothing: measurement paper only)"
    elif m >= gate["go_mode_match"] and mm["chance_uniform"] < gate["max_chance"] and d >= gate["go_delta_auroc"]:
        v["verdict"] = "GO"
    else:
        v["verdict"] = "REVIEW (between thresholds: human decision, log it in notes/decisions.md)"
    if gain_w and "delta" in gain_w and gain_w["delta"] < gate["pivot_delta_auroc"]:
        v["notes"].append("circularity risk: world surprise s_w adds < 0.01 AUROC while model surprise does")
    if not all(v["checks"].values()):
        v["notes"].append("one or more validity checks failed: the verdict is provisional until they pass")
    return v


def analyse_model(data: Path, mdir: Path, cfg: dict) -> dict:
    items = read_jsonl(data / "items.jsonl")
    cells = read_json(data / "cells.json")
    cands = read_json(data / "candidates.json")
    rows = read_jsonl(mdir / "scores.jsonl.gz")
    counts = read_jsonl(data / "counts.jsonl") if (data / "counts.jsonl").exists() else []
    table, rel = assemble(items, cells, cands, rows, counts, alpha=cfg["analysis"]["w_alpha"])
    write_jsonl(mdir / "item_table.jsonl", table)
    mm = mode_match_stats(table)
    counts_ok = len(table) > 0 and np.mean([r["c"] is not None and r["n"] is not None for r in table]) >= 0.95
    tab_c = [r for r in table if r["c"] is not None and r["n"] is not None] if counts_ok else table
    gain_m = auroc_gain(tab_c, "s_m", use_counts=counts_ok) if table else None
    gain_w = auroc_gain(tab_c, "s_w", use_counts=counts_ok) if table else None
    ghosts = read_json(data / "ghosts.json")
    gv = [g["verified"] for gl in ghosts.values() for g in gl]
    fg_agree = None
    if (mdir / "freegen.jsonl").exists():
        fg = read_jsonl(mdir / "freegen.jsonl")
        pred = {r["item_id"]: r["pred"] for r in table}
        labels = {s: [c["label"] for c in cs] for s, cs in cands.items()}
        mapped = [r for r in fg if r["mapped"] is not None and r["ref"] in pred]
        if mapped:
            fg_agree = float(np.mean([labels[r["slot"]][r["mapped"]] == pred[r["ref"]] for r in mapped]))
    bok = float(np.mean([all(r["boundary_ok"]) for r in rows])) if rows else None
    res = {"model": mdir.name, "mode_match": mm, "auroc_gain_s_m": gain_m, "auroc_gain_s_w": gain_w,
           "split_half_reliability": rel, "ghosts_verified_fraction": float(np.mean(gv)) if gv else 0.0,
           "freegen_agreement": fg_agree, "boundary_ok_rate": bok, "counts_ok": bool(counts_ok)}
    res["decision"] = decide(mm, gain_m, gain_w, bool(counts_ok), rel, res["ghosts_verified_fraction"],
                             fg_agree, cfg["gate"])
    return res


def render(results: list[dict], primary: str | None) -> str:
    L = ["# Gate G0: are hallucinations prior modes, and does surprise matter?", ""]
    for r in results:
        mm, d = r["mode_match"], r["decision"]
        L += [f"## {r['model']}{'  (primary)' if r['model'] == primary else ''}", "",
              f"**Verdict: {d['verdict']}**", ""]
        if not mm.get("n_items"):
            L.append("- no scored items with a ghost prior (check ghosts.json and cells.json)")
        if mm.get("n_wrong"):
            L += [f"- accuracy {mm['accuracy']:.3f} on {mm['n_items']} items; wrong answers {mm['n_wrong']}",
                  f"- mode-match among wrong answers: **{mm['mode_match']:.3f}** "
                  f"(95% cluster CI {mm['mode_match_ci'][0]:.3f}-{mm['mode_match_ci'][1]:.3f})",
                  f"- chance: uniform {mm['chance_uniform']:.3f}, frequency-matched {mm['chance_frequency']:.3f}; "
                  f"generic-subject mode-match {mm['generic_mode_match']}"]
        for k in ("auroc_gain_s_m", "auroc_gain_s_w"):
            g = r[k] or {}
            if "delta" in g:
                L.append(f"- {k}: base {g['auroc_base']:.3f} -> full {g['auroc_full']:.3f}, "
                         f"delta **{g['delta']:+.3f}** (CI {g['delta_ci']})")
            else:
                L.append(f"- {k}: {g.get('note', 'n/a')}")
        L += [f"- split-half reliability of F: {r['split_half_reliability']}",
              f"- ghosts verified zero-count: {r['ghosts_verified_fraction']:.2f}",
              f"- free-generation agreement: {r['freegen_agreement']}; boundary_ok rate {r['boundary_ok_rate']}",
              "", "Validity checks:"] + [f"- [{'x' if ok else ' '}] {k}" for k, ok in d["checks"].items()]
        L += [f"- note: {n}" for n in d["notes"]] + [""]
    L += ["Decision rule and thresholds: configs/g0.yaml (pre-registered). A human signs below.", "",
          "Signed: ____________________  date: __________"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--results", default="results/G0")
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--gate-out", default="results/gates")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = []
    for mdir in sorted(Path(args.results).iterdir()):
        if (mdir / "scores.jsonl.gz").exists():
            res.append(analyse_model(Path(args.data), mdir, cfg))
    if not res:
        raise SystemExit(f"no model results under {args.results}")
    primary = cfg["gate"].get("primary_model_tag")
    out = Path(args.gate_out)
    write_json(out / "G0.json", res)
    (out / "G0.md").write_text(render(res, primary), encoding="utf-8")
    print(render(res, primary))


if __name__ == "__main__":
    main()
