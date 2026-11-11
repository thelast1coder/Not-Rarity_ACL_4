"""Build the G0 entity set (laptop phase, no GPU).

  PYTHONPATH=src python -m snr.build_entities --config configs/g0.yaml --out data/g0
  PYTHONPATH=src python -m snr.build_entities --config configs/g0.yaml --out data/g0 --raw data/raw_rows.jsonl

Steps
1. Fetch raw rows from Wikidata (or read a saved --raw file), save them to data/raw_rows.jsonl.
2. Aggregate per person; keep Latin-script names with a given name and a surname.
3. Name-origin class, leave-one-out: the modal citizenship among OTHER people with the same
   surname (fallback: same given name). It is a descriptor of the name, computed without
   the person's own citizenship, and it is never written into a prompt.
4. Candidate sets: top-K values per slot. Truth = person's values within the candidate set.
5. Cells = (origin, profession, birth decade). W(a | cell) = world distribution (Wikidata)
   of the slot among people in the cell; the analysis uses it leave-one-out.
6. Leakage audit: drop people whose name contains a truth value (e.g. a surname that is a
   country name).
7. Select cells (spread across origins) and sample items across the popularity range.
"""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_yaml, read_jsonl, write_json, write_jsonl

NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'’\-\. ]+$")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def parse_name(name: str):
    if not name or "(" in name or not NAME_RE.match(name):
        return None
    toks = [t for t in name.replace(",", " ").split() if t.lower().strip(".") not in SUFFIXES]
    if len(toks) < 2:
        return None
    given, surname = toks[0], toks[-1]
    if len(given.strip(".")) <= 1 or len(surname.strip(".")) <= 1:
        return None  # initials only
    return given, surname


def aggregate(rows: list[dict]) -> dict[str, dict]:
    people: dict[str, dict] = {}
    for r in rows:
        nm = parse_name(r["name"])
        if nm is None:
            continue
        p = people.setdefault(r["qid"], {
            "qid": r["qid"], "name": " ".join(r["name"].split()), "given": nm[0], "surname": nm[1],
            "birth_year": r["birth_year"], "groups": set(), "citizenship": set(),
            "values": defaultdict(set), "sitelinks": 0})
        p["birth_year"] = min(p["birth_year"], r["birth_year"])
        p["groups"].add(r["group"])
        p["citizenship"].add(r["citizenship"])
        p["values"][r["slot"]].add(r["value"])
        p["sitelinks"] = max(p["sitelinks"], r["sitelinks"])
    return people


def _weighted_counts(people, key):
    counts: dict[str, Counter] = defaultdict(Counter)
    n: Counter = Counter()
    for p in people.values():
        k = p[key]
        n[k] += 1
        for c in p["citizenship"]:
            counts[k][c] += 1.0 / len(p["citizenship"])
    return counts, n


def origin_classes(people: dict, min_support: int, min_purity: float) -> dict[str, str | None]:
    """Leave-one-out modal citizenship among others sharing the surname (fallback given name)."""
    out = {}
    tables = [_weighted_counts(people, "surname"), _weighted_counts(people, "given")]
    for q, p in people.items():
        origin = None
        for (counts, n), key in zip(tables, ("surname", "given")):
            k = p[key]
            others = n[k] - 1
            if others < min_support:
                continue
            c = counts[k].copy()
            for cit in p["citizenship"]:
                c[cit] -= 1.0 / len(p["citizenship"])
            total = sum(v for v in c.values() if v > 1e-9)
            if total <= 0:
                continue
            top, w = max(c.items(), key=lambda kv: (kv[1], kv[0]))
            if w / total >= min_purity:
                origin = top
                break
        out[q] = origin
    return out


def leaky(name: str, values: set[str], surfaces: dict[str, str]) -> bool:
    toks = {t.lower().strip(".'’") for t in name.split()}
    for v in values:
        for s in {v, surfaces.get(v, v)}:
            if s.lower() in toks or s.lower() in name.lower().split(" "):
                return True
    return False


def build(rows: list[dict], cfg: dict, seed: int = 0) -> dict:
    rng = random.Random(seed)
    people = aggregate(rows)
    origins = origin_classes(people, cfg["origin_min_support"], cfg["origin_min_purity"])
    aliases = cfg.get("value_surface", {})
    prof_of_group = cfg["profession_of_group"]
    slot_groups = cfg["slot_groups"]  # slot -> groups that carry it

    # candidate sets per slot (top-K values among people in the slot's groups)
    candidates = {}
    for slot, groups in slot_groups.items():
        cnt = Counter()
        for p in people.values():
            if not (p["groups"] & set(groups)):
                continue
            vals = p["citizenship"] if slot == "citizenship" else p["values"].get(slot, set())
            for v in vals:
                cnt[v] += 1
        top = [v for v, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg["top_k_values"][slot]]]
        candidates[slot] = [{"label": v, "surface": aliases.get(v, v)} for v in top]

    # eligible items per slot
    items_by_slot: dict[str, list[dict]] = {}
    report = {"people": len(people), "with_origin": sum(o is not None for o in origins.values())}
    for slot, groups in slot_groups.items():
        labels = {c["label"] for c in candidates[slot]}
        surf = {c["label"]: c["surface"] for c in candidates[slot]}
        elig = []
        n_leaky = 0
        for p in people.values():
            if origins[p["qid"]] is None:
                continue
            for g in sorted(p["groups"] & set(groups)):
                vals = p["citizenship"] if slot == "citizenship" else p["values"].get(slot, set())
                truth = sorted(set(vals) & labels)
                if not truth:
                    continue
                if leaky(p["name"], set(truth), surf):
                    n_leaky += 1
                    continue
                decade = p["birth_year"] // 10 * 10
                prof = prof_of_group[g]
                elig.append({
                    "slot": slot, "qid": p["qid"], "name": p["name"], "given": p["given"],
                    "surname": p["surname"], "origin": origins[p["qid"]], "profession": prof,
                    "decade": decade, "birth_year": p["birth_year"], "sitelinks": p["sitelinks"],
                    "cell": f"{origins[p['qid']]}|{prof}|{decade}", "truth": truth,
                    "truth_surface": [surf[t] for t in truth],
                })
                break  # one profession per person per slot
        items_by_slot[slot] = elig
        report[f"{slot}_eligible"] = len(elig)
        report[f"{slot}_leaky_dropped"] = n_leaky

    # world distribution W per (slot, cell) over ALL eligible people in the cell
    cells: dict[str, dict] = {}
    for slot, elig in items_by_slot.items():
        by_cell = defaultdict(list)
        for it in elig:
            by_cell[it["cell"]].append(it)
        cells[slot] = {}
        for cell, its in by_cell.items():
            w = Counter()
            for it in its:
                for t in it["truth"]:
                    w[t] += 1.0 / len(it["truth"])
            o, prof, dec = cell.split("|")
            cells[slot][cell] = {"origin": o, "profession": prof, "decade": int(dec),
                                 "pool_size": len(its), "w_counts": dict(w)}

    # select cells and sample items
    sel = cfg["selection"]
    chosen_items = []
    for slot, elig in items_by_slot.items():
        pool = {c: v for c, v in cells[slot].items() if v["pool_size"] >= sel["min_cell_pool"]}
        ranked = sorted(pool, key=lambda c: (-pool[c]["pool_size"], c))
        per_origin = Counter()
        picked = []
        for c in ranked:
            o = pool[c]["origin"]
            if per_origin[o] >= sel["max_cells_per_origin"]:
                continue
            picked.append(c)
            per_origin[o] += 1
            if len(picked) >= sel["max_cells_per_slot"]:
                break
        cap = math.ceil(sel["items_per_slot"] / max(1, len(picked)))
        for c in picked:
            its = [it for it in elig if it["cell"] == c]
            rng.shuffle(its)
            its.sort(key=lambda it: it["sitelinks"])
            if len(its) > cap:  # evenly spaced over popularity
                step = len(its) / cap
                its = [its[int(i * step)] for i in range(cap)]
            chosen_items.extend(its)
        report[f"{slot}_cells_selected"] = len(picked)
        report[f"{slot}_origins_selected"] = len(per_origin)
        cells[slot] = {c: cells[slot][c] for c in picked}
    for i, it in enumerate(chosen_items):
        it["item_id"] = f"{it['slot']}:{it['qid']}"
    report["items"] = len(chosen_items)

    # name pools per origin for ghosts: people whose origin class matches one of their citizenships
    pools = defaultdict(lambda: {"given": Counter(), "surname": Counter()})
    for q, p in people.items():
        o = origins[q]
        if o is not None and o in p["citizenship"]:
            pools[o]["given"][p["given"]] += 1
            pools[o]["surname"][p["surname"]] += 1
    used_origins = {v["origin"] for s in cells.values() for v in s.values()}
    name_pools = {o: {"given": [g for g, _ in pools[o]["given"].most_common()],
                      "surname": [s for s, _ in pools[o]["surname"].most_common()]}
                  for o in sorted(used_origins)}
    real_names = sorted({p["name"] for p in people.values()})
    return {"items": chosen_items, "cells": cells, "candidates": candidates, "name_pools": name_pools,
            "real_names": real_names, "report": report}


def write_outputs(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "items.jsonl", res["items"])
    write_json(out / "cells.json", res["cells"])
    write_json(out / "candidates.json", res["candidates"])
    write_json(out / "name_pools.json", res["name_pools"])
    (out / "real_names.txt").write_text("\n".join(res["real_names"]), encoding="utf-8")
    write_json(out / "build_report.json", res["report"])
    lines = ["# G0 data sanity report", ""] + [f"- {k}: {v}" for k, v in res["report"].items()]
    lines += ["", "Check: every slot has >= 10 selected cells, >= 5 origins and close to items_per_slot items.",
              "If not, fix QIDs or thresholds in configs/g0.yaml and log it in notes/decisions.md."]
    (out / "build_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--out", default="data/g0")
    ap.add_argument("--raw", default=None, help="saved raw rows (jsonl); skips the network")
    ap.add_argument("--endpoint", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)["data"]
    if args.raw:
        rows = read_jsonl(args.raw)
    else:
        from .wikidata import fetch_group

        rows = []
        for g in cfg["groups"]:
            got = fetch_group(g, cfg, args.endpoint or cfg["endpoint"])
            print(f"[wikidata] {g}: {len(got)} rows")
            rows.extend(got)
        write_jsonl(Path(args.out).parent / "raw_rows.jsonl", rows)
    res = build(rows, cfg, seed=cfg.get("seed", 0))
    write_outputs(res, Path(args.out))


if __name__ == "__main__":
    main()
