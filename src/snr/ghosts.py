"""Ghost probes: fictitious names matched to each name-origin class.

  PYTHONPATH=src python -m snr.ghosts --data data/g0 --config configs/g0.yaml

A ghost recombines a real given name and a real surname from the same origin pool, so
the name carries the same origin cue as real names, but the person does not exist.
Validity requirement (pre-registered): the full ghost name has ZERO exact-match
occurrences in the OLMo-2 pretraining corpus (infini-gram). Without network access the
ghosts are marked verified=False and G0 reports them as UNVERIFIED.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .common import load_yaml, read_json, write_json


def make_ghosts(name_pools: dict, real_names: set[str], n_per_origin: int, seed: int = 0,
                counter=None, pool_top: int = 300, max_tries: int = 5000) -> dict:
    """counter: callable name -> int count, or None (unverified)."""
    rng = random.Random(seed)
    out = {}
    for origin, pools in sorted(name_pools.items()):
        given = pools["given"][:pool_top]
        sur = pools["surname"][:pool_top]
        if len(given) < 3 or len(sur) < 3:
            out[origin] = []
            continue
        ghosts, seen, tries = [], set(), 0
        while len(ghosts) < n_per_origin and tries < max_tries:
            tries += 1
            g, s = rng.choice(given), rng.choice(sur)
            name = f"{g} {s}"
            if g == s or name in real_names or name in seen:
                continue
            seen.add(name)
            if counter is not None:
                c = counter(name)
                if c is None or c > 0:
                    continue
                ghosts.append({"name": name, "count": c, "verified": True})
            else:
                ghosts.append({"name": name, "count": None, "verified": False})
        out[origin] = ghosts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/g0")
    ap.add_argument("--config", default="configs/g0.yaml")
    ap.add_argument("--offline", action="store_true", help="skip the zero-count check (UNVERIFIED ghosts)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    data = Path(args.data)
    pools = read_json(data / "name_pools.json")
    real = set((data / "real_names.txt").read_text(encoding="utf-8").splitlines())
    counter = None
    if not args.offline:
        from .infinigram import InfiniGram

        ig = InfiniGram(index=cfg["counts"]["index"], cache_path=data / "infinigram_cache.jsonl")
        ig.ping()

        def counter(name):
            try:
                return ig.count(name)["count"]
            except Exception as e:  # network trouble: treat as unknown, do not use
                print(f"[ghosts] count failed for {name!r}: {e}")
                return None

    gh = make_ghosts(pools, real, cfg["ghosts"]["n_per_origin"], seed=cfg["ghosts"].get("seed", 0),
                     counter=counter)
    write_json(data / "ghosts.json", gh)
    for o, g in gh.items():
        print(f"{o:30s} {len(g):3d} ghosts  verified={sum(x['verified'] for x in g)}")


if __name__ == "__main__":
    main()
