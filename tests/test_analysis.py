"""The analysis must recover known ground truth from synthetic scores."""
import math

import numpy as np

from snr.analysis import analyse_model, auroc_gain, mode_match_stats
from snr.common import write_json, write_jsonl

CFG = {"analysis": {"w_alpha": 0.5},
       "gate": {"primary_model_tag": "m", "go_mode_match": 0.5, "max_chance": 0.1, "go_delta_auroc": 0.03,
                "kill_mode_match": 0.3, "pivot_delta_auroc": 0.01, "min_reliability": 0.9,
                "min_freegen_agreement": 0.85, "min_wrong": 10}}


def make_world(tmp_path, n_cells=10, per_cell=30, seed=0, errors_are_modes=True):
    """Each cell has a prior mode. 'Known' items answer the truth; unknown items answer
    the mode if errors_are_modes, otherwise a random wrong value. Correctness is driven by
    surprise (truth == mode) not by counts, so adding s_m must raise AUROC."""
    rng = np.random.default_rng(seed)
    K = 16
    labels = [f"v{k}" for k in range(K)]
    cands = {"citizenship": [{"label": a, "surface": a} for a in labels]}
    items, rows, counts, cells = [], [], [], {"citizenship": {}}
    for ci in range(n_cells):
        cell = f"O{ci}|scientist|1960"
        mode = ci % K
        prior = np.full(K, 0.02)
        prior[mode] = 1 - 0.02 * (K - 1)
        w = {labels[mode]: per_cell * 0.7}
        for g in range(20):  # ghosts: noisy around the prior
            p = np.clip(prior + rng.normal(0, 0.005, K), 1e-4, None)
            rows.append({"kind": "ghost", "slot": "citizenship", "cell": cell, "ref": g,
                         "logp": list(np.log(p / p.sum())), "ntok": [1] * K, "boundary_ok": [True] * K})
        rows.append({"kind": "generic", "slot": "citizenship", "cell": cell, "ref": "generic",
                     "logp": list(np.log(np.full(K, 1 / K))), "ntok": [1] * K, "boundary_ok": [True] * K})
        for j in range(per_cell):
            typical = j < int(per_cell * 0.6)
            truth = mode if typical else int(rng.choice([k for k in range(K) if k != mode]))
            w[labels[truth]] = w.get(labels[truth], 0) + (0 if typical else 1)
            known = rng.random() < 0.3
            if known or typical:
                ans = truth
            else:
                ans = mode if errors_are_modes else int(rng.choice([k for k in range(K) if k not in (truth,)]))
            p = np.full(K, 0.01)
            p[ans] = 1.0
            iid = f"citizenship:Q{ci}_{j}"
            items.append({"item_id": iid, "slot": "citizenship", "cell": cell, "truth": [labels[truth]],
                          "name": f"N{ci}_{j}", "sitelinks": 1})
            rows.append({"kind": "real", "slot": "citizenship", "cell": cell, "ref": iid,
                         "logp": list(np.log(p / p.sum())), "ntok": [1] * K, "boundary_ok": [True] * K})
            counts.append({"item_id": iid, "c": int(rng.integers(0, 50)), "n": int(rng.integers(1, 500))})
        cells["citizenship"][cell] = {"origin": f"O{ci}", "profession": "scientist", "decade": 1960,
                                      "pool_size": per_cell, "w_counts": w}
    data = tmp_path / "data"
    mdir = tmp_path / "results" / "m"
    write_jsonl(data / "items.jsonl", items)
    write_json(data / "cells.json", cells)
    write_json(data / "candidates.json", cands)
    write_jsonl(data / "counts.jsonl", counts)
    write_json(data / "ghosts.json", {f"O{ci}": [{"name": "x", "count": 0, "verified": True}] * 20
                                      for ci in range(n_cells)})
    write_jsonl(mdir / "scores.jsonl.gz", rows)
    return data, mdir


def test_recovers_defaults_and_surprise(tmp_path):
    data, mdir = make_world(tmp_path)
    res = analyse_model(data, mdir, CFG)
    mm = res["mode_match"]
    assert mm["mode_match"] > 0.99
    assert mm["chance_uniform"] < 0.2
    assert res["split_half_reliability"] > 0.95
    assert res["auroc_gain_s_m"]["delta"] > 0.03
    assert res["decision"]["verdict"].startswith("GO")


def test_random_errors_are_killed(tmp_path):
    data, mdir = make_world(tmp_path, errors_are_modes=False)
    res = analyse_model(data, mdir, CFG)
    assert res["mode_match"]["mode_match"] < 0.3
    assert res["decision"]["verdict"].startswith("KILL")


def test_missing_counts_block(tmp_path):
    data, mdir = make_world(tmp_path)
    (data / "counts.jsonl").unlink()
    res = analyse_model(data, mdir, CFG)
    assert res["decision"]["verdict"].startswith("BLOCKED")


def test_mode_match_chance_definition():
    table = [{"correct": False, "pred": "a", "mode": "a", "mode_in_truth": False, "K": 5, "n_truth": 1,
              "freq_mode_share": 0.5, "mode_generic": "b", "cell": "x"},
             {"correct": False, "pred": "c", "mode": "b", "mode_in_truth": True, "K": 5, "n_truth": 1,
              "freq_mode_share": 0.0, "mode_generic": "c", "cell": "y"},
             {"correct": True, "pred": "b", "mode": "b", "mode_in_truth": True, "K": 5, "n_truth": 1,
              "freq_mode_share": 0.0, "mode_generic": "b", "cell": "y"}]
    s = mode_match_stats(table, n_boot=10)
    assert s["mode_match"] == 0.5 and math.isclose(s["chance_uniform"], (0.25 + 0) / 2)
    assert s["mode_match_given_mode_wrong"] == 1.0 and s["generic_mode_match"] == 0.5


def test_auroc_gain_zero_when_feature_is_noise():
    rng = np.random.default_rng(1)
    table = [{"slot": "citizenship", "cell": f"c{i % 10}", "correct": bool(rng.random() < 0.5),
              "c": int(rng.integers(0, 9)), "n": int(rng.integers(1, 9)), "s_m": float(rng.random())}
             for i in range(400)]
    g = auroc_gain(table, "s_m", use_counts=True, n_boot=50)
    assert abs(g["delta"]) < 0.08
