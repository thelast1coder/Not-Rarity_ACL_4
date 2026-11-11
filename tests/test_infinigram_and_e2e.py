import json
from pathlib import Path

import pytest
from conftest import small_cfg, synthetic_rows

from snr.analysis import analyse_model, render
from snr.build_entities import build, write_outputs
from snr.common import read_jsonl, write_json, write_jsonl
from snr.ghosts import make_ghosts
from snr.infinigram import InfiniGram, InfiniGramError
from snr.run_g0 import run


class FakeResp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        if len(self.calls) <= self.fail_first:
            raise ConnectionError("boom")
        if "AND" in json["query"]:
            assert json["max_diff_tokens"] == 100 and json["max_clause_freq"] == 500000
            return FakeResp({"count": 3, "approx": True})
        return FakeResp({"count": 42, "approx": False})


def test_infinigram_cache_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = FakeSession(fail_first=1)
    ig = InfiniGram(cache_path=tmp_path / "c.jsonl", session=s, min_interval=0)
    assert ig.count("Rahul Sharma") == {"count": 42, "approx": False}
    assert ig.count("Rahul Sharma AND India OR Indian") == {"count": 3, "approx": True}
    n = len(s.calls)
    ig.count("Rahul Sharma")
    assert len(s.calls) == n  # cached
    ig2 = InfiniGram(cache_path=tmp_path / "c.jsonl", offline=True)
    assert ig2.count("Rahul Sharma")["count"] == 42  # cache survives restart
    with pytest.raises(InfiniGramError):
        ig2.count("never asked")


def test_server_error_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class ErrSession:
        def post(self, url, json=None, timeout=None):
            return FakeResp({"error": "bad query"})

    ig = InfiniGram(cache_path=tmp_path / "c.jsonl", session=ErrSession(), retries=2, min_interval=0)
    out = ig.count_many(["x"], workers=1, progress=False)
    assert out == [None]
    assert not (tmp_path / "c.jsonl").exists()


def test_forbidden_reply_is_retried_then_error(tmp_path, monkeypatch):
    """A rate-limit reply ({"message": "Forbidden"}, no count) must retry, then raise; never cache or KeyError."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    class ForbiddenThenOk:
        def __init__(self, n_forbidden):
            self.n, self.calls = n_forbidden, 0

        def post(self, url, json=None, timeout=None):
            self.calls += 1
            if self.calls <= self.n:
                return FakeResp({"message": "Forbidden"})
            return FakeResp({"count": 0, "approx": False})

    ig = InfiniGram(cache_path=tmp_path / "c.jsonl", session=ForbiddenThenOk(2), retries=3, min_interval=0)
    assert ig.count("Osmo Lehtinen") == {"count": 0, "approx": False}  # recovered after backoff

    ig2 = InfiniGram(cache_path=tmp_path / "d.jsonl", session=ForbiddenThenOk(99), retries=2, min_interval=0)
    with pytest.raises(InfiniGramError, match="Forbidden"):
        ig2.count("Osmo Lehtinen")
    assert not (tmp_path / "d.jsonl").exists()


@pytest.mark.parametrize("chat", [False, True])
def test_end_to_end_tiny(tmp_path, tok, model, chat):
    res = build(synthetic_rows(), small_cfg())
    data = tmp_path / "data"
    write_outputs(res, data)
    write_json(data / "ghosts.json", make_ghosts(res["name_pools"], set(res["real_names"]), n_per_origin=6))
    items = read_jsonl(data / "items.jsonl")
    write_jsonl(data / "counts.jsonl", [{"item_id": it["item_id"], "c": i % 7, "n": 10 + i} for i, it in enumerate(items)])
    od = run(model, tok, "local/tiny-model", chat, data, tmp_path / "results", batch_size=8, progress=False)
    assert (od / "scores.jsonl.gz").exists() and (od / "freegen.jsonl").exists()
    cfg = {"analysis": {"w_alpha": 0.5},
           "gate": {"primary_model_tag": "tiny-model", "go_mode_match": 0.5, "max_chance": 0.1,
                    "go_delta_auroc": 0.03, "kill_mode_match": 0.3, "pivot_delta_auroc": 0.01,
                    "min_reliability": 0.9, "min_freegen_agreement": 0.85, "min_wrong": 5}}
    r = analyse_model(data, od, cfg)
    assert r["mode_match"]["n_items"] == len(items)
    md = render([r], "tiny-model")
    assert "Verdict:" in md
    json.dumps(r)  # serialisable
