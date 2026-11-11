"""Name-side exposure counts: sampling, exactness, the estimator and whole-word matching (no network)."""
import random

from snr.counts import forms_regex, name_side_count, sample_occurrences
from snr.infinigram import InfiniGram


class FakeResp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class FakeCorpus:
    """One name with occurrences spread over shards; occurrence i co-occurs with the value iff i in `hits`."""

    def __init__(self, segments, hits):
        self.segments, self.hits, self.calls = segments, set(hits), []
        self.flat = [(s, r) for s, (a, b) in enumerate(segments) for r in range(a, b)]

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        if json["query_type"] == "find":
            return FakeResp({"cnt": len(self.flat), "segment_by_shard": self.segments, "token_ids": [1, 2, 3]})
        assert json["query_type"] == "get_doc_by_rank"
        assert json["max_disp_len"] == 2 * (100 + 3)
        i = self.flat.index((json["s"], json["rank"]))
        ctx = "she was born in Italy and" if i in self.hits else "she played in Indiana with Italyx"
        return FakeResp({"spans": [[ctx, None], ["Ada Rossi", "0"], [" later.", None]], "blocked": False})


def test_sample_covers_all_when_small_and_maps_ranks():
    segs = [[0, 10, 13], [1, 0, 0], [2, 5, 7]]
    assert sample_occurrences(segs, 5, 50, "k") == [(0, 10), (0, 11), (0, 12), (2, 5), (2, 6)]


def test_sample_is_seeded_uniform_and_in_range():
    segs = [[0, 100, 400], [3, 0, 700]]
    a = sample_occurrences(segs, 1000, 50, "item1")
    assert a == sample_occurrences(segs, 1000, 50, "item1") and len(set(a)) == 50
    assert all((s == 0 and 100 <= r < 400) or (s == 3 and 0 <= r < 700) for s, r in a)
    assert a != sample_occurrences(segs, 1000, 50, "item2")


def test_whole_word_case_sensitive_match():
    rx = forms_regex(["India", "Indian", "U.S."])
    assert rx.search("an Indian chemist") and rx.search("the U.S. team")
    assert not rx.search("Indiana Jones") and not rx.search("india") and not rx.search("XU.S.A")


def test_exact_when_n_le_k(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    corpus = FakeCorpus([[0, 6], [10, 14]], hits={0, 3, 9})
    ig = InfiniGram(cache_path=tmp_path / "c.jsonl", session=corpus, min_interval=0)
    r = name_side_count(ig, "Ada Rossi", ["Italy", "Italian"], k=50, window=100, seed_key="x")
    assert r == {"n": 10, "n_approx": False, "c": 3, "c_approx": False, "c_hits": 3, "c_sampled": 10}
    n_calls = len(corpus.calls)
    name_side_count(ig, "Ada Rossi", ["Italy"], k=50, window=100, seed_key="x")
    assert len(corpus.calls) == n_calls  # find and windows are cached


def test_estimate_unbiased_when_n_gt_k(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    rng = random.Random(0)
    hits = set(rng.sample(range(2000), 300))  # true c = 300
    corpus = FakeCorpus([[0, 1200], [0, 800]], hits=hits)
    ig = InfiniGram(cache_path=None, session=corpus, min_interval=0)
    ests = [name_side_count(ig, "Ada Rossi", ["Italy"], k=50, window=100, seed_key=f"s{i}")["c"] for i in range(200)]
    mean = sum(ests) / len(ests)
    assert abs(mean - 300) < 25  # sd of the mean is about 7
    r = name_side_count(ig, "Ada Rossi", ["Italy"], k=50, window=100, seed_key="s0")
    assert r["c_approx"] and r["c_sampled"] == 50 and r["n"] == 2000


def test_zero_name_count_is_exact_zero(tmp_path, monkeypatch):
    corpus = FakeCorpus([], hits=set())
    ig = InfiniGram(cache_path=None, session=corpus, min_interval=0)
    r = name_side_count(ig, "Nobody Here", ["Italy"], k=50, window=100, seed_key="z")
    assert r["n"] == 0 and r["c"] == 0 and not r["c_approx"]
    assert [c["query_type"] for c in corpus.calls] == ["find"]
