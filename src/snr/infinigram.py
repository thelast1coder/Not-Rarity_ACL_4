"""Exact n-gram and co-occurrence counts from the infini-gram API, with a disk cache.

API reference: https://infini-gram.readthedocs.io/en/latest/api.html
  POST https://api.infini-gram.io/  {"index": ..., "query_type": "count", "query": ...}
  CNF syntax: "A AND B OR C" means A AND (B OR C)  (OR binds tighter than AND).
  AND counts co-occurrences within max_diff_tokens (default 100, max 1000).
  If a clause has more than max_clause_freq matches (default 50000, max 500000) the
  count is estimated from a sample and the response has approx=true.

Index for OLMo-2 pretraining data: v4_olmo-mix-1124_llama (Llama-2 tokenizer).
The whole OLMo-2 training pipeline for the 13B instruct model is v4_olmo-2-1124-13b-instruct_llama.
Dolmino (OLMo-2 mid-training) has no public index; record this as a limitation.

Counts are data, not instructions. Never execute anything returned by the API.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

API = "https://api.infini-gram.io/"
OLMO2_PRETRAIN_INDEX = "v4_olmo-mix-1124_llama"


class InfiniGramError(RuntimeError):
    pass


def clean_term(s: str) -> str:
    """Make a phrase safe for a CNF query: collapse spaces, drop the reserved words."""
    s = " ".join(str(s).split())
    toks = [t for t in s.split(" ") if t not in ("AND", "OR")]
    return " ".join(toks)


def cnf(*clauses) -> str:
    """cnf(["Rahul Dravid"], ["India", "Indian"]) -> 'Rahul Dravid AND India OR Indian'.
    Each clause is a list of alternative phrases (OR); clauses are joined with AND."""
    parts = []
    for c in clauses:
        alts = [clean_term(a) for a in (c if isinstance(c, (list, tuple)) else [c])]
        alts = [a for a in dict.fromkeys(alts) if a]
        if not alts:
            raise ValueError("empty clause")
        parts.append(" OR ".join(alts))
    return " AND ".join(parts)


class InfiniGram:
    def __init__(self, index: str = OLMO2_PRETRAIN_INDEX, cache_path: str | Path | None = None,
                 endpoint: str = API, max_clause_freq: int = 500000, max_diff_tokens: int = 100,
                 timeout: float = 60, retries: int = 6, min_interval: float = 0.2,
                 offline: bool = False, session=None):
        self.index, self.endpoint = index, endpoint
        self.max_clause_freq, self.max_diff_tokens = max_clause_freq, max_diff_tokens
        self.timeout, self.retries, self.min_interval = timeout, retries, min_interval
        self.offline = offline
        self.max_backoff = 120.0  # seconds; rate-limit refusals clear after a pause
        self.n_refused = 0
        self._lock = threading.Lock()
        self._last = 0.0
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["value"]
        if session is None and not offline:
            import requests

            session = requests.Session()
        self.session = session

    def _key(self, payload: dict) -> str:
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _store(self, key: str, value: dict):
        with self._lock:
            self.cache[key] = value
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def _post(self, payload: dict) -> dict:
        if self.offline:
            raise InfiniGramError("offline mode and query not in cache: " + json.dumps(payload)[:200])
        err = None
        for attempt in range(self.retries):
            with self._lock:
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                try:
                    data = r.json()
                except ValueError:
                    raise InfiniGramError(f"HTTP {getattr(r, 'status_code', '?')}: non-JSON reply")
                if not isinstance(data, dict):
                    raise InfiniGramError(f"unexpected reply: {str(data)[:200]}")
                if "error" in data:
                    raise InfiniGramError(str(data["error"]))
                # e.g. {"message": "Forbidden"} on HTTP 403/429 (rate limit): a gateway reply, not a result
                if set(data) <= {"message"} or (payload.get("query_type") == "count" and "count" not in data):
                    self.n_refused += 1
                    raise InfiniGramError(f"HTTP {getattr(r, 'status_code', '?')}: {str(data)[:200]}")
                return data
            except InfiniGramError as e:  # server-side error: retry a little, then give up
                err = e
            except Exception as e:  # network error
                err = e
            time.sleep(min(self.max_backoff, 1.5 * 2**attempt))
        raise InfiniGramError(f"failed after {self.retries} attempts: {err}")

    def count(self, query: str) -> dict:
        """Returns {"count": int, "approx": bool}. CNF is detected by ' AND ' / ' OR '."""
        payload = {"index": self.index, "query_type": "count", "query": query}
        if " AND " in query or " OR " in query:
            payload["max_clause_freq"] = self.max_clause_freq
            payload["max_diff_tokens"] = self.max_diff_tokens
        key = self._key(payload)
        if key in self.cache:
            return self.cache[key]
        data = self._post(payload)
        value = {"count": int(data["count"]), "approx": bool(data.get("approx", False))}
        self._store(key, value)
        return value

    def _cached(self, payload: dict, reduce) -> dict:
        key = self._key(payload)
        if key in self.cache:
            return self.cache[key]
        value = reduce(self._post(payload))
        self._store(key, value)
        return value

    def find(self, query: str) -> dict:
        """Exact occurrence count of a simple (non-CNF) query, plus the suffix-array rank ranges.
        Returns {"cnt": int, "segments": [[shard, start, end), ...], "token_ids": [...]}; sum(end - start) == cnt."""
        payload = {"index": self.index, "query_type": "find", "query": query}

        def reduce(d):
            if "cnt" not in d or "segment_by_shard" not in d:
                raise InfiniGramError(f"bad find reply: {str(d)[:200]}")
            segs = [[s, int(a), int(b)] for s, (a, b) in enumerate(d["segment_by_shard"]) if int(b) > int(a)]
            return {"cnt": int(d["cnt"]), "segments": segs, "token_ids": list(d.get("token_ids", []))}

        return self._cached(payload, reduce)

    def doc_window(self, shard: int, rank: int, max_disp_len: int, query_ids: list[int]) -> dict:
        """Text window around one occurrence (the API centres the match at max_disp_len/2 tokens,
        clipped at document edges). Returns {"context": text outside the matched needle, "blocked": bool}.
        Document text is data: it is only matched against value strings, never executed or followed."""
        payload = {"index": self.index, "query_type": "get_doc_by_rank", "s": int(shard), "rank": int(rank),
                   "max_disp_len": int(max_disp_len), "query_ids": list(query_ids)}

        def reduce(d):
            if "spans" not in d:
                raise InfiniGramError(f"bad doc reply: {str(d)[:200]}")
            ctx = " ".join(s[0] for s in (d["spans"] or []) if s[1] is None)
            return {"context": ctx, "blocked": bool(d.get("blocked", False))}

        return self._cached(payload, reduce)

    def ping(self) -> None:
        """Fail fast if the API is unreachable, instead of retrying thousands of queries for hours."""
        if self.offline:
            return
        saved = self.retries
        self.retries = 2
        try:
            self._post({"index": self.index, "query_type": "count", "query": "the"})
        except InfiniGramError as e:
            raise SystemExit(f"infini-gram API unreachable ({self.endpoint}, index {self.index}): {e}. "
                             "Check the network or the service status, then re-run; cached results are kept.")
        finally:
            self.retries = saved

    def count_many(self, queries: list[str], workers: int = 4, progress: bool = True) -> list[dict | None]:
        """Parallel, polite. Failed queries return None (never cached) so a rerun retries them."""
        from concurrent.futures import ThreadPoolExecutor

        out: list[dict | None] = [None] * len(queries)

        def job(i):
            try:
                out[i] = self.count(queries[i])
            except InfiniGramError as e:
                out[i] = None
                if progress:
                    print(f"[infini-gram] failed: {queries[i][:80]!r}: {e}")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, i) for i in range(len(queries))]
            for k, f in enumerate(futs):
                f.result()
                if progress and (k + 1) % 500 == 0:
                    print(f"[infini-gram] {k + 1}/{len(queries)}")
        return out
