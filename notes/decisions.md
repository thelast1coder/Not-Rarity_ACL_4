# SNR decision log

One entry per non-trivial decision: date, decision, reason, who. Threshold changes after seeing data
must be logged here and reported in the paper.

- 2026-11-11: Project set up from the research report "Default Showing Through". Core reframed around C2
  (surprise law, λ(c), causal price per bit). C1 is the opening observation; C8 (implicit frames) is cut
  to half a page because PhantomBench, Cheng et al. and DecoPrompt cover the presupposition effect. (PI + Claude)
- 2026-11-11: G0 thresholds pre-registered in configs/g0.yaml: mode-match ≥ 0.50 with chance < 0.10, and
  ΔAUROC(s_m) ≥ 0.03 → GO; mode-match < 0.30 → KILL; ΔAUROC < 0.01 → PIVOT. Primary model:
  OLMo-2-1124-7B-Instruct; base is a replication. (PI + Claude)
- 2026-11-11: Name-origin class = leave-one-out modal citizenship among other people with the same surname
  (fallback: given name), support ≥ 3, purity ≥ 0.5. It is never written into prompts. (Claude)
- 2026-11-11: G0 slots: citizenship (athletes and scientists), sport (athletes), field of work (scientists).
  Prompts state profession and birth decade only. Candidates are the top-K Wikidata values per slot. (Claude)
- 2026-11-11: Wikidata QIDs for scientist occupations were written from memory; verify them via the build report.
  The container used for setup could not reach Wikidata, Hugging Face or infini-gram, so G0 data has not been
  built yet. (Claude)
- 2026-11-11: Scientist occupation QIDs verified against Wikidata labels on qlever.dev: all nine are correct
  (Q901 scientist, Q1650915 researcher, Q169470 physicist, Q593644 chemist, Q170790 mathematician,
  Q864503 biologist, Q82594 computer scientist, Q188094 economist, Q1622272 university teacher). (Claude)
- 2026-11-11: QLever endpoint moved to https://qlever.dev/api/wikidata (the old uni-freiburg URL returns 308).
  Updated `wikidata.py`. (Claude)
- 2026-11-11: Data setting change, made before any model data was seen: `data.limit_per_group` raised from
  400,000 to 2,000,000. Reason: the athlete query hit the 400,000-row cap, and LIMIT without ORDER BY returned
  an arbitrary-order subset with zero US athletes. A COUNT query gives 992,835 athlete rows (885,712 people).
  `fetch_group` now raises an error if a result reaches the cap. The truncated raw dump was lost (overwritten
  when a resumed session re-ran the rename); the build from it is kept in `data/g0_truncated/` for reference only.
  The full dump is `data/raw_rows.jsonl` (athlete 992,835 rows, scientist 307,437); G0 data is rebuilt from it
  with `--raw`, and the rebuild reproduces the first full build exactly. No gate threshold changed. (Claude)
- 2026-11-11: File I/O in `common.py` (GPU ledger) set to explicit UTF-8; the laptop runs Windows (cp1252 default). (Claude)
- 2026-11-11: infini-gram started replying {"message": "Forbidden"} (rate limit) mid-run while a sibling project
  was also querying it. `infinigram._post` now treats any count reply without "count" as an error (retry with
  backoff, never cached), and the default request spacing is 0.2 s instead of 0.05 s. New test
  `test_forbidden_reply_is_retried_then_error`. Ghost verification restarted; cached answers are reused. (Claude)
- 2026-11-11 (OPEN, needs PI): Exposure count c is not usable as built. `snr.counts` gives c for 865/900 items
  (35 failed on infini-gram 403 rate limits), but 642/865 are approx, and only 85 of those have c > 0. Cause:
  the value clause (e.g. "New Zealand OR New Zealander") exceeds max_clause_freq (500,000), so infini-gram
  subsamples it and almost never catches a co-occurrence with a rare name; c collapses to 0 and is not a
  reliable lower bound. The 223 exact rows are genuine zeros (n = 0). 193/880 items have n = 0.
  Proposed fix: count from the name side (fetch each name occurrence with a +/-100-token window, match value
  forms locally; exact when n <= K, unbiased estimate when n > K). Cost from the n distribution:
  K=25: 12,830 calls (438/880 items exact); K=50: 23,208 (493); K=100: 41,592 (534). (Claude)
- 2026-11-11: DECIDED (PI): exposure c switches to name-side counting with K = 50 (`counts.method: name_side`).
  n comes from infini-gram `find` (exact). If n <= 50, every occurrence window is read and c is exact; else a
  seeded uniform sample of 50 gives c = n * hits / 50 (c_approx = True, unbiased). Window: max_disp_len =
  2 * (100 + L) tokens centred on the match (API behaviour checked by hand: match placed at max_disp_len/2,
  clipped at document edges). Value forms are matched as whole words, case-sensitive. The old CNF counts
  are kept in `data/g0/counts_cnf.jsonl` for comparison. Pilot on two citizenship items gave c = 0 for names
  whose occurrences are mostly other people (name ambiguity), so c is low but genuine. The run is paced at
  1 call/s with one worker: ABS (`abst.counts`) and VOICE (`voice.datalinks`) query infini-gram from the
  same machine and IP at the same time, and the shared rate limit returns 403 Forbidden. Tests:
  `tests/test_counts_name_side.py`. Docs 02 and 03 need the c definition updated. (PI + Claude)
