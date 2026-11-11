# Phases and gates (binding)

A gate report is `results/gates/G*.md`. A human signs it before the next phase starts.
Thresholds were pre-registered on 2026-11-11 in `configs/g0.yaml`. Any change after seeing data
must be logged in `notes/decisions.md` and reported in the paper.

## P0: laptop (no GPU)

1. `pytest` passes.
2. Wikidata build: every slot has ≥ 10 selected cells, ≥ 5 origins, and close to 300 items.
3. Ghosts: 20 per origin, all verified zero-count. Unverified ghosts are allowed only for the smoke test.
4. Counts: c and n present for ≥ 95% of items. The approx flag rate is recorded.

## P1: G0 (about 2 GPU-hours)

**Validity checks.** All must pass, or the verdict is provisional:
- V1: split-half reliability of F ≥ 0.90.
- V2: ghosts zero-count verified.
- V3: free-generation agreement ≥ 0.85.
- V4: ≥ 60 wrong answers.
- V5: counts available.

**Verdicts** (primary model: OLMo-2-1124-7B-Instruct; the base model is a replication):

| Verdict | Condition | Action |
|---|---|---|
| GO | mode-match ≥ 0.50, uniform chance < 0.10, ΔAUROC(s_m) ≥ 0.03 | build D1, run E1 and E2 |
| REVIEW | between thresholds | PI decides and logs it |
| PIVOT | ΔAUROC(s_m) < 0.01 | measurement paper only (C1 + E4 + E7); rethink C2 |
| KILL | mode-match < 0.30 | headline dead; write a postmortem in `notes/postmortem.md` |
| BLOCKED | counts missing or no wrong answers | fix data first |

Also flag a **circularity risk** if ΔAUROC(s_w) < 0.01 while ΔAUROC(s_m) ≥ 0.03.

## P2: G1 (after E2)

- **Go:** the λ(c) CIs separate the two learners in at least two exposure bins.
- **Otherwise:** drop C3 and keep the law.

## P3: G2 (after the E5 pilot, 1.5 GPU-hours)

**Check 1: are the injected facts learned at 32 exposures?**
- If not, raise the learning rate or switch to full 1B continued pretraining.

**Check 2: is the causal g/b CI within ±50%?**
- If not, report it as directional only.

## P4: writing

- Writing freeze is 20 December 2026.
- Red-team the paper against `docs/05_related_work_audit.md`.
- Run a final arXiv sweep in late December.
