# Experiments: design, inputs, outputs, dependencies, status

Status codes: READY (code done and tested), TODO (spec only), RUNNING, DONE.

## D0: G0 data (laptop, no GPU). READY

**Wikidata (QLever endpoint).**
- Humans born 1900-2000 with an English label and a citizenship.
- Two groups:
  - athletes, with sport (P641);
  - scientists, with field of work (P101) and an occupation in a fixed list.
- Commands:
  - `python -m snr.build_entities` (saves `data/raw_rows.jsonl`, then builds);
  - `python -m snr.ghosts`;
  - `python -m snr.counts`.

**Name-origin class.** Leave-one-out modal citizenship among *other* people with the same surname.
- Fallback: same given name.
- Requirements: support ≥ 3, purity ≥ 0.5.

**Cells and selection.**
- Cell = (origin, profession, decade).
- G0 selects up to 30 cells per slot, at most 3 per origin, about 300 items per slot spread across popularity.

**Ghosts.** 20 per origin. Each recombines a real given name and a real surname from the same origin
pool, with **zero** exact-match count in `v4_olmo-mix-1124_llama`.

**Counts.**
- n = exact count of the full name.
- c = occurrences of the name with a surface form of a true value within about 100 tokens, counted from the
  name side (exact if n ≤ 50, else estimated from a seeded sample of 50 occurrences). Infini-gram CNF counts
  are not used for c: they subsample common values and return c ≈ 0 for rare names.

**Outputs** in `data/g0/`:
- `items.jsonl`, `cells.json`, `candidates.json`, `name_pools.json`, `ghosts.json`, `counts.jsonl`;
- `build_report.md`;
- `infinigram_cache.jsonl`.

## G0: are errors prior modes, and does surprise matter? READY (about 2 GPU-hours)

- **Models:** OLMo-2-1124-7B-Instruct (primary, chat template) and OLMo-2-1124-7B (base prompt).
- **Prompts:** see `src/snr/prompts.py`. Real entity, ghosts and generic baseline use *the same template*;
  only the name changes.
- **Scoring:** exact candidate log-probabilities over the top-K values per slot, plus free greedy
  generation mapped back to candidates (agreement check).

**Statistics** (`snr.analysis`):
1. **Mode-match.** The share of wrong argmax answers equal to argmax F, with a 95% cluster bootstrap CI over cells.
   - Chance baselines: uniform over wrong candidates, and frequency-matched.
   - Also report the generic-subject mode-match: the ghost prior must beat it to show descriptors matter.
2. **ΔAUROC.** Grouped 5-fold logistic regression (by cell) predicting correctness from [log c, log n, slot]
   with and without s_m (and separately s_w). Cluster bootstrap CI on the difference.

**Validity checks:**
- V1: split-half reliability of F ≥ 0.90.
- V2: ghosts verified zero-count.
- V3: free-generation agreement ≥ 0.85.
- V4: ≥ 60 wrong answers.
- V5: counts present for ≥ 95% of items.

**Gate:** see `configs/g0.yaml` and `docs/04`.

## D1: stratified main sample (laptop). TODO

- About 3,000 people, stratified jointly over log c (5 bins including c = 0 and c = 1) and s_w (5 bins),
  so that count and surprise are decorrelated by design.
- Slots: citizenship, country of birth, occupation, sport, instrument, field of work, educated-at country,
  language, plus numeric birth and death years.
- Multi-valued truths: any valid value counts as correct.
- Exclusions: leaky cells, and protected attributes.
- Requirement: re-use the G0 code. Extend `wikidata.py` with the extra slots, and add a `--stratify` mode to `build_entities.py`.

## E1: type prior and default errors at scale (5 GPU-h). TODO

- **Design:** all slots × 10 open models. Models:
  - OLMo-2 1B, 7B, 13B (base and instruct);
  - Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Mistral-7B;
  - OLMo-2-32B in 4-bit.
- **Outputs:**
  - mode-match per model and slot;
  - cross-model prior similarity (Jensen-Shannon);
  - "model A's prior predicts model B's errors" rate.
- **Code:** generalise `run_g0.py` into `run_scores.py` with a models list. `analysis` gains cross-model tables.

## E2: surprise law and learner discrimination (2 GPU-h). TODO

- **Design:** OLMo-2 1B/7B/13B, exact c and n, on D1.
- **Fits:**
  - mixed-effects logistic (statsmodels or a PyMC/bambi equivalent);
  - iso-accuracy lines;
  - λ(c) per exposure bin, with controls for answer-string length and unconditional value log-probability.
- **Gate G1:** λ(c) separates the learners in at least two bins.

## E3: category adjustment (1 GPU-h). TODO

- **Design:** score candidate years 1880-2010 for each entity and each ghost. ŷ is the expectation over years; μ_d is the ghost mean.
- **Fit:** w(c) by count bin.
- **Controls:** digit bias (Benford's Curse); use all entities.

## E4: knowledge share by training stage (1.5 GPU-h). TODO

- OLMo-2-7B checkpoints: base, SFT, DPO, instruct.
- Variance decomposition of logit p(a*|e) into prior and entity terms. Report κ with bootstrap CIs.

## E5: causal price per bit (6 GPU-h). TODO

**Arm 1: knowledge fixed, prior moved.** Query identical facts with descriptors that shift F:
- profession cue present or absent;
- English vs Hindi/Tamil (exploration X2);
- five same-cell ghosts sharing a value as in-context priming.

**Arm 2: prior fixed, knowledge moved.**
- 400 synthetic people, with ghost names from measured cells.
- Values assigned at controlled surprise: 5 bins, from the prior mode down to F < 0.01.
- Each fact shown 1, 2, 4, 8, 16 or 32 times in paraphrased documents, mixed into replay text.
- Training: continued pretraining of OLMo-2-1B, small learning rate, 5-10M tokens, 3 seeds. 7B LoRA with 1 seed is a robustness check.

**Readout:** the causal iso-accuracy line and λ(c). Success means the causal g/b falls inside the observational CI.

**Pilot first (1.5 GPU-h)** for gate G2.

## E6: implicit frames (3 GPU-h). TODO

**Sample:** facts the model abstains on when asked directly.

**Frames:**
- embedded question;
- user presupposition;
- an appositive the model must complete ("X, the ___-born chemist, ...");
- a citation-support sentence;
- a travel or biography plan;
- **self-presupposition** (the model's own earlier turn introduced the slot).

**Report:**
- the check gap (direct abstention minus frame abstention);
- the share of fabrications equal to the prior mode.

**Validation:** slot extraction checked on 300 items by two annotators (κ ≥ 0.8). This is about 6-8 human hours.

## E7: detection and decoding (2 GPU-h). TODO

**Detector.**
- Score: K(â) = log p(â|e) - log F(â|d), on extracted factual slots in long-form biographies.
- Baselines: semantic entropy, SelfCheckGPT, P(True), WikiMem generic subtraction, PMI-DC / contextual calibration.
- Primary metric: AUROC on the **consistent-hallucination subset** (errors with low semantic entropy).

**Decoding.**
- Rule: logit - λ̂ log F at factual slots, with λ̂ taken from E2 (not tuned). It fires only when K signals partial evidence; otherwise abstain.
- Baselines: CoDA (overshadowing paper), context-aware decoding, DoLa.
- Report: surprising truths recovered vs accuracy lost on typical truths.

## Exploration items (capped; see PROJECT.md section 7)

| Item | What | Cap |
|---|---|---|
| X1 | Second domain: scholarly references | 1.5 GPU-h |
| X2 | Query language as a prior shift | 1.0 GPU-h |
| X3 | Familiarity gate | 1.0 GPU-h |
| X4 | 32B scale point | 1.0 GPU-h |
