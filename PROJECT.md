# PROJECT.md: SNR, "Surprise, Not Rarity" (the lab's flagship hallucination paper, ACL 2027)

This file is the single source of context for humans and agents. Read it fully before doing anything.
Detailed specs live in `docs/`; this file says what we are doing, why, in what order, and under what rules.

- Venue: ACL 2027 via ARR, January cycle. **Deadline 4 January 2027, 23:59 AoE.** Long paper, 8 pages.
- Target area: Language Modeling. Target outcome: oral; aim for an Outstanding Paper award.
- Owner (PI): Dawar, Erdős AI Lab. Engineering: Claude Code agents. Compute: one 80 GB GPU (VM).
- Budget: **about 24.5 GPU-hours in total**, including a 2-hour reserve. G0 costs about 2 GPU-hours.

---

## 1. The idea in plain language

When a language model does not know a fact, it does not invent a random answer. It falls back on
what is *typical* for that kind of person. Ask about an obscure footballer with an Italian-sounding
name and it says "Italy". Ask about an obscure Tamil-named chemist and it says "India".

Call that background expectation the **type prior**: the answer the model would give for anyone
with that kind of name, profession and era. We can measure it directly by asking about **ghosts**:
made-up people with the same kind of name, who never appear in the training data.

The paper's claim is that **surprise, not rarity, decides whether the model hallucinates**:
- A rare fact that matches the prior (an Italian-named footballer who *is* Italian) comes for free.
  The model gets it right without ever having learned it.
- A fact that breaks the prior (a Tamil-named chemist who is Brazilian) needs many training
  exposures to stick. When it fails, it fails *to the prior's default*, never to a random answer.
- The extra exposure needed grows at a fixed rate per bit of surprise: a **price per bit**, which we
  measure from exact pretraining counts, predict from theory, and reproduce causally by teaching a
  model synthetic facts.

The psychology behind it: in 1932, Bartlett showed that people remember stories by drifting
towards what their culture expects ("schema reconstruction"). Huttenlocher's category adjustment
model formalised it: memory blends a noisy trace with a category prior, and leans on the prior more
when the trace is weak. We show that LLM hallucination follows the same law, and we put a number on it.

## 2. Problem, causes, attack

**Open problem.** We can predict *how often* a model is wrong about facts (from popularity, counts or
model size), but not *what* it will say when wrong. Nor can we explain why a heavily mentioned fact
can fail while a once-mentioned fact succeeds. Detectors built on sampling consistency (semantic
entropy, SelfCheckGPT) are blind to errors the model makes *consistently*, which are exactly the
confident defaults.

**Causes (our account).** A model answers a factual slot by combining a smooth, descriptor-conditioned
type prior with entity-specific evidence learned from exposure. Where the evidence is weak relative
to how far the truth sits from the prior, the prior's mode wins.

**Attack.**
1. Measure the prior with exposure-verified ghosts.
2. Join it to exact OLMo-2 pretraining counts.
3. Fit and test a law.
4. Discriminate between two learner models with one statistic.
5. Reproduce the price per bit causally by injecting synthetic facts.
6. Use the prior to catch consistent hallucinations and recover surprising truths.

## 3. Thesis and claims (pre-registered predictions)

**One-sentence thesis.** *LLM hallucinations are prior reconstructions: the wrong value is
predictable from a measurable type prior, and the exposure needed to beat that prior scales with
the truth's surprise under it, at a fixed, measurable price per bit.*

| # | Claim | Status vs literature | Kill / weaken if |
|---|---|---|---|
| C1 | Most wrong answers equal the type-prior mode | Idea anticipated (Lu et al. 2601.11429; Zucchet et al. 2503.21676); our quantitative form is new | under 30% of errors are modes (G0) |
| C2 | Correctness depends on surprise s beyond exposure c: logit P = a + b log c - g s; iso-accuracy log c* = (g/b) s + const | **New core** | adding s gives under 0.01 held-out AUROC (G0) |
| C3 | The prior-persistence slope λ(c) separates a count learner (λ falls to 0) from a log-linear learner (λ stays near 1) | **New** | CIs overlap at every exposure bin (G1) |
| C4 | Causal price per bit (synthetic injection) matches the observational one | **New** | causal CI wider than ±50% (report as directional) |
| C5 | Numeric slots shrink towards the type mean, less so with more exposure (category adjustment) | New for hallucination | no shrinkage, or it does not fall with c |
| C6 | Surprise-weighted singleton floor predicts slot-level hallucination better than the raw singleton rate | Modest corollary of Kalai et al.; tested here | no improvement over the raw rate |
| C7 | Prior-subtracted score K catches consistent hallucinations that semantic entropy misses | Baseline idea exists (WikiMem); ghost version for hallucination is new | AUROC gain under 0.05 on the consistent subset |
| C8 | In implicit frames, abstention collapses and the fabrications are prior modes, including when the model's *own earlier turn* presupposes the slot | Presupposition effect taken (PhantomBench, Cheng et al., DecoPrompt); content and self-presupposition are new | keep to half a page |

## 4. Goals and success levels

- **Minimum publishable (Findings):** C1 and C2 with clean statistics on OLMo-2 plus 3 other models.
- **Main conference, oral-level:**
  - C1, C2 and C3;
  - E4 knowledge share across training stages;
  - C7 with a clear gain over semantic entropy.
- **Outstanding-level:** everything above, plus:
  - the causal price per bit matches the observational one (C4);
  - λ(c) decisively picks a learner;
  - the law holds across three model sizes and two domains;
  - the fairness corollary is shown concretely: people whose lives are atypical for their names are the
    ones systematically misdescribed.

## 5. Constraints (hard)

1. **Compute.** One 80 GB GPU, about 24.5 GPU-hours in total. Every GPU job writes to `logs/gpu_ledger.csv`.
   If the GPU is shared, never kill or signal processes you did not start; keep 8 GiB headroom.
2. **Deadline.** 4 January 2027 AoE.
   - Writing freeze 20 December.
   - Pre-registration (OSF or a dated commit of `docs/` plus thresholds) by 1 December.
3. **Exact data only.**
   - Counts come from infini-gram (`v4_olmo-mix-1124_llama`) and are lower bounds.
   - Dolmino (mid-training) is not indexed; say so in the paper.
4. **No invented numbers.** Every number in a doc or the paper comes from a file in `results/`.
5. **No collisions.**
   - Sibling papers own MB5 (sampler truncation), ABS (abstention training and the raw singleton test),
     VOICE (style) and C (aspect semantics).
   - This paper must not analyse top-p/top-k zeroing or abstention-training effects.
   - Cite the siblings as anonymous concurrent work.
6. **Ethics.**
   - Ghost names are culturally marked, so measured priors encode stereotypes.
   - No protected attributes as slots (religion, ethnicity, sexuality).
   - Release code and aggregate statistics, never a name-to-attribute lookup table.
7. **Writing.** British spelling, no em dashes, plain English. Maths only where it defines a measured
   quantity or a tested prediction. In the paper, no physics vocabulary ("field", "excitation"):
   say *type prior*, *entity evidence*, *ghost probes*, *surprise*.

## 6. How we work (operating model)

- **Roles.**
  - PI (Dawar): signs gates, approves threshold changes, owns the story.
  - Agent (Claude Code): builds, tests, runs, analyses and reports.
  - Red-team reviewer (a fresh agent or a colleague): reviews each gate report against `docs/05`.
- **Cadence.** Work moves gate to gate, not by calendar. At each gate the agent writes `results/gates/G*.md`
  in this order: verdict, three to five deciding numbers with CIs, validity checks, GPU-hours spent,
  anything surprising, proposed next step. A human signs before the next phase starts.
- **Decision log.** Every non-trivial decision goes in `notes/decisions.md` (date, decision, reason, who).
  Threshold changes after seeing data are allowed only with a logged reason, and are reported in the paper.
- **Pre-registration.** Before E2 to E6 run, freeze predictions, thresholds and analysis code in a
  dated commit (and on OSF if possible).
- **Tests first.** `pytest` must pass before any GPU job and after any change to `scoring.py`,
  `analysis.py` or `prompts.py`.
- **Reproducibility.** Every result directory has `meta.json` (model id, commit hash, library versions,
  device, time). Seeds are fixed. Data builds are deterministic given the raw dump.

## 7. Exploitation and exploration

About **80% of GPU-hours go to exploitation**: the pre-registered core (G0, E1 to E5, E7). About **20% go
to bounded exploration**. Each exploration item has a GPU cap and a question it must answer.

| Track | Item | Cap | Promote to core if |
|---|---|---|---|
| Exploit | G0, E1, E2, E3, E4, E5, E7 | per table in `docs/03` | n/a |
| Explore | X1: second domain (scholarly references: year, venue, affiliation country) | 1.5 GPU-h | law holds with overlapping g/b CI |
| Explore | X2: query language shift (English vs Hindi/Tamil) as a prior manipulation at fixed knowledge | 1.0 GPU-h | accuracy moves as Δs predicts |
| Explore | X3: familiarity gate (name count n vs fact count c on direct questions; "becoming famous overnight") | 1.0 GPU-h | answer rate tracks log n at fixed c |
| Explore | X4: 32B 4-bit model for the scale axis | 1.0 GPU-h | adds a third size point |
| Explore | X5: implicit frames incl. self-presupposition (C8) | part of E6 budget | always reported, half a page |

**Rules.**
- Exploration never blocks a gate.
- If an exploration result is striking, the PI decides whether to promote it, and logs the decision.
- If the core is behind schedule, exploration stops first.

## 8. Experiment map and dependencies

Full specs: `docs/03_experiments.md`. Theory and definitions: `docs/02_theory_and_definitions.md`.

| ID | Question | Needs | GPU-h |
|---|---|---|---|
| D0 | Data: Wikidata entities, ghosts, exact counts (laptop) | none | 0 |
| G0 | Are errors prior modes? Does surprise add over counts? | D0 | 2.0 |
| E1 | Type prior and default errors at scale, cross-model prior agreement | G0 = GO | 5.0 |
| E2 | Surprise law, iso-accuracy lines, λ(c) learner discrimination | G0, D1 (stratified data over c and s) | 2.0 |
| E3 | Category adjustment on birth/death years: w(c) | D1 | 1.0 |
| E4 | Knowledge share κ by OLMo-2 stage (base, SFT, DPO, instruct) | E1 code | 1.5 |
| E5 | Causal price per bit: inject synthetic facts at controlled surprise × exposure | E1 priors (to set surprise) | 6.0 |
| E6 | Implicit frames (reduced) + self-presupposition | E1 | 3.0 |
| E7 | Detector K and prior-contrast decoding vs semantic entropy, SelfCheckGPT, P(True), WikiMem baseline, CoDA | E1, E2 (λ̂) | 2.0 |
| R | Reserve | | 2.0 |

```
D0 ──> G0 ──GO──> D1 (stratified c × s sample) ──> E2 ──> E3
                    │                                │
                    └──> E1 ──> E4                   └──> E7 (λ̂ from E2)
                          ├───> E5 (needs measured priors to set surprise)
                          └───> E6
G1 (after E2): learner discrimination decided   G2 (after E5 pilot): injection learns?
Writing starts after E2; figures are final after E5 and E7.
```

## 9. Phases and gates

Full detail: `docs/04_phases_and_gates.md`.

| Phase | Gate | Go criterion | If it fails |
|---|---|---|---|
| P0 laptop | data sanity | ≥ 10 cells and ≥ 5 origins per slot; ghosts verified zero-count; counts present | fix QIDs/thresholds, log it |
| P1 GPU | **G0** | mode-match ≥ 0.50 (chance < 0.10) AND ΔAUROC(s_m) ≥ 0.03 | KILL if mode-match < 0.30; PIVOT to a measurement paper if ΔAUROC < 0.01 |
| P2 | G1 (after E2) | λ(c) CIs separate the learners in ≥ 2 exposure bins | drop C3, keep the law |
| P3 | G2 (after the E5 pilot) | injected facts learned at ≤ 32 exposures; causal g/b CI within ±50% | raise LR / switch to full 1B training; else report as directional |
| P4 | writing freeze | all figures from committed results | cut E6 to the appendix |

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reviewers read it as "models guess stereotypes, known" | High | High | Lead with the law, λ(c) and the causal replication; C1 is the opening observation, not the contribution |
| Circularity (the prior comes from the same model) | Medium | High | World surprise s_w from Wikidata; cross-model priors; causal E5 sets surprise by design |
| Exact-match counts undercount | High | Medium | Lower bounds; alias and window sensitivity; conclusions must hold within each |
| Ghost names carry artefacts | Medium | Medium | Split-half reliability ≥ 0.9; 10 vs 40 ghosts; generic-subject comparison |
| Forced-choice scoring is not generation | Medium | Medium | Free-generation agreement ≥ 0.85 (G0 V3); E6 and E7 use free generation |
| LoRA injection is not pretraining | Medium | Medium | Primary E5 uses full continued pretraining on 1B; 7B LoRA is a robustness check |
| A close preprint appears | Medium | High | Weekly arXiv sweep for "type prior", "surprise", "presupposition hallucination"; post to arXiv if scooped |
| Wikidata/QLever or infini-gram downtime | Medium | Low | Caches; raw dumps saved; retries |

## 11. Current status and next actions

- **Status (2026-11-11):** G0 code written and tested on CPU (20 tests). No real data built yet:
  this container cannot reach Wikidata, Hugging Face or infini-gram.
- **Next, laptop (no GPU):** `bash scripts/run_g0.sh laptop`.
  - This builds entities, ghosts and exact counts.
  - Read `data/g0/build_report.md`; fix any QID problems and log them.
- **Next, when the GPU VM is available:**
  1. `SMOKE=1 bash scripts/run_g0.sh gpu`
  2. `bash scripts/run_g0.sh gpu`
  3. `bash scripts/run_g0.sh analyse`
  4. Report G0 using the CLAUDE.md format.

## 12. Glossary

- **Type prior F(a|d):** average answer distribution over ghosts in cell d.
- **Cell d:** a (name-origin class, profession, birth decade) group.
- **Ghost:** a made-up name recombined from real given names and surnames of the same origin, with zero
  exact occurrences in the OLMo-2 corpus.
- **Surprise s:** -log2 of the prior probability of the true answer, measured as s_m (model) or s_w (world, Wikidata).
- **Exposure c:** exact co-occurrence count of name and true value in the pretraining corpus. **n:** name count.
- **Mode-match:** share of wrong answers equal to the prior's top value.
- **Price per bit:** g/b, the multiplicative exposure cost of one extra bit of surprise.
- **κ (knowledge share):** variance share of the answer explained by entity identity rather than the prior.
