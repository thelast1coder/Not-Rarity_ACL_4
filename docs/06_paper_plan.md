# Paper plan (8 pages, ACL long)

## Title options

1. Surprise, Not Rarity: Language Models Hallucinate Their Priors
2. The Price of a Surprising Fact: A Law of Prior Reconstruction in Language Models
3. Remembering Like Bartlett: Hallucinations as Schema Reconstruction in LLMs
4. Typical Facts Are Free: Exposure, Surprise and the Shape of LLM Hallucinations

## Structure and page budget

1. **Introduction (1.0 p).** The Bartlett story in one paragraph, Figure 1, three contributions.
2. **Prior reconstruction: model and predictions (1.5 p).**
   - the two learners and λ(c);
   - the iso-accuracy law;
   - the surprise-weighted floor;
   - category adjustment.
3. **Measuring type priors with ghosts (0.75 p).** Construction, validation, exact counts.
4. **Hallucinations are defaults (0.75 p).** E1, cross-model priors.
5. **The price of surprise (1.5 p).** E2, E3, E4 and the causal E5.
6. **Where the check fails (0.5 p).** Reduced E6, including self-presupposition.
7. **Using the prior: detection and decoding (0.75 p).** E7.
8. **Related work (0.75 p).** Plus limitations and ethics.

**Appendix:**
- proofs;
- the ghost-list policy;
- count queries;
- full tables and per-model fits;
- the pre-registration;
- the annotation guide;
- failure cases.

## Figure 1

- **Left:** the type-prior map for one cell (e.g. Tamil-named scientists born in the 1950s), with real-entity wrong answers landing on the prior's mode.
- **Right:** the (surprise, log2 count) plane, coloured by accuracy, with the fitted iso-accuracy lines and the causal injection points overlaid.
  - The lower-left wedge is labelled "free (typical)".
  - The upper-right band is labelled "fails to the default".

## Headline claims (fill from results only)

- __% (CI __) of wrong greedy answers equal the ghost-prior mode, against __% chance. Model A's prior predicts model B's errors in __% of cases.
- At fixed exposure, each bit of surprise multiplies the exposure needed for 50% accuracy by __ (g/b = __).
- λ(c) stays at __ across four orders of magnitude of exposure.
- Causal g/b = __ vs observational __.
- Birth-year shrinkage 1 - w = __ at c < 10, falling to __ at c > 10^4.
- κ goes from __ (base) to __ (instruct).
- The floor's error is __ vs __ for the raw singleton rate.
- K reaches AUROC __ on consistent hallucinations, where semantic entropy reaches __.

## Reviewer objections and prepared answers

1. **"Obviously models guess plausible values."**
   - Plausibility is known (Lu et al.).
   - We contribute the quantitative law, the price per bit, its causal replication and the learner discrimination.
2. **"Circular: the prior comes from the same model."**
   - Ghosts have verified zero exposure.
   - s_w from Wikidata gives the same law.
   - Cross-model priors predict errors.
   - E5 sets surprise by design.
3. **"Exact-match counts are poor proxies."**
   - They are lower bounds.
   - We run alias, window and stage-index sensitivity checks; the conclusions must hold within each.
4. **"Ghost names carry artefacts."**
   - Split-half reliability is checked.
   - We compare 10 vs 40 ghosts.
   - We compare against the generic-subject baseline.
5. **"This is knowledge overshadowing."**
   - Overshadowing's log-linearity is what the multiplicative learner predicts. We explain it and test where it breaks.
6. **"Presupposition effects are known."** Agreed, and cited. We claim only the fabricated content and self-presupposition.
7. **"LoRA injection is not pretraining."** The primary result uses full continued pretraining on 1B.
8. **"Forced choice is not generation."** Free-generation agreement is reported, and E6 and E7 use free generation.

## Ethics paragraph (draft)

Ghost names are culturally marked, so measured priors encode stereotypes. We avoid protected
attributes as slots, release only code and aggregate statistics, and present the fairness result as
harm documentation: people whose lives are atypical for their names are the ones systematically
misdescribed. Real-person data is limited to public Wikidata fields.
