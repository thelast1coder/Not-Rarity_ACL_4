# Theory and definitions (the code must match this file)

## Objects

- Entity e, slot r (citizenship, sport, field of work, birth year, ...), candidate values a ∈ A_r, truth set T_e.
- Descriptors d(e): the cues available in the prompt. These are:
  - the name-origin class (carried by the name itself, never stated in the prompt);
  - the stated profession;
  - the stated birth decade.
- Model slot distribution: p(a | x) = softmax over candidates of log p(candidate string | prompt x).
  The candidate log-probability is the exact sum over its tokens (`scoring.py`).

## Measured quantities

| Symbol | Definition | Code |
|---|---|---|
| F(a\|d) | type prior: mean of p(a\|ghost) over the cell's ghosts (20 per origin) | `analysis.assemble` |
| G(a\|d) | generic-subject baseline: "This person is a ... born in the ...s" (WikiMem-style) | same |
| s_m(e) | model surprise: -log2 Σ_{a∈T_e} F(a\|d(e)) | same |
| s_w(e) | world surprise: -log2 Σ_{a∈T_e} W_loo(a\|d(e)), Wikidata cell distribution, leave-one-out, add-0.5 | same |
| c_e | co-occurrence count: occurrences of the exact full name with a surface form of a true value (whole word) within about 100 tokens, OLMo-2 corpus. Counted from the name side: exact if n_e ≤ 50, else n_e × (share of a seeded sample of 50 occurrences); see `notes/decisions.md` 2026-11-11 | `counts.py` |
| n_e | exact count of the full name | `counts.py` |
| K_e(a) | entity evidence: log p(a\|e) - log F(a\|d(e)) | E7 |
| mode | argmax_a F(a\|d) | G0 |

## Two learner models with different predictions (claim C3)

**(i) Additive (Dirichlet-multinomial count) learner.**
- Setup: prior Dir(α F(·|d)); the truth a* is observed c times.
- Posterior predictive:
  p(a | e) = [c·1(a = a*) + α F(a|d)] / (c + α)
- Mode switch: the mode becomes the truth when c > α [F(m|d) - F(a*|d)], where m is the prior mode.
- Shape of the threshold: it is linear in a *probability-mass gap* and saturates at α F(m|d). In
  log c against s, the iso-accuracy curve is concave and flattens.

**(ii) Multiplicative (log-linear) learner.**
- Setup: logits z_a = log F(a|d) + β log(1 + c)·1(a = a*).
- Correctness: logit P(correct) ≈ β log c - s_nat + log[1/(1 - F(a*))].
- Iso-accuracy line: **straight and unbounded**, log c* = (s - s_mode)/β.
- Price per bit: each extra bit of surprise multiplies the required exposure by 2^(1/β).

**Discriminating statistic: the prior-persistence slope λ(c).** Within exposure bins, regress
logit p(a*|e) on log F(a*|d), with two controls: answer-string length, and the unconditional value
log-probability.
- Log-linear learner: λ(c) ≈ 1 at every c.
- Additive learner: λ(c) → 0 once c grows past α.

The nesting model p ∝ F^λ(c) exp(entity evidence) fits both. The log-linearity of the knowledge
overshadowing law (Zhang et al. 2502.16143) is what the multiplicative learner predicts. So this
test also explains why that law takes its form, and where it should break.

## Observational law (claim C2)

Mixed-effects logistic regression:

logit P(correct_e) = a + b log c_e - g s_e + h log n_e + (1|slot) + (1|cell) + (1|model)

- Price per bit = g/b.
- Iso-accuracy line at P = 0.5: log2 c* = (g/b)·s + const.
- Fit s_m and s_w separately.
- Pre-registered prediction: g > 0 with a 95% CI excluding 0, and s adds held-out predictive
  power beyond log c and log n.
- G0 runs a cheap version: cross-validated logistic AUROC with and without s, grouped by cell.

## Category adjustment for numeric slots (claim C5; Huttenlocher, Hedges and Duncan 1991)

ŷ_e - μ_d = w(c_e)·(y_e - μ_d) + ε

- ŷ_e: expected year over scored candidate years (avoids digit-sampling artefacts; see Benford's Curse, 2506.01734).
- μ_d: the ghost-prior mean year.
- Prediction: w(c) < 1 at low exposure, rising towards 1 with c.
- Validity: the regressor y_e is exact, so this is not a regression-to-the-mean artefact. Use all entities, not only wrong ones.

## Knowledge share (E4)

Decompose logit p(a*|e) = prior term(d, a*) + entity term(e). Then

κ = Var(entity) / [Var(prior) + Var(entity)]

Report κ per slot, model and training stage, with bootstrap CIs.

## Surprise-weighted singleton floor (claim C6; an honest corollary, not a big theorem)

Kalai et al. (2509.04664; Nature 2026) lower-bound hallucination on arbitrary facts by the singleton
rate sr (Good–Turing missing mass). If answers on effectively unseen entities can depend only on d,
and the world draws answers from W(·|d), then for a calibrated non-abstaining sampler:

- error on an unseen entity = 1 - Σ_a q(a|d) W(a|d), which is 1 - Σ_a W(a|d)² when q = W;
- floor: Hall ≥ sr · E_d[1 - Σ_a W(a|d)²] - (estimation terms);
- for greedy decoding: Hall ≥ sr · E_d[1 - max_a W(a|d)].

*Proof sketch:*
1. Condition on the missing-mass event.
2. By exchangeability of unseen entities within a cell, the answer distribution depends on d alone.
3. Expected accuracy is the inner product of q and W, which is maximised by the mode and equals the collision probability under calibration.

*Checks:*
- With uniform W over K values, the floor reduces to sr·(1 - 1/K).
- It sits below sr whenever W is predictable ("typical facts are free").
- It needs descriptors not to leak the truth beyond W, so leaky cells are excluded.
- Under greedy decoding it also predicts C1 (errors on unseen facts equal the prior mode).

## Familiarity gate (exploration X3)

Model: answer when familiarity φ(n_e) > τ, where familiarity is driven by name exposure n, not fact exposure c.

Predictions:
- conditional on c, the direct-question answer rate rises with log n;
- name-only exposure (in context, or LoRA "becoming famous overnight", after Jacoby et al. 1989) raises answering and hallucination without raising accuracy.

## Math-washing audit

**Keep:** the two learners with λ(c), the iso-accuracy law, w(c), κ and the floor. Each maps to a fitted number with a CI and a kill criterion.

**Cut:** any field, quanta, manifold or curvature language, and any theorem whose conclusion is not tested.
