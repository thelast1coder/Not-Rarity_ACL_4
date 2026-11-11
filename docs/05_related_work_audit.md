# Adversarial related-work audit (as of 2026-11-11)

This is a lead list. Verify every reference on arXiv or the ACL Anthology before it enters the paper.
Re-sweep weekly. Search terms: "type prior", "surprise", "typicality", "presupposition hallucination",
"default answer", "category adjustment", "knowledge overshadowing".

| # | Work | What it did | What stays open for us | Risk |
|---|---|---|---|---|
| 1 | Lu et al., Relational Linearity, 2601.11429 | SyntHal synthetic subjects; linearity predicts hallucinate vs refuse (r .58-.84); plausible fillers | measured descriptor prior; error = its mode; counts and surprise | **High** for C1 |
| 2 | Zucchet et al., 2503.21676 | synthetic: plateau at population statistics; hallucinations emerge with knowledge | real models, real slots, price per bit | Medium |
| 3 | Zhang et al., Knowledge Overshadowing, 2502.16143 (ACL 2025 Findings) | log-linear law in relative popularity; CoDA decoding | prior from ghosts and Wikidata; exact counts; learner discrimination; causal surprise | **Medium-high** for C2 |
| 4 | Predictable Confabulations, 2605.18732 | sigmoid in log size and topic frequency (R² ≈ 0.6) | surprise term; the wrong value | Low-medium |
| 5 | Kalai et al., 2509.04664; Nature 2026 | singleton floor; evaluation incentives | typicality-weighted floor tested with exact counts | Medium (corollary) |
| 6 | Guo and Li, 2602.00906 (ICML 2026) | rate-distortion theorem for membership testing | different mechanism; cite | Low |
| 7 | WikiMem 2507.11128; LMP2 audit 2602.17483 | generic-subject baseline subtraction for privacy audits | descriptor-matched ghosts; hallucination detection | **Medium** for C7 |
| 8 | When Bias Pretends to Be Truth, 2511.07318 | spurious correlations evade detectors (synthetic) | a fix, on real models | Medium (motivation) |
| 9 | Ferrando et al., ICLR 2025, 2411.14257 | entity-recognition directions steer refusal/hallucination | behavioural n vs c gate test | Low-medium |
| 10 | PhantomBench, 2606.11105 | existence 16.2% vs meaning 33.4% hallucination | content of fabrications; real partially known entities | **High** for C8 |
| 11 | Cheng, Hawkins, Jurafsky, 2601.04435 | at-issue 0.74 vs not-at-issue 0.43 correction | model's own content; factual slots | **High** for C8 |
| 12 | DecoPrompt, 2411.07457 | direct-rejected fictitious claims hallucinated when presupposed | same | High for C8 |
| 13 | Vaquet, Pütz, Paassen, Findings of EMNLP 2026 | presupposition shifts the classifier operating point | behavioural, content-level | Medium |
| 14 | SLAQ, 2510.11218 | short vs long-form factual (mis)alignment | abstention; priors | Low |
| 15 | HalluHard, 2602.01031 | short-form elicits more abstention; self-conditioning across turns | prior content; counts | Low-medium |
| 16 | Kim et al., Correlated Errors, ICML 2025, 2506.07962 | about 60% agreement when both wrong on HELM | shared priors as the cause | Low |
| 17 | Large Legal Fictions, 2401.01301 | bias towards recognisable justices | generalisation and law | Low |
| 18 | Misra et al., typicality in LMs, 2105.02987 | weak typicality effects | link to errors | Low |
| 19 | LLM memory effects, 2509.17138 | DRM lures recalled at 0.114 | factual hallucination; category adjustment | Low |
| 20 | Chang et al., NeurIPS 2024 | fictional knowledge injection into OLMo pretraining | surprise as the manipulated variable | Medium (method) |
| 21 | Benford's Curse, 2506.01734 | digit bias in numeric hallucination | control in E3 | Low |
| 22 | 2607.00447; Two Axes of Abstention 2607.08456; PhantomFill 2607.20492 | priors override constraints; abstention axes; schema-forced fabrication | cite; not our law | Low-medium |
| 23 | Empty Shelves or Lost Keys? 2602.14080 | encoded vs recalled facts | no typicality | Low |
| 24 | When the Wrong Key Wins, 2609.15106 | keyword-entity competition | does not predict the wrong value | Low |
| 25 | Kaplan, Gekhman et al., COLM 2026 | fine-tuning forgetting via key overlap | different object | Low |

## Verdict

**C1 is a replication plus measurement.** The idea is anticipated; our quantitative form is new.

**These are new as far as this audit can establish:**
- the C2 surprise law;
- C3, the λ(c) learner discrimination;
- C4, the causal price per bit;
- C5, numeric category adjustment;
- C6, the floor.

**C8 is mostly taken.** Keep only the prior-content and self-presupposition angles.

Not re-verified in the last pass: Vaquet et al., PhantomFill, Two Axes of Abstention, 2607.16643, 2607.13568, Facts in Stats.
