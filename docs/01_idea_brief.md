# One-page brief: Surprise, Not Rarity (shareable with collaborators)

**Problem.** We can predict how often language models get facts wrong, but not what they say when wrong,
nor why a heavily mentioned fact can fail while a once-mentioned one succeeds. Sampling-consistency
detectors miss errors the model makes confidently and consistently.

**Insight.**
- When a model lacks a fact, it answers with what is typical for that kind of person: the type prior.
- Rare facts that match the prior come for free.
- Surprising facts need many training exposures, and when they fail, they fail to the prior's default.

**Method.**
1. Measure the prior directly with "ghosts": made-up names of the same origin, with zero occurrences in the
   OLMo-2 training corpus.
2. Join model answers to exact pretraining counts (infini-gram).
3. Fit a law: logit P(correct) = a + b log c - g·surprise. The ratio g/b is the **price per bit** of surprise.
4. Discriminate two learning mechanisms (additive count learner vs log-linear learner) with one statistic, the
   prior-persistence slope λ(c).
5. Reproduce the price causally by injecting synthetic facts at controlled surprise and exposure.
6. Use the prior to catch consistent hallucinations and recover surprising truths.

**Why it matters.**
- It says *what* a model will hallucinate before it does.
- It replaces "rarity" with a better axis.
- It gives the first causal law of hallucination.
- It documents a fairness harm: people whose lives are atypical for their names are systematically misdescribed.

**Cost.** About 24.5 GPU-hours on one 80 GB GPU. G0 (the go/no-go test) costs about 2.

**Status.** G0 code is ready and tested on CPU; data building runs on a laptop; G0 runs as soon as a GPU is available.
