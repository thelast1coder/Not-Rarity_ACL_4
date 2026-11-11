# SNR: Surprise, Not Rarity (ACL 2027)

Language models hallucinate their priors: wrong answers are the type prior's default, and the exposure
needed to beat the prior grows with the truth's surprise at a measurable price per bit.

- Start here: `PROJECT.md`, then `CLAUDE.md` (agent manual).
- Theory: `docs/02_theory_and_definitions.md`. Experiments: `docs/03_experiments.md`. Gates: `docs/04_phases_and_gates.md`.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q tests                      # CPU, under a minute
bash scripts/run_g0.sh laptop                  # Wikidata + ghosts + exact counts (network, no GPU)
SMOKE=1 bash scripts/run_g0.sh gpu             # small real-model smoke (DEVICE=cpu works)
bash scripts/run_g0.sh gpu                     # G0 model pass, about 2 GPU-hours
bash scripts/run_g0.sh analyse                 # -> results/gates/G0.md
```
