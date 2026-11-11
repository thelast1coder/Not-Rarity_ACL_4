#!/usr/bin/env bash
# Gate G0 for SNR. Run from the repository root. Three stages:
#   bash scripts/run_g0.sh laptop        # no GPU: tests, Wikidata build, ghosts, exact counts
#   SMOKE=1 bash scripts/run_g0.sh gpu   # real-model smoke on 30 items (CPU allowed: DEVICE=cpu)
#   bash scripts/run_g0.sh gpu           # G0 model pass: OLMo-2-7B instruct and base (about 2 GPU-hours)
#   bash scripts/run_g0.sh analyse       # gate decision -> results/gates/G0.md
set -euo pipefail
export PYTHONPATH=src
STAGE=${1:-laptop}
DATA=${DATA:-data/g0}
DEVICE=${DEVICE:-cuda}

case "$STAGE" in
  laptop)
    python -m snr.check_env
    python -m pytest -q tests
    [ -f "$DATA/items.jsonl" ] || python -m snr.build_entities --config configs/g0.yaml --out "$DATA"
    echo "Read $DATA/build_report.md. Fix QIDs or thresholds if a slot is thin, then re-run."
    [ -f "$DATA/ghosts.json" ] || python -m snr.ghosts --data "$DATA" --config configs/g0.yaml
    python -m snr.counts --data "$DATA" --config configs/g0.yaml
    ;;
  gpu)
    python -m snr.check_env
    python -m pytest -q tests
    if [ "${SMOKE:-0}" = "1" ]; then
      python -m snr.run_g0 --model Qwen/Qwen2.5-0.5B-Instruct --chat --data "$DATA" --out results/smoke \
        --limit-items 30 --max-ghosts 4 --device "$DEVICE" --batch-size 8
      python -m snr.analysis --data "$DATA" --results results/smoke --gate-out results/smoke_gate
      echo "Smoke: any verdict is fine. What must hold: the run completes and boundary_ok rate is near 1."
      exit 0
    fi
    python -m snr.run_g0 --model allenai/OLMo-2-1124-7B-Instruct --chat --data "$DATA" --out results/G0 --device "$DEVICE"
    python -m snr.run_g0 --model allenai/OLMo-2-1124-7B --no-chat --data "$DATA" --out results/G0 --device "$DEVICE"
    ;;
  analyse)
    python -m snr.analysis --data "$DATA" --results results/G0 --config configs/g0.yaml
    ;;
  *)
    echo "usage: bash scripts/run_g0.sh [laptop|gpu|analyse]"; exit 2 ;;
esac
