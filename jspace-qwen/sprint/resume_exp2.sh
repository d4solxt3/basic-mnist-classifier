#!/bin/bash
# Resume Experiment 2 after the container restart (2026-11-16 ~22:15 UTC): tuning and the confirmation
# stage were complete and saved; the leave-one-task-out stage had saved layers 4 and 7. Each stage skips
# what is already saved, so this re-runs only LOTO (from layer 11), the verdict and the conditional M4.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
echo "=== $(date -u +%H:%M:%S) RESUMED after container restart"
run LOTO python sprint/exp2.py --stage loto
run VERDICT python sprint/exp2.py --stage verdict
if python -c "import json,sys; c=json.load(open('sprint/results/exp2_verdict.json'))['criteria']; sys.exit(0 if c['C1_loto'] and c['C2_gap_halved'] and c['C3_no_loss'] and c['C4_fp'] else 1)"; then
  python -c "import torch; a=torch.load('sprint/results/maps_exp2_seed100_B0.pt'); b=torch.load('sprint/results/maps_exp2_seed100_B2.pt'); a.update(b); torch.save(a, 'sprint/results/maps_exp2_seed100.pt')"
  run M4_EARLY python sprint/swap_check.py --maps sprint/results/maps_exp2_seed100.pt --layers 4 7 --tag exp2_early
  run M4_EARLYMID python sprint/swap_check.py --maps sprint/results/maps_exp2_seed100.pt --layers 4 7 11 14 --tag exp2_earlymid
else
  echo "=== M4 skipped: confirm criteria 1-4 not all met (pre-registered condition)"
fi
echo "=== EXP2 DONE"
