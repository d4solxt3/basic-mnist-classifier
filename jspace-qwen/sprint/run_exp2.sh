#!/bin/bash
# Experiment 2 as pre-registered (sprint/prereg_exp2.md), started only after Experiment 1's queue has finished.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
until grep -q "=== EXP1 DONE" sprint/logs/exp1.log; do sleep 60; done
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
run TUNE python sprint/exp2.py --stage tune
run CONFIRM python sprint/exp2.py --stage confirm
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
