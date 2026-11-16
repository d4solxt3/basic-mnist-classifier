#!/bin/bash
# Experiment 1 as pre-registered (sprint/PLAN.md): tuning -> confirmation -> LOTO -> verdict -> M4 if C1-C3 pass.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
run TUNE python sprint/exp1.py --stage tune
run CONFIRM python sprint/exp1.py --stage confirm
run LOTO python sprint/exp1.py --stage loto
run VERDICT python sprint/exp1.py --stage verdict
if python -c "import json,sys; c=json.load(open('sprint/results/exp1_verdict.json'))['criteria']; sys.exit(0 if c['C1_mean_ge_3pt'] and c['C2_wins_ge_16'] and c['C3_no_fp_increase'] else 1)"; then
  run M4_EARLY python sprint/swap_check.py --maps sprint/results/maps_exp1_seed100.pt --layers 4 7 --tag exp1_early
  run M4_EARLYMID python sprint/swap_check.py --maps sprint/results/maps_exp1_seed100.pt --layers 4 7 11 14 --tag exp1_earlymid
else
  echo "=== M4 skipped: C1-C3 not all met (pre-registered condition)"
fi
echo "=== EXP1 DONE"
