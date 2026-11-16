#!/bin/bash
# E0.2: baseline metrics after the search-noise run, then the M4 swap baseline on seed-0 J++ maps.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
while kill -0 1163 2>/dev/null; do sleep 20; done
echo "=== $(date -u +%H:%M:%S) START baseline"
python sprint/baseline.py --seeds 0 1 2 3 4 5 --tag tuning
echo "=== $(date -u +%H:%M:%S) START swaps"
python sprint/swap_check.py --maps sprint/results/maps_baseline_tuning_seed0.pt --layers 4 7 --tag base_early
python sprint/swap_check.py --maps sprint/results/maps_baseline_tuning_seed0.pt --layers 4 7 11 14 --tag base_earlymid
echo "=== $(date -u +%H:%M:%S) BASELINE DONE"
