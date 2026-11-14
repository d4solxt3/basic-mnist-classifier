#!/bin/bash
# Exploratory follow-up to the swap experiments: equal-width (5-layer) windows across depth, to
# separate depth from the number of edited layers. Flexible generalisation only, J-lens and logit
# lens, scales 1 and 2. The 20-24 window is the main swap run (results/swaps_main.json).
# Run single-threaded at low priority alongside round 3.
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for w in "3 7" "7 11" "11 15" "15 19"; do
  set -- $w
  echo "=== $(date -u +%H:%M:%S) START window $1-$2"
  nice -n 10 python scripts/swaps.py --jlens lenses/exact_final.pt --band $1 $2 --scales 1 2 --exps flexgen --tag win$1_$2
  echo "=== $(date -u +%H:%M:%S) END rc=$?"
done
echo "=== WINDOWS DONE"
