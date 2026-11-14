#!/bin/bash
# Handoff after job A: stop old queue, run contention-free evals, start queue 2.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
until grep -q "START S1" logs/queue.log; do sleep 10; done
pkill -f run_queue.sh; sleep 1; pkill -f "fit.py --method sketch"; sleep 2
echo "=== $(date -u +%H:%M:%S) readout_eval J-lens final"
python scripts/readout_eval.py --lens J-lens=lenses/exact_final.pt --regress --filter --tag main > logs/readout_main.log 2>&1
python scripts/choose_band.py results/readout_main.json "J-lens|raw" >> logs/after_A.log 2>&1
echo "=== $(date -u +%H:%M:%S) jpp_compare plain"
python scripts/jpp_compare.py --plain exact --lrp none --tag plain > logs/jpp_plain.log 2>&1
echo "=== $(date -u +%H:%M:%S) start queue2"
setsid nohup scripts/run_queue2.sh > logs/queue2.log 2>&1 < /dev/null &
echo "=== handoff done"
