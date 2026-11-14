#!/bin/bash
# Run round 2 of the J++ comparison right after JCA, before the swap experiments,
# then resume queue3 from SW (scripts/run_queue3b.sh).
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
until grep -q "START SW" logs/queue3.log; do sleep 5; done
pkill -f run_queue3.sh; sleep 1; pkill -f "swaps.py"; sleep 2
echo "=== $(date -u +%T) START R2" >> logs/queue3.log
python scripts/jpp_splits.py --seeds 0 1 2 3 4 5 --bags 8 --tag round2 > logs/round2.log 2>&1
echo "=== $(date -u +%T) END R2 rc=$?" >> logs/queue3.log
setsid nohup scripts/run_queue3b.sh > logs/queue3b.log 2>&1 < /dev/null &
echo "=== handoff to queue3b done" >> logs/queue3.log
