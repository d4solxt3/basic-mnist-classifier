#!/bin/bash
# Handoff after the LRP fit: stop queue 2 as it starts JC, start queue 3.
set -u
cd "$(dirname "$0")/.."
until grep -q "START JC" logs/queue2.log; do sleep 5; done
pkill -f run_queue2.sh; sleep 1; pkill -f "jpp_compare.py --plain exact --lrp lrp --tag main"; sleep 2
echo "=== $(date -u +%H:%M:%S) start queue3"
setsid nohup scripts/run_queue3.sh > logs/queue3.log 2>&1 < /dev/null &
echo "=== handoff done"
