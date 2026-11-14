#!/bin/bash
# Once queue 3b finishes the ignition run (i.e. starts S1), stop it and hand over to queue 3c,
# which runs round 3 of the J++ comparison first and then the sketch fits.
cd "$(dirname "$0")/.."
until grep -q "START S1" logs/queue3b.log; do sleep 5; done
pkill -f run_queue3b.sh; sleep 1; pkill -f "fit.py --method sketch"; sleep 2
echo "=== $(date -u +%T) handoff to queue3c (S1 restarts there)" >> logs/queue3b.log
setsid nohup scripts/run_queue3c.sh > logs/queue3c.log 2>&1 < /dev/null &
