#!/bin/bash
# Long compute queue (sequential: all jobs share the 4 CPU cores).
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
PY=python
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
run A  $PY scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt \
          --snapshots 1 2 4 8 16 32 --per-prompt-layers 7 14 21 --tag exact
run S1 $PY scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
run S2 $PY scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
run B  $PY scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt --lrp \
          --snapshots 8 16 32 --tag lrp
run S3 $PY scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
run S4 $PY scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
echo "=== QUEUE DONE"
