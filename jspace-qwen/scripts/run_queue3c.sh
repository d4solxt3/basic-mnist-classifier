#!/bin/bash
# Queue 3c: round 3 of the J++ comparison (paper/prereg_round3.md), then the sketch-estimator tail of queue 3b.

set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
run LB python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --lrp --tag lrp_later
run R3 python scripts/jpp_splits.py --seeds $(seq 6 25) --bags 8 --candidates Bagged-J++ RB-mix RLB-mix \
    --lrp-b lenses/lrp_later_final.pt --oop --tag round3
run S1 python scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
run S2 python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
run S3 python scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
run S4 python scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
run CU python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none
echo "=== QUEUE3C DONE"
