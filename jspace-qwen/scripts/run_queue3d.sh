#!/bin/bash
# Queue 3d (post hoc, after the online control variate diverged): shaped identity-CV sketch (running),
# shrunk online control variate, learning curves for both, and the offline theory/Monte Carlo check.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
while kill -0 26206 2>/dev/null; do sleep 20; done
echo "=== sk16idw finished"
run S6 python scripts/fit.py --method sketch --K 16 --cv shrink --n 256 --snapshots 16 32 64 128 256 --tag sk16sh --seed 1
run CU python scripts/curves.py --tags sk16idw sk16sh
run TH python scripts/sketch_theory.py --seeds 8 --steps 512 --threads 4
echo "=== QUEUE3D DONE"
