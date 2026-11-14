#!/bin/bash
# Queue 3b: the rest of queue 3 (from the swap experiments), resumed after round 2 of the J++ comparison.

set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
BAND=$(python -c "import json;b=json.load(open('results/band.json'))['band'];print(b[0],b[1])")
run SW  python scripts/swaps.py --jlens lenses/exact_final.pt --regress --band $BAND --tag main
run SWM python scripts/swaps.py --jlens lenses/exact_final.pt --band 7 20 --scales 1 2 4 --tag midband
run IG  python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 10 --controls --tag main
run S1 python scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
run S2 python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
run S3 python scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
run S4 python scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
run CU python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none
echo "=== QUEUE3B DONE"
