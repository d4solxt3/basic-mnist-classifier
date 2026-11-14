#!/bin/bash
# Second compute queue (started after job A): LRP fit first (needed for J++), then sketches.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
run B  python scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt --lrp \
          --snapshots 8 16 32 --tag lrp
BAND=$(python -c "import json;b=json.load(open('results/band.json'))['band'];print(b[0],b[1])")
run SW  python scripts/swaps.py --jlens lenses/exact_final.pt --regress --band $BAND --tag main
run SWM python scripts/swaps.py --jlens lenses/exact_final.pt --band 7 20 --scales 1 2 4 --tag midband
run IG  python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 10 --controls --tag main
run JC  python scripts/jpp_compare.py --plain exact --lrp lrp --tag main
run JCA python scripts/jpp_compare.py --plain exact --lrp lrp --all-items --tag main_all
run S1 python scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
run S2 python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
run S3 python scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
run S4 python scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
run CU python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none
echo "=== QUEUE2 DONE"
