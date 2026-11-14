#!/bin/bash
# Third compute queue (replaces queue 2 after the LRP fit): broadcast lens first,
# so the J++ head-to-head includes it, then the remaining experiments.
set -u
cd "$(dirname "$0")/.."
source /home/user/venv/bin/activate
run() { echo "=== $(date -u +%H:%M:%S) START $1"; shift; "$@"; echo "=== $(date -u +%H:%M:%S) END rc=$?"; }
# Broadcast lens: same 8 corpus prompts as the reference lens's n=8 snapshot, targets p' > p only.
run BL  python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --tag later
EXTRA="--extra J-lens-n8=lenses/exact_n8.pt B-lens=lenses/later_final.pt --mix JB-mix=lenses/exact_final.pt,lenses/later_final.pt"
run JC  python scripts/jpp_compare.py --plain exact --lrp lrp $EXTRA --tag main
run JCA python scripts/jpp_compare.py --plain exact --lrp lrp $EXTRA --all-items --tag main_all
BAND=$(python -c "import json;b=json.load(open('results/band.json'))['band'];print(b[0],b[1])")
run SW  python scripts/swaps.py --jlens lenses/exact_final.pt --regress --band $BAND --tag main
run SWM python scripts/swaps.py --jlens lenses/exact_final.pt --band 7 20 --scales 1 2 4 --tag midband
run IG  python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 10 --controls --tag main
run S1 python scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
run S2 python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
run S3 python scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
run S4 python scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
run CU python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none
echo "=== QUEUE3 DONE"
