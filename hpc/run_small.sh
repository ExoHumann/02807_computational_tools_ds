#!/bin/bash
#BSUB -J afp_small
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o hpc/logs/%J_small.out
#BSUB -e hpc/logs/%J_small.err

# Fingerprints for FMA small: timed trial, then cache, train and encode.
# Needs hpc/setup_small.sh to have run (processed audio in data/processed, MuQ in ~/.cache/huggingface).
# Resumable: resubmitting skips finished shards.
set -euo pipefail

REPO=$HOME/02807_computational_tools_ds
TRIAL=/tmp/${USER}_afp_trial_${LSB_JOBID}
trap 'rm -rf "$TRIAL"' EXIT
cd "$REPO"
afp() { uv run --no-sync python -m afp.fingerprint "$@"; }

echo "== $(date) host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

if [ ! -d data/processed/embeddings/clean ]; then
    echo "== $(date) timed trial"
    mkdir -p "$TRIAL/processed"
    ln -s "$REPO/data/processed/audio" "$TRIAL/processed/audio"
    cp data/processed/manifest.csv "$TRIAL/processed/"
    start=$SECONDS
    afp --data-dir "$TRIAL" cache --limit-tracks 200 --pairs 500 --queries 100 --batch-size 16
    elapsed=$((SECONDS - start))
    uv run --no-sync python - "$TRIAL/processed/embeddings" "$elapsed" <<'EOF'
import sys
from pathlib import Path
import numpy as np
root, elapsed = Path(sys.argv[1]), float(sys.argv[2])
keys = ("embeddings", "clean", "distorted")
n = sum(len(v) for f in root.glob("*/shard_*.npz") for k, v in np.load(f).items() if k in keys)
print(f"trial: {n} windows in {elapsed:.0f} s (includes model load) = {elapsed / n:.3f} s/window")
EOF
fi

echo "== $(date) cache"
time afp --data-dir data cache --pairs 10000 --queries 2000 --batch-size 16

echo "== $(date) train"
time afp --data-dir data --model models/projection_head.pt train

echo "== $(date) encode"
time afp --data-dir data --model models/projection_head.pt encode
echo "== $(date) done"
