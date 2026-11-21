#!/bin/bash
#BSUB -J afp_tiktok
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 02:00
#BSUB -o hpc/logs/%J_tiktok.out
#BSUB -e hpc/logs/%J_tiktok.err

# Held-out TikTok-like queries for FMA small: speech chunks, MuQ features of the queries, then encode.
# Needs hpc/run_small.sh to have finished (references and projection head). Resumable.
set -euo pipefail

cd "$HOME/02807_computational_tools_ds"
echo "== $(date) host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

if [ ! -f data/processed/speech.npy ]; then
    echo "== $(date) speech"
    time uv run --no-sync python -m afp.data --speech
fi

echo "== $(date) cache-tiktok"
time uv run --no-sync python -m afp.fingerprint --data-dir data cache-tiktok --queries 2000 --batch-size 16

echo "== $(date) encode"
time uv run --no-sync python -m afp.fingerprint --data-dir data --model models/projection_head.pt encode
echo "== $(date) done"
