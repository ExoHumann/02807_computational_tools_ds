#!/bin/bash
#BSUB -J afp_encode
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 01:00
#BSUB -o hpc/logs/%J_encode.out
#BSUB -e hpc/logs/%J_encode.err

# CPU only: add RMS levels to clean shards cached before they were stored, then encode heads v1 and v2.
set -euo pipefail

cd "$HOME/02807_computational_tools_ds"
echo "== $(date) rms"
uv run --no-sync python -c "
import pandas as pd
from afp.fingerprint import cache_clean
cache_clean(pd.read_csv('data/processed/manifest.csv'), 'data/processed', backbone=None)
"
echo "== $(date) encode v1"
uv run --no-sync python -m afp.fingerprint --data-dir data --model models/projection_head_v1.pt encode \
    --device cpu --out-dir fingerprints_v1
echo "== $(date) encode v2"
uv run --no-sync python -m afp.fingerprint --data-dir data --model models/projection_head.pt encode --device cpu
echo "== $(date) done"
