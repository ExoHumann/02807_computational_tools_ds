#!/bin/bash
#BSUB -J afp_setup
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 06:00
#BSUB -o hpc/logs/%J_setup.out
#BSUB -e hpc/logs/%J_setup.err

# One-off setup: download FMA small and the MuQ weights so later GPU jobs start straight away.
# The zips go to the node's local disk (home quota is too small for them); the processed audio
# (about 11.5 GB) goes to data/processed in the repo and MuQ (about 1.3 GB) to ~/.cache/huggingface.
# Safe to resubmit: the zip is downloaded again but finished tracks are skipped.
set -euo pipefail

REPO=$HOME/02807_computational_tools_ds
LOCAL=/tmp/${USER}_afp_setup
trap 'rm -rf "$LOCAL"' EXIT
mkdir -p "$LOCAL/data" "$REPO/data/processed"
ln -sfn "$REPO/data/processed" "$LOCAL/data/processed"
cd "$REPO"

echo "== $(date) host $(hostname)"; df -h /tmp; du -sh "$HOME"

echo "== $(date) data"
time uv run --no-sync python -m afp.data --subset small --data-dir "$LOCAL/data" --workers "$LSB_DJOB_NUMPROC"

echo "== $(date) MuQ weights"
time uv run --no-sync python -c "from afp.embed import MuQBackbone; MuQBackbone('cpu'); print('MuQ ok')"

echo "== $(date) done"
wc -l data/processed/manifest.csv data/processed/skipped.csv
du -sh data/processed "$HOME/.cache/huggingface" "$HOME"
