#!/bin/bash
#BSUB -J afp_voice
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 03:00
#BSUB -o hpc/logs/%J_voice.out
#BSUB -e hpc/logs/%J_voice.err

# Head v2: retrain with voice-over pairs (LibriSpeech train-clean-100 speakers) and compare with head v1 on the same
# queries. The TikTok-like queries are regenerated so they record the voice-over level. Resumable.
# Needs hpc/run_small.sh to have finished.
set -euo pipefail

cd "$HOME/02807_computational_tools_ds"
P=data/processed
afp() { uv run --no-sync python -m "$@"; }
echo "== $(date) host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

[ -f models/projection_head_v1.pt ] || cp models/projection_head.pt models/projection_head_v1.pt
if [ ! -d $P/embeddings/pairs_v1 ] && [ -d $P/embeddings/pairs ]; then mv $P/embeddings/pairs $P/embeddings/pairs_v1; fi
first=$(ls $P/embeddings/queries_tiktok/shard_0000.npz 2>/dev/null || true)
if [ -n "$first" ] && ! uv run --no-sync python -c "import numpy, sys; sys.exit('voice_over_db' not in numpy.load('$first'))"; then
    rm -rf $P/embeddings/queries_tiktok
fi

for split in train test; do
    if [ ! -f $P/speech_$split.npy ]; then
        echo "== $(date) speech $split"
        time afp afp.data --speech $split
    fi
done

echo "== $(date) pairs with voice-over"
time afp afp.fingerprint --data-dir data cache --voice-over --pairs 10000 --queries 2000 --batch-size 16

echo "== $(date) cache-tiktok"
time afp afp.fingerprint --data-dir data cache-tiktok --queries 2000 --batch-size 16

echo "== $(date) train v2"
time afp afp.fingerprint --data-dir data --model models/projection_head.pt train

echo "== $(date) encode v1"
afp afp.fingerprint --data-dir data --model models/projection_head_v1.pt encode --out-dir fingerprints_v1
echo "== $(date) encode v2"
afp afp.fingerprint --data-dir data --model models/projection_head.pt encode
echo "== $(date) done"
