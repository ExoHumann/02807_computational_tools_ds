#!/bin/bash
#BSUB -J afp_gpucheck
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 00:15
#BSUB -o hpc/logs/%J_gpucheck.out
#BSUB -e hpc/logs/%J_gpucheck.err
cd ~/02807_computational_tools_ds
nvidia-smi
echo "TMPDIR=$TMPDIR"; df -h /tmp "$TMPDIR" /scratch 2>/dev/null
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
