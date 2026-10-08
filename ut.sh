#!/bin/bash
#SBATCH --job-name=uno_train
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=ut_%j.out
#SBATCH --error=ut_%j.err

# Train one model with the survival-based checkpoint selection.
#
#   sbatch --export=ALL,END=1000 ut.sh                  Laplacian output (lap)
#   sbatch --export=ALL,END=1000,VARIANT=base ut.sh     the old architecture
#
# VARIANT=lap (default) writes Work/models/train_t<END>_lap/; VARIANT=base
# writes Work/models/train_t<END>/.
set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= TRAINING ======================="
python uno_train.py --end "$END" --variant "$VARIANT"

echo
echo "Job finished at $(date)"
