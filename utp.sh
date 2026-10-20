#!/bin/bash
#SBATCH --job-name=uno_ch
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=utp_%j.out
#SBATCH --error=utp_%j.err

# Train, predict to 10000 and draw one model.
#
#   sbatch --export=ALL,END=1000 utp.sh                 Laplacian output (lap)
#   sbatch --export=ALL,END=1000,VARIANT=base utp.sh    the old architecture
#   sbatch --export=ALL,END=500,MODES=24 utp.sh         25.1M parameters
#   sbatch --export=ALL,END=1000,WIDTH=32,SEED=1 utp.sh a seed replicate
#   g=$(sbatch --parsable gt.sh)
#   sbatch --dependency=afterok:$g --export=ALL,END=1000 utp.sh
#
# Training uses the survival-based checkpoint selection; the prediction label
# is train_t<END>_lap_pred_t10000 (VARIANT=base: train_t<END>_pred_t10000;
# MODES=24: train_t<END>_lap_m24_pred_t10000, side by side with the default;
# SEED=1: train_t<END>_lap_s1_pred_t10000, the training seed adds s<N>),
# and a same-label cache built from another checkpoint is archived
# automatically (README section 6).
#
# The run's own Results/ folder is drawn by uno_predict.py; the last step only
# refreshes Results/README.md and the comparison images (make_submission.py
# --overview). For a full redraw of every run folder: sbatch sp.sh

set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}
MODES=${MODES:-20}
WIDTH=${WIDTH:-40}
SEED=${SEED:-0}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= TRAINING ======================="
python uno_train.py --end "$END" --variant "$VARIANT" --modes "$MODES" --width "$WIDTH" --seed "$SEED"
echo
echo "======================= PREDICTION ======================="
python uno_predict.py --end "$END" --variant "$VARIANT" --modes "$MODES" --width "$WIDTH" --seed "$SEED"

echo
echo "======================= SUBMISSION ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"