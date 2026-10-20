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
#   sbatch --export=ALL,END=500,MODES=24 ut.sh          25.1M parameters
#   sbatch --export=ALL,END=1000,WIDTH=32,SEED=1 ut.sh  a seed replicate
#
# VARIANT=lap (default) writes Work/models/train_t<END>_lap/; VARIANT=base
# writes Work/models/train_t<END>/.  MODES (default 20) and WIDTH (default 40)
# set the model size; a non-default value adds a tag to the folder, e.g.
# Work/models/train_t0500_lap_m24/ (17.9M -> 25.1M parameters at MODES=24).
# SEED (default 0) is the training seed (weight init, pair draws, noise); a
# non-zero value adds s<N>, e.g. Work/models/train_t1000_lap_w32s1/, next to
# the seed-0 run, to measure run-to-run noise.
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
echo "Job finished at $(date)"
