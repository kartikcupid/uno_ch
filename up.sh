#!/bin/bash
#SBATCH --job-name=uno_predict
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=up_%j.out
#SBATCH --error=up_%j.err

# Predict and draw with an already trained model: Work/models/train_t<END>_lap/
# (VARIANT=lap, the default, Laplacian output) or Work/models/train_t<END>/
# (VARIANT=base, the old architecture).
#
#   sbatch --export=ALL,END=1000 up.sh          predict to PAIRS[1000] = 10000
#   sbatch --export=ALL,END=300,VARIANT=base up.sh   existing base model
#   t=$(sbatch --parsable --export=ALL,END=1000 ut.sh)
#   g=$(sbatch --parsable gt.sh)
#   sbatch --dependency=afterok:$t:$g --export=ALL,END=1000 up.sh
#
# Tuning + 50-seed rollouts to 10000 take ~2-3 h once the ground truth exists
# (gt.sh).  Resumable: seeds already in Work/rollout_cache/<label>/ for the
# same checkpoint and schedule are skipped.  The final make_submission.py call
# redraws every complete run and Results/README.md + comparison.png.

set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}
T_END=${T_END:-10000}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= PREDICTION (END=$END, VARIANT=$VARIANT, T_END=$T_END) ======================="
python uno_predict.py --end "$END" --variant "$VARIANT" --t-end "$T_END"

echo
echo "======================= RESULTS ======================="
python make_submission.py

echo
echo "Job finished at $(date)"
