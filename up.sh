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
# (VARIANT=base, the old architecture); MODES / WIDTH (defaults 20 / 40) add
# the size tag, e.g. Work/models/train_t<END>_lap_m24/; SEED (default 0) adds
# s<N> for a model trained with that training seed (train_t<END>_lap_w32s1/).
#
#   sbatch --export=ALL,END=1000 up.sh          predict to PAIRS[1000] = 10000
#   sbatch --export=ALL,END=300,VARIANT=base up.sh   existing base model
#   sbatch --export=ALL,END=500,MODES=24 up.sh       the 25.1M model
#   sbatch --export=ALL,END=1000,WIDTH=32,SEED=1 up.sh   a seed replicate
#   t=$(sbatch --parsable --export=ALL,END=1000 ut.sh)
#   g=$(sbatch --parsable gt.sh)
#   sbatch --dependency=afterok:$t:$g --export=ALL,END=1000 up.sh
#
# Tuning + 50-seed rollouts to 10000 take ~2-3 h once the ground truth exists
# (gt.sh).  Resumable: seeds already in Work/rollout_cache/<label>/ for the
# same checkpoint and schedule are skipped.  The run's own Results/ folder is
# drawn by uno_predict.py; the final make_submission.py --overview call only
# refreshes Results/README.md and the comparison images. For a full redraw of
# every run folder: sbatch sp.sh

set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}
T_END=${T_END:-10000}
MODES=${MODES:-20}
WIDTH=${WIDTH:-40}
SEED=${SEED:-0}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= PREDICTION (END=$END, VARIANT=$VARIANT, MODES=$MODES, WIDTH=$WIDTH, SEED=$SEED, T_END=$T_END) ======================="
python uno_predict.py --end "$END" --variant "$VARIANT" --modes "$MODES" --width "$WIDTH" --seed "$SEED" --t-end "$T_END"

echo
echo "======================= RESULTS ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"
