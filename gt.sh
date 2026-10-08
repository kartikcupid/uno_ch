#!/bin/bash
#SBATCH --job-name=uno_gt
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=gt_%j.out
#SBATCH --error=gt_%j.err

# Ground truth only: every file a prediction to T_END needs, into Work/gt_cache/
# (test + tune seeds for all five quenches at l = 128, all box-size seeds at
# l = 160/192/256), solved in float64 on the GPU.  No model is loaded, so it
# can run while training runs.  Every model is predicted to t = 10000 and
# every training run validates on 10 seeds at l = 128 to 10000, so one job
# with the default T_END writes every file that training and prediction need:
# chain them with afterok on this job, or they solve the files again.
#
#   sbatch gt.sh                                T_END = 10000
#   sbatch --export=ALL,T_END=20000 gt.sh
#   g=$(sbatch --parsable gt.sh)
#   sbatch --dependency=afterok:$g --export=ALL,END=1000 utp.sh
#
# A larger T_END later reuses nothing from a shorter one, but a shorter run
# reuses the first frames of a longer one.  Files already in the cache are
# skipped, so a job that hits the time limit can simply be resubmitted;
# re-point jobs chained on the old id with scontrol update job=<id>
# dependency=afterok:<new id> (an afterok on a timed-out job never starts).
# Log: Work/predict/gt_t<T_END>/predict.log.

set -euo pipefail

T_END=${T_END:-10000}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= GROUND TRUTH (t_end=$T_END) ======================="
python uno_predict.py --gt-only --t-end "$T_END"

echo
echo "Job finished at $(date)"
