#!/bin/bash
#SBATCH --job-name=uno_predict
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=o_%j.out
#SBATCH --error=o_%j.err

set -euo pipefail

module load python
source env/bin/activate

python make_submission.py
