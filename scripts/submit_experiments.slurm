#!/bin/bash
#SBATCH --job-name=prism_experiments
#SBATCH --output=prism_%j.out
#SBATCH --error=prism_%j.err
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00

# -----------------------------------------------------------------------------
# PRISM Stage 1-3 Experiment Runner for CURC Alpine resources.
# Allocates one A100 GPU and 8 CPU cores.
# -----------------------------------------------------------------------------

# Purge existing modules and load required compiler/MPI/CUDA environments
module purge
module load anaconda
module load cuda/12.1

# Activate the conda environment (assuming one is already created)
# conda activate prism_env

# Echo system info
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to exactly where the project sits
export PYTHONPATH=$(pwd)

echo "==================================="
echo "Starting PRISM Phase 1: QA"
echo "==================================="
python -m prism.stage1_qa

echo "==================================="
echo "Starting PRISM Phase 2: Synthetic"
echo "==================================="
python -m prism.stage2_db
python -m prism.stage2_api

echo "==================================="
echo "Starting PRISM Phase 3: Web"
echo "==================================="
python -m prism.stage3_web

echo "==================================="
echo "All PRISM experiments complete."
echo "==================================="
