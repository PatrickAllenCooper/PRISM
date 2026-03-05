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
#SBATCH --mem=64G

# -----------------------------------------------------------------------------
# PRISM Stage 1-3 Experiment Runner for CURC Alpine resources.
# Allocates one A100 GPU, 8 CPU cores, and 64GB RAM.
# Loads SmolLM2-135M-Instruct (~270MB) twice (policy + reference).
# -----------------------------------------------------------------------------

set -e

module purge
module load anaconda
module load cuda/12.1

conda activate prism_env

export HF_HOME=/scratch/alpine/paco0228/hf_cache
export TRANSFORMERS_CACHE=/scratch/alpine/paco0228/hf_cache
export PYTHONPATH=$(pwd)

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME: $HF_HOME"

mkdir -p results

echo "==================================="
echo "Starting PRISM Phase 1: QA"
echo "==================================="
python -m prism.stage1_qa

echo "==================================="
echo "Starting PRISM Phase 2a: Database"
echo "==================================="
python -m prism.stage2_db

echo "==================================="
echo "Starting PRISM Phase 2b: API"
echo "==================================="
python -m prism.stage2_api

echo "==================================="
echo "Starting PRISM Phase 3: Web"
echo "==================================="
python -m prism.stage3_web

echo "==================================="
echo "All PRISM experiments complete."
echo "Results saved to results/ directory."
echo "==================================="

ls -la results/
