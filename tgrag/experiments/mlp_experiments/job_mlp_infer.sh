#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/mlp_infer_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/mlp_infer_job_%j.err

# Exit on error
set -e

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
uv run python  mlp_inference.py