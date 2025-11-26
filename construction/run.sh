#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/may8-pipeline-%j.out
#SBATCH --error=logs/may8-pipeline-%j.err
#SBATCH --cpus-per-task=8                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=128G
#SBATCH --time=168:00:00                       # The job will run for 1 day
#SBATCH --job-name=may8-pipeline

export HOME="/home/mila/k/kondrupe"
module load python/3.10
source ~/CGfullgraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

bash pipeline.sh 'May 2025' 'May 2025' 8
