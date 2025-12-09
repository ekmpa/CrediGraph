#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/January-pipeline-%j.out
#SBATCH --error=logs/January-pipeline-%j.err
#SBATCH --cpus-per-task=8                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=256G
#SBATCH --time=168:00:00                       # The job will run for 1 day
#SBATCH --job-name=January-pipeline

export HOME="/home/mila/k/kondrupe"
export PATH="$HOME/bin:$PATH"
module load python/3.10
source ~/CGfullgraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

bash pipeline.sh 'January 2025' 'January 2025'

# if memory: try lock on aggregate or batch
