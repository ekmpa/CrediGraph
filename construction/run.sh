#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/run-pipeline-%j.out
#SBATCH --error=logs/run-pipeline-%j.err
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=128G
#SBATCH --time=168:00:00                       # The job will run for 1 day
#SBATCH --job-name=run-pipeline

export HOME="/home/mila/k/kondrupe"
module load python/3.10
source ~/CGfullgraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

# bash pipeline.sh CC-Crawls/CC-2024-nov.txt --keep 45300 #15900 --exclusive

bash pipeline.sh 'January 2020' 'February 2020' 2
