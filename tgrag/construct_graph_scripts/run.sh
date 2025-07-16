#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/nov-main-%j.out
#SBATCH --error=logs/nov-main-%j.err
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=256G
#SBATCH --time=80:00:00                       # The job will run for 1 day
#SBATCH --job-name=nov-main

export HOME="/home/mila/k/kondrupe"
module load python/3.10
source ~/CrediGraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

#bash end-to-end.sh CC-Crawls/one-crawl.txt
python main.py --slices CC-MAIN-2024-nov
