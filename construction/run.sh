#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/March-pipeline-%j.out
#SBATCH --error=logs/March-pipeline-%j.err
#SBATCH --cpus-per-task=8                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1   
#SBATCH --mem=256G
#SBATCH --time=168:00:00                       # The job will run for 1 day
#SBATCH --job-name=March-pipeline

export HOME="/home/mila/k/kondrupe"
export PATH="$HOME/bin:$PATH"
module load python/3.10
source ~/CGfullgraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

exit_script() {
    echo "Preemption signal, saving myself"
    trap - SIGTERM # clear the trap
    # Optional: sends SIGTERM to child/sub processes
    kill -- -$$
}

trap exit_script SIGTERM

bash pipeline.sh 'March 2025' 'March 2025'

# if memory: try lock on aggregate or batch
