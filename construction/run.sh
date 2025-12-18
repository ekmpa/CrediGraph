#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/Feb-pipeline-%j.out
#SBATCH --error=logs/Feb-pipeline-%j.err
#SBATCH --cpus-per-task=8                     # Ask for 4 CPUs
#SBATCH --mem=256G
#SBATCH --time=168:00:00    
#SBATCH --job-name=Feb                  # The job will run for 1 day

if [ $# -lt 1 ]; then 
    echo "Usage: $0 <start-month> [<end-month>]"
    echo "e.g.: $0 'January 2025' 'March 2025'."
    exit 1
elif [ $# -eq 1 ]; then
    START_MONTH="$1"
    END_MONTH="$1"
elif [ $# -eq 2 ]; then 
    START_MONTH="$1"
    END_MONTH="$2"
fi

# removed #SBATCH --gres=gpu:rtx8000:1   

export PATH="$HOME/bin:$PATH"
module load python/3.10
source ~/CrediGraph/.venv/bin/activate
export JAVA_HOME="$HOME/java/jdk-17"

exit_script() {
    echo "Preemption signal, saving myself"
    start_short=$(echo "$1" | cut -c1-3)
    end_short=$(echo "${2:-$1}" | cut -c1-3)
    job_name="${start_short}-${end_short}"
    sbatch --job-name="$job_name" run.sh "$START_MONTH" "$END_MONTH" 
    trap - SIGTERM 
    kill -- -$$
}

trap exit_script SIGTERM

bash pipeline.sh "$START_MONTH" "$END_MONTH" 8
bash process.sh "$START_MONTH" "$END_MONTH"

# if memory: try lock on aggregate or batch
