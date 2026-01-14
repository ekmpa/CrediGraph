#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mila/a/abdallah/scratch/jobs_log/cc-content-ext/cc-content-ext_job_%j.out
#SBATCH --error=/home/mila/a/abdallah/scratch/jobs_log/cc-content-ext/cc-content-ext_job_%j.err

# Exit on error
set -e


if [ -z "$1" ]; then
      CRAWL=ccmain202508
else
      CRAWL=$1
fi
CRAWL=${CRAWL,,}
wetFilesOrder="credigraph_${CRAWL}_wetFilesOrder.txt"
if [ -z "$2" ]; then
      Month=Feb2025
else
      Month=$2
fi

if [ -z "$3" ]; then
      sidx=0
else
      sidx=$3
fi

if [ -z "$4" ]; then
      eidx=10
else
      eidx=$4
fi

if [ -z "$5" ]; then
      batch_size=10
else
      batch_size=$5
fi
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "CRAWL:     $CRAWL"
echo "Month:     $Month"
echo "sidx:     $sidx"
echo "eidx:     $eidx"
echo "batch_size:     $batch_size"
export JAVA_HOME=~/jdk-17.0.12/
export PATH=$PATH:$JAVA_HOME/bin

# Execute Python script
# Use `uv run --offline` on clusters without internet access on compute nodes.
for ((i=$sidx; i<$eidx; i+=$batch_size)); do
    echo "#########################################################################################"
    echo ./end-to-end.sh  CC-Crawls/${Month}.txt $i $((i+$batch_size-1)) [wet] ../data/${Month}/${Month}_domains_${sidx}_${eidx}.csv content_ext_table spark-warehouse/$wetFilesOrder
    ./end-to-end.sh  CC-Crawls/${Month}.txt $i $((i+$batch_size-1)) [wet] ../data/${Month}/${Month}_domains_${sidx}_${eidx}.csv content_ext_table spark-warehouse/$wetFilesOrder
    # rm -r  ~/scratch/crawl-data/CC-MAIN-2025-08/segments
done