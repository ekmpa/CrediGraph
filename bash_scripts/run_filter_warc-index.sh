
#!/bin/bash

# Fail on first error
set -e

# Check if CRAWL argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CRAWL-ID>"
    echo "Example: $0 CC-MAIN-2017-13"
    exit 1
fi
CRAWL="$1"

if [ -z "$2" ]; then
    outputTableName="wat_extract_content_table"
else
  outputTableName="$2"
fi

if [ -z "$3" ]; then
      seed_list='../data/dqr/domain_pc1.csv'
else
      seed_list=$3
fi

if [ -z "$4" ]; then
      start_idx=1
else
      start_idx=$4
fi

if [ -z "$5" ]; then
      end_idx=10
else
      end_idx=$5
fi
echo "outputTableName=$outputTableName"
echo "seed_list=$seed_list"

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster usage
if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi

INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Set PySpark to use the virtualenv's Python
export PYSPARK_PYTHON="$VENV_PATH/bin/python"
export PYSPARK_DRIVER_PYTHON="$VENV_PATH/bin/python"


# Run the Spark job
# Local testing: use "$INPUT_DIR/test_wat.txt"
# XX:+UseG1GC -> explicitly tells the JVM to use the Garbage-First (G1) Garbage Collector
echo "INPUT_DIR=$INPUT_DIR"
"$VENV_PATH/bin/spark-submit" \
    --driver-memory 15g \
    --executor-memory 10g \
    --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py" \
    "$PROJECT_ROOT/tgrag/cc-scripts/filter_cc_index.py" \
    "$INPUT_DIR/${CRAWL}_test_cc-index-table_${start_idx}_${end_idx}.txt" \
    "$outputTableName" \
    --trusted_domains "$seed_list" \
    --output_format "parquet" \
    --output_compression "snappy" \
    --log_level "WARN"

    ################### delete processed batch files ###############
    while IFS= read -r line; do
        file_Path="${line##*:}"
        echo "delete file path=$file_Path"
        rm -f "$file_Path"
    done <  "$INPUT_DIR/${CRAWL}_test_cc-index-table_${start_idx}_${end_idx}.txt"