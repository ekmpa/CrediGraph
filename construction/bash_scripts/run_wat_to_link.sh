#!/bin/bash

# Fail on first error
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <CRAWL-ID> <subfolder_id> [output_table_name]"
  echo "Example: $0 CC-MAIN-2025-21 3 wat_output_table"
  exit 1
fi

CRAWL="$1"
subfolder_id="$2"
output_table="${3:-wat_output_table}"  # Hussien: do we need to be able to change this?

# get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBFOLDER_PATH="$SCRIPT_DIR"
CRAWL_FOLDER="$(dirname "$SUBFOLDER_PATH")"
CONSTRUCTION_DIR="$(dirname "$CRAWL_FOLDER")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
SPARK_WAREHOUSE="$SUBFOLDER_PATH/spark-warehouse"
INPUT_DIR_SUFFIX="input${subfolder_id}"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster usage
if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi

INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/$INPUT_DIR_SUFFIX"
# TODO: error handling on the input file?

# virtual environment set up
source "$VENV_PATH/bin/activate"
export PYSPARK_PYTHON="$VENV_PATH/bin/python"
export PYSPARK_DRIVER_PYTHON="$VENV_PATH/bin/python"

MAX_RETRIES=10
RETRY_DELAY=5

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[INFO] Starting Spark job (attempt $attempt)..."

    # randomize UI port to reduce clashes
    export SPARK_UI_PORT=$(( 4040 + RANDOM % 2000 ))

    set +e

    "$VENV_PATH/bin/spark-submit" \
        --conf "spark.ui.port=$SPARK_UI_PORT" \
        --conf "spark.port.maxRetries=40" \
        --master local[1] \
        --conf "spark.sql.warehouse.dir=$SPARK_WAREHOUSE" \
        --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py" \
        "$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py" \
        "$INPUT_DIR/test_wat.txt" \
        "$output_table" \
        --input_base_url https://data.commoncrawl.org/
    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "[INFO] Spark job succeeded on attempt $attempt"
        break
    fi

    echo "[WARN] Spark job failed (exit $exit_code); retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
done

if [ $exit_code -ne 0 ]; then
    echo "[ERROR] Spark job failed after $MAX_RETRIES attempts"
    exit 1
fi
