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

"$VENV_PATH/bin/spark-submit" \
  --driver-memory 10g \
  --executor-memory 5g \
  --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py" \
  "$PROJECT_ROOT/tgrag/cc-scripts/wet_extract_domain_urls.py" \
  "$INPUT_DIR/test_wet.txt" \
  "$outputTableName" \
  --trusted_domains "../data/cc-label+deg_3.txt" \
  --output_format "parquet" \
  --output_compression "snappy" \
  --log_level "WARN"
