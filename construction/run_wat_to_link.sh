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
    outputTableName="wat_output_table"
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
# Local testing: use "$INPUT_DIR/test_wat.txt"
# Cluster / full usage: ""$INPUT_DIR/all_wat_$CRAWL.txt"
"$VENV_PATH/bin/spark-submit" \
  --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py" \
  "$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py" \
  "$INPUT_DIR/test_wat.txt" \
  "$outputTableName" \
  --input_base_url https://data.commoncrawl.org/



#  ../../data/crawl-data/CC-MAIN-2025-21/input/test_wat.txt   wat_output_table   --input_base_url https://data.commoncrawl.org/
