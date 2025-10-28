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

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCTION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
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


# Run the Spark job
# Local testing: use "$INPUT_DIR/test_wat.txt"
# Cluster / full usage: ""$INPUT_DIR/all_wat_$CRAWL.txt"
"$VENV_PATH/bin/spark-submit" \
  --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py" \
  "$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py" \
  "$INPUT_DIR/test_wat.txt" \
  "$output_table" \
  --input_base_url https://data.commoncrawl.org/



#  ../../data/crawl-data/CC-MAIN-2025-21/input/test_wat.txt   wat_output_table   --input_base_url https://data.commoncrawl.org/
