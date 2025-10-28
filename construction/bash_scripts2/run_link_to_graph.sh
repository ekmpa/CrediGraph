#!/bin/bash

# Fail on first error
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <CRAWL-ID> <subfolder_id>"
  echo "Example: $0 CC-MAIN-2025-21 3"
  exit 1
fi

CRAWL="$1"
subfolder_id="$2"
subfolder_name="bash_scripts${subfolder_id}"

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCTION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster use
if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi

INPUT_DIR_SUFFIX="input${subfolder_id}"
SPARK_WAREHOUSE_PATH="$CONSTRUCTION_DIR/$subfolder_name/spark-warehouse"
OUTPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/output_text_dir_${subfolder_id}"

# virtual environment set up
source "$VENV_PATH/bin/activate"
export PYSPARK_PYTHON="$VENV_PATH/bin/python"
export PYSPARK_DRIVER_PYTHON="$VENV_PATH/bin/python"

# Clean previous outputs if they exist
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Cleaning up:"
rm -rf "$SPARK_WAREHOUSE_PATH/host_graph_output_vertices"
rm -rf "$SPARK_WAREHOUSE_PATH/host_graph_output_edges"

"$VENV_PATH/bin/spark-submit" \
  --driver-memory 64g \
  --executor-memory 4g \
  --conf spark.sql.shuffle.partitions=512 \
  --conf spark.io.compression.codec=snappy \
  --conf spark.default.parallelism=512 \
  --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py,$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py,$PROJECT_ROOT/tgrag/cc-scripts/json_importer.py" \
  "$PROJECT_ROOT/tgrag/cc-scripts/hostlinks_to_graph.py" \
  "$SPARK_WAREHOUSE_PATH/wat_output_table" \
  host_graph_output \
  --output_format "parquet" \
  --output_compression "snappy" \
  --log_level "WARN" \
  --save_as_text "$OUTPUT_DIR" \
  --vertex_partitions 2
