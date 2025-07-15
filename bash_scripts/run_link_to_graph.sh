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
# For cluster use
if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
    SPARK_WAREHOUSE="spark-warehouse"
else
    DATA_DIR="$SCRATCH"
    SPARK_WAREHOUSE="$SCRATCH/spark-warehouse"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Set PySpark to use the virtualenv's Python
export PYSPARK_PYTHON="$VENV_PATH/bin/python"
export PYSPARK_DRIVER_PYTHON="$VENV_PATH/bin/python"

#Set output for text
OUTPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/output_text_dir"

# Clean previous outputs if they exist
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Cleaning up:"
rm -rf "$SPARK_WAREHOUSE/host_graph_output_vertices"
rm -rf "$SPARK_WAREHOUSE/host_graph_output_edges"

"$VENV_PATH"/bin/spark-submit \
  --driver-memory 2g \
  --executor-memory 2g \
  --conf spark.sql.shuffle.partitions=4 \
  --conf spark.io.compression.codec=snappy \
  --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py,$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py,$PROJECT_ROOT/tgrag/cc-scripts/json_importer.py" \
  "$PROJECT_ROOT/tgrag/cc-scripts/hostlinks_to_graph.py" \
  "$SPARK_WAREHOUSE/$outputTableName" \
  host_graph_output \
  --output_format "parquet" \
  --output_compression "snappy" \
  --log_level "WARN" \
  --save_as_text "$OUTPUT_DIR" \
  --vertex_partitions 2
