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

# get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBFOLDER_PATH="$SCRIPT_DIR"
CRAWL_FOLDER="$(dirname "$SUBFOLDER_PATH")"
CONSTRUCTION_DIR="$(dirname "$CRAWL_FOLDER")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
SPARK_WAREHOUSE_PATH="$SUBFOLDER_PATH/spark-warehouse"
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Spark warehouse path: $SPARK_WAREHOUSE_PATH"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster use
if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi

OUTPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/output_text_dir${subfolder_id}"

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

MAX_RETRIES=10
RETRY_DELAY=5

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[INFO] Starting Spark job (attempt $attempt)..."

    # randomize UI port to reduce clashes
    export SPARK_UI_PORT=$((4040 + RANDOM % 2000))

    set +e
    "$VENV_PATH/bin/spark-submit" \
        --master local[1] \
        --driver-memory 64g \
        --executor-memory 4g \
        --conf spark.sql.shuffle.partitions=512 \
        --conf spark.io.compression.codec=snappy \
        --conf spark.default.parallelism=512 \
        --conf spark.sql.warehouse.dir="$SPARK_WAREHOUSE_PATH" \
        --py-files "$PROJECT_ROOT/tgrag/cc-scripts/sparkcc.py,$PROJECT_ROOT/tgrag/cc-scripts/wat_extract_links.py,$PROJECT_ROOT/tgrag/cc-scripts/json_importer.py" \
        "$PROJECT_ROOT/tgrag/cc-scripts/hostlinks_to_graph.py" \
        "$SPARK_WAREHOUSE_PATH/wat_output_table" \
        host_graph_output \
        --output_format "parquet" \
        --output_compression "snappy" \
        --log_level "WARN" \
        --save_as_text "$OUTPUT_DIR" \
        --vertex_partitions 2

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


#  ../../data/crawl-data/CC-MAIN-2025-21/input/test_wat.txt   wat_output_table   --input_base_url https://data.commoncrawl.org/
