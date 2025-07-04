#!/bin/bash
set -e

# Check if CRAWL argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <CRAWL-ID>"
    echo "Example: $0 CC-MAIN-2017-13"
    exit 1
fi

CRAWL="$1"

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster use
if [ -z "$SCRATCH" ]; then
    echo "[WARN] SCRATCH not set, using local data directory."
    DATA_DIR="$PROJECT_ROOT/data"
    SPARK_WAREHOUSE="spark-warehouse"
else
    DATA_DIR="$SCRATCH"
    SPARK_WAREHOUSE="$SCRATCH/CrediGraph/spark-warehouse"
    echo "Using SCRATCH directory: $DATA_DIR"
fi

# Base URL used to download the path listings
BASE_URL=https://data.commoncrawl.org


mkdir -p "$DATA_DIR/"
INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"
mkdir -p "$INPUT_DIR"

if [ -e "$INPUT_DIR/test.txt" ]; then
    echo "File $INPUT_DIR/test.txt already exists"
    echo "... delete it to write a new one"
    exit 1
fi

for data_type in warc wat wet; do

    echo "Downloading Common Crawl paths listings (${data_type} files of $CRAWL)..."

    mkdir -p "$DATA_DIR/crawl-data/$CRAWL/"
    listing="$DATA_DIR/crawl-data/$CRAWL/$data_type.paths.gz"
    cd "$DATA_DIR/crawl-data/$CRAWL/"
    wget --timestamping "$BASE_URL/crawl-data/$CRAWL/$data_type.paths.gz"
    cd -

    echo "Downloading sample ${data_type} file..."

    file=$(gzip -dc "$listing" | head -1)
    full_path="$DATA_DIR/$file"
    mkdir -p "$(dirname "$full_path")"
    cd "$(dirname "$full_path")"
    wget --timestamping "$BASE_URL/$file"
    cd -

    echo "Writing input file listings..."

    input="$INPUT_DIR/test_${data_type}.txt"
    echo "Test file: $input"
    echo "file:$full_path" >>"$input"

    input="$INPUT_DIR/all_${data_type}_${CRAWL}.txt"
    echo "All ${data_type} files of ${CRAWL}: $input"
    gzip -dc "$listing" >"$input"

done
