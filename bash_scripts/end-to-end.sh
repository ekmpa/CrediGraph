#!/bin/bash
set -e

# Check if crawl list file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <crawl-list.txt>"
    exit 1
fi

CRAWL_LIST_FILE="$1"

if [ ! -f "$CRAWL_LIST_FILE" ]; then
    echo "File not found: $CRAWL_LIST_FILE"
    exit 1
fi

# Get the root of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"

while read -r CRAWL || [[ -n "$CRAWL" ]]; do
    # Skip empty lines or comments
    [[ -z "$CRAWL" || "$CRAWL" =~ ^# ]] && continue

    echo "Processing $CRAWL..."
    echo "Removing previous $CRAWL spark-warehouse"

    rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse" # Remove re-created directories before running

    ./get_data.sh "$CRAWL"
    echo "Data Downloaded for $CRAWL."
    echo "###################################  run_wat_to_link ###################################"
    ./run_wat_to_link.sh "$CRAWL"
    echo "wat_output_table constructed for $CRAWL."
    echo "###################################  run_link_to_graph ###################################"
    ./run_link_to_graph.sh "$CRAWL"
    echo "Compressed graphs constructed for $CRAWL."
    echo "********************** End Of the Task **********************"

done < "$CRAWL_LIST_FILE"
