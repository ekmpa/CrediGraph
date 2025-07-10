#!/bin/bash
set -e

# Check if crawl list file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <crawl-list.txt>"
    exit 1
fi

if [ -z "$2" ]; then
      start_idx=1
else
      start_idx=$2
fi

if [ -z "$2" ]; then
      end_idx=10
else
      end_idx=$3
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

    # Use SCRATCH if defined, else fallback to project-local data dir
    # For cluster usage
    if [ -z "$SCRATCH" ]; then
        rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse" # Remove re-created directories before running
    else
        rm -rf "$SCRATCH/spark-warehouse" # Remove re-created directories before running
    fi
    echo $CRAWL
    echo "################################### run get data ###################################"
    ./get_data.sh "$CRAWL" $start_idx $end_idx
    echo "Data Downloaded for $CRAWL."
    echo "###################################  run_wat_to_link ###################################"
    ./run_wat_to_link.sh "$CRAWL"
    echo "wat_output_table constructed for $CRAWL."
    echo "###################################  run_link_to_graph ###################################"
    ./run_link_to_graph.sh "$CRAWL"
    echo "Compressed graphs constructed for $CRAWL."
    echo "********************** End Of the Task **********************"

done < "$CRAWL_LIST_FILE"
