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

CRAWL_ARG="$1"

if [ -f "$CRAWL_ARG" ]; then
  echo "Crawl list file detected: $CRAWL_ARG"
  CRAWLS=$(cat "$CRAWL_ARG")
else
  echo "Single crawl ID detected: $CRAWL_ARG"
  CRAWLS="$CRAWL_ARG"
fi

# Get the root of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

while read -r CRAWL || [[ -n "$CRAWL" ]]; do
    # Skip empty lines or comments
    [[ -z "$CRAWL" || "$CRAWL" =~ ^# ]] && continue

    echo "Processing $CRAWL..."
    echo "Removing previous $CRAWL spark-warehouse"

    rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse" # Remove re-created directories before running

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

done <<< "$CRAWLS"
