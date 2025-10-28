#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <start-month> <end-month>"
    echo "e.g.: $0 'January 2025' 'February 2025'"
    exit 1
fi

START_MONTH="$1"
END_MONTH="$2"

export PYTHONPATH="$(pwd)/.."

get_cc_indices() {
    uv run python -c "
from tgrag.utils.data_loading import interval_to_CC_slices
indices = interval_to_CC_slices(\"$1\", \"$2\")
print(' '.join(indices))
" "$1" "$2"
}

CRAWL_INDICES=$(get_cc_indices "$START_MONTH" "$END_MONTH")

echo "Resolved CC indices: $CRAWL_INDICES"

# Set up env & paths, using scratch if possible (running on a cluster is indeed recommended)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_URL=https://data.commoncrawl.org

if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH/crawl-data"
fi


process_crawl() {
    local CRAWL=$1
    echo "[INFO] Starting crawl: $CRAWL"

    mkdir -p "$DATA_DIR/$CRAWL/output"
    rm -rf "$DATA_DIR/$CRAWL/output"/*

    wget -q "$BASE_URL/crawl-data/$CRAWL/wat.paths.gz" -O /tmp/wat.paths.gz
    TOTAL_FILES=$(gzip -dc /tmp/wat.paths.gz | wc -l)
    echo "[INFO] Total WAT files: $TOTAL_FILES"

    BATCH_SIZE=2
    START_IDX=0

    for (( start_idx=$START_IDX; start_idx<=TOTAL_FILES; start_idx+=BATCH_SIZE )); do
        end_idx=$((start_idx+BATCH_SIZE-1))
        if [ "$end_idx" -gt "$TOTAL_FILES" ]; then
            end_idx=$TOTAL_FILES
        fi

        echo "[INFO] Start processing batch $start_idx-$end_idx for $CRAWL @ $(date '+%Y-%m-%d %H:%M:%S')"
        rm -rf "$DATA_DIR/$CRAWL/segments/"*

        bash "$SCRIPT_DIR/end-to-end.sh" "$CRAWL" "$start_idx" "$end_idx"
    done

    echo "[INFO] Starting aggregation for $CRAWL @ $(date '+%Y-%m-%d %H:%M:%S')"
    uv run python ../tgrag/construct_graph_scripts/construct_aggregate.py \
        --source "$DATA_DIR/$CRAWL/output_text_dir" \
        --target "$DATA_DIR/$CRAWL/output"
    echo "[INFO] Finished aggregation for $CRAWL @ $(date '+%Y-%m-%d %H:%M:%S')"
}

# Launch each crawl in parallel
for CRAWL in $CRAWL_INDICES; do
    process_crawl "$CRAWL" > "logs/${CRAWL}.log" 2>&1 &
done

wait
echo "All crawls completed."
