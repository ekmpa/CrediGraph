#!/bin/bash
set -e

# Check crawl list file argument
if [ -z "$1" ]; then
    echo "Usage: $0 <crawl-list.txt> [--keep]"
    exit 1
fi

CRAWL_LIST_FILE="$1"

if [ ! -f "$CRAWL_LIST_FILE" ]; then
    echo "File not found: $CRAWL_LIST_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_URL=https://data.commoncrawl.org

if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH/crawl-data"
fi

while read -r CRAWL || [[ -n "$CRAWL" ]]; do

    if [ "$2" != "--keep" ]; then
        rm -rf "$DATA_DIR/$CRAWL/output"
    fi
    mkdir -p "$DATA_DIR/$CRAWL/output"

    wget -q "$BASE_URL/crawl-data/$CRAWL/wat.paths.gz" -O /tmp/wat.paths.gz
    TOTAL_FILES=$(gzip -dc /tmp/wat.paths.gz | wc -l)
    echo "Total WAT files: $TOTAL_FILES"

    STEP=300

    for (( start_idx=1; start_idx<=TOTAL_FILES; start_idx+=STEP )); do
        end_idx=$((start_idx+STEP-1))

        if [ "$end_idx" -gt "$TOTAL_FILES" ]; then # Clamp to TOTAL_FILES if end_idx exceeds it
            end_idx=$TOTAL_FILES
        fi

        echo "Running slice: $start_idx-$end_idx"

        bash end-to-end.sh "$CRAWL" $start_idx $end_idx

        python ../tgrag/utils/aggregate.py --source "$DATA_DIR/$CRAWL/output_text_dir" --target "$DATA_DIR/$CRAWL/output"

        for f in edges vertices; do
            target_file="$DATA_DIR/$CRAWL/output/${f}.txt.gz"
            if [ -f "$target_file" ]; then
                num_lines=$(gzip -dc "$target_file" | wc -l)
                echo "[INFO] After slice $start_idx-$end_idx: $f has $num_lines records"
            fi
        done
    done
done < "$CRAWL_LIST_FILE"
