#!/bin/bash
# set -e
set -euo pipefail

# TODOS
# clean the logging
# and use logger?

if [ $# -lt 2 ]; then # TODO: make the Jan/Feb default values? So users could launch without an interval to get an example 2-month graph?
    echo "Usage: $0 <start-month> <end-month> [num-subfolders]"
    echo "e.g.: $0 'January 2025' 'February 2025' 9"
    exit 1
fi

START_MONTH="$1"
END_MONTH="$2"
NUM_SUBFOLDERS="${3:-9}"  # default to 9 (latest crawls have 90K files -> 10K each)

# Set up env & paths, using scratch if possible (running on a cluster is indeed recommended)
export PYTHONPATH="$(pwd)/.."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONSTRUCTION_DIR="$PROJECT_ROOT/construction"
TEMPLATE_SCRIPTS="$CONSTRUCTION_DIR/bash_scripts"
BASE_URL=https://data.commoncrawl.org
mkdir -p "$CONSTRUCTION_DIR/logs"


get_cc_indices() {
    uv run python -c "
from tgrag.utils.data_loading import interval_to_CC_slices
indices = interval_to_CC_slices(\"$1\", \"$2\")
print(' '.join(indices))
" #"$1" "$2"
}

CRAWL_INDICES=$(get_cc_indices "$START_MONTH" "$END_MONTH")

echo "Running CrediBench on CC slices: $CRAWL_INDICES"

if [ -z "$SCRATCH" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi


process_crawl() {
    local CRAWL=$1
    echo "[INFO] Starting crawl: $CRAWL"
    mkdir -p "$DATA_DIR/crawl-data/$CRAWL"

    mkdir -p "$DATA_DIR/crawl-data/$CRAWL/output"
    rm -rf "$DATA_DIR/crawl-data/$CRAWL/output"/*

    wget -q "$BASE_URL/crawl-data/$CRAWL/wat.paths.gz" -O /tmp/wat.paths.gz
    TOTAL_FILES=$(gzip -dc /tmp/wat.paths.gz | wc -l)
    echo "[INFO] Total WAT files: $TOTAL_FILES"

    FILES_PER_SUBSET=$(( (TOTAL_FILES + NUM_SUBFOLDERS - 1) / NUM_SUBFOLDERS ))
    echo "[INFO] Splitting into $NUM_SUBFOLDERS subsets, ~$FILES_PER_SUBSET files each"

    for (( i=0; i<NUM_SUBFOLDERS; i++ )); do
        (
            SUBFOLDER_ID=$((i + 1))
            SUBFOLDER_NAME="bash_scripts$SUBFOLDER_ID"
            OUTPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/output$SUBFOLDER_ID"
            SEGMENT_DIR="$DATA_DIR/crawl-data/$CRAWL/segments$SUBFOLDER_ID"
            TARGET_SCRIPTS="$CONSTRUCTION_DIR/$SUBFOLDER_NAME"
            if [ ! -d "$TARGET_SCRIPTS" ]; then
                echo "[INFO] Creating $TARGET_SCRIPTS from template"
                cp -r "$TEMPLATE_SCRIPTS" "$TARGET_SCRIPTS"
            fi
            mkdir -p "$OUTPUT_DIR" "$SEGMENT_DIR"

            START_IDX=$((i * FILES_PER_SUBSET))
            END_IDX=$(((i + 1) * FILES_PER_SUBSET - 1))
            if [ "$END_IDX" -ge "$TOTAL_FILES" ]; then END_IDX=$((TOTAL_FILES - 1)); fi

            echo "[INFO][$CRAWL][Subfolder $SUBFOLDER_ID] Processing $START_IDX to $END_IDX"

            BATCH_SIZE=300
            for (( batch_start=START_IDX; batch_start<=END_IDX; batch_start+=BATCH_SIZE )); do
                batch_end=$((batch_start + BATCH_SIZE - 1))
                if [ "$batch_end" -gt "$END_IDX" ]; then batch_end=$END_IDX; fi

                echo "[INFO][$CRAWL][Subfolder $SUBFOLDER_ID] Batch: $batch_start-$batch_end"
                rm -rf "$SEGMENT_DIR"/*
                bash "$TARGET_SCRIPTS/end-to-end.sh" "$CRAWL" "$batch_start" "$batch_end" "$SUBFOLDER_ID"
            done
        ) > "$CONSTRUCTION_DIR/logs/${CRAWL}_sub$((i+1)).log" 2>&1 &
    done

    wait
    echo "[INFO][$CRAWL] All subsets done. Starting merge..."

    # Merge outputs pairwise
    local merged_output="$DATA_DIR/$CRAWL/output1"
    for (( i=2; i<=NUM_SUBFOLDERS; i++ )); do
        local src="$DATA_DIR/$CRAWL/output$i"
        echo "[INFO][$CRAWL] Merging $src into $merged_output"
        uv run python tgrag/construct_graph_scripts/merge_ext.py \
            --source "$src" \
            --target "$merged_output"
    done
    echo "[INFO][$CRAWL] Final merged output: $merged_output"
}

# Launch each crawl in parallel
for CRAWL in $CRAWL_INDICES; do
    process_crawl "$CRAWL" > "$CONSTRUCTION_DIR/logs/${CRAWL}.log" 2>&1 &
done

wait

# TODO : remove each subfolder here.
echo "All crawls completed."
