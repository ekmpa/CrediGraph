#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <start-month> <end-month> [num-subfolders]"
    echo "e.g.: $0 'January 2025' 'February 2025' 9"
    exit 1
fi

# TODO:
# comments
# fallback for fetch error ?

START_MONTH="$1"
END_MONTH="$2"
NUM_SUBFOLDERS="${3:-9}"

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
"
}

CRAWL_INDICES=$(get_cc_indices "$START_MONTH" "$END_MONTH")

if [ -z "${SCRATCH:-}" ]; then
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

    WAT_PATHS_FILE=$(mktemp /tmp/wat.paths.XXXXXX.gz)
    MAX_RETRIES=3
    SUCCESS=0

    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "[INFO] Attempt $attempt downloading WAT paths for $CRAWL..."
        wget -q "$BASE_URL/crawl-data/$CRAWL/wat.paths.gz" -O "$WAT_PATHS_FILE"

        if gzip -t "$WAT_PATHS_FILE"; then
            echo "[INFO] Valid WAT file downloaded."
            SUCCESS=1
            break
        else
            echo "[WARN] Corrupted WAT file (attempt $attempt), retrying..."
            rm -f "$WAT_PATHS_FILE"
            sleep 2
        fi
    done

    if [[ "$SUCCESS" -ne 1 ]]; then
        echo "[ERROR] Failed to get valid WAT paths file for $CRAWL after $MAX_RETRIES attempts" >&2
        return 1
    fi

    TOTAL_FILES=$(gzip -dc "$WAT_PATHS_FILE" | wc -l)
    echo "[INFO] Total WAT files: $TOTAL_FILES"

    FILES_PER_SUBSET=$(( (TOTAL_FILES + NUM_SUBFOLDERS - 1) / NUM_SUBFOLDERS ))
    echo "[INFO] Splitting into $NUM_SUBFOLDERS subsets, ~$FILES_PER_SUBSET files each"

    CRAWL_DIR="$CONSTRUCTION_DIR/$CRAWL"
    mkdir -p "$CRAWL_DIR"

    pids=()

    for (( i=0; i<NUM_SUBFOLDERS; i++ )); do
        (
            SUBFOLDER_ID=$((i + 1))
            SUBFOLDER_NAME="bash_scripts$SUBFOLDER_ID"
            OUTPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/output$SUBFOLDER_ID"
            SEGMENT_DIR="$DATA_DIR/crawl-data/$CRAWL/segments$SUBFOLDER_ID"
            TARGET_SCRIPTS="$CRAWL_DIR/$SUBFOLDER_NAME"

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
        ) >"$CONSTRUCTION_DIR/logs/${CRAWL}_sub$((i+1))_out.log" \
          2>"$CONSTRUCTION_DIR/logs/${CRAWL}_sub$((i+1))_err.log" &

        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "[INFO][$CRAWL] All subsets done. Starting merge..."

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

# Launches each crawl in parallel with stdout and stderr logs.
for CRAWL in $CRAWL_INDICES; do
    process_crawl "$CRAWL" \
      >"$CONSTRUCTION_DIR/logs/${CRAWL}_main_out.log" \
      2>"$CONSTRUCTION_DIR/logs/${CRAWL}_main_err.log" &
done

wait

# Clean out subfolders
for dir in "$CONSTRUCTION_DIR"/CC-*; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
    fi
done

echo "[INFO] Cleanup done."

echo "All crawls completed."
