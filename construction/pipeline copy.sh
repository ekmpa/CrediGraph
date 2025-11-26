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

write_summary() {
    mkdir -p logs
    SUMMARY_FILE="logs/${CRAWL}-summary.txt"
    {
        echo "Nodes: $TOTAL_NODES"
        echo "Edges: $TOTAL_EDGES"
        echo "Steps: $TOTAL_STEPS"
        echo "Files decompressed: $TOTAL_FILES_DECOMPRESSED"
        # TO DO: add stats from topological experiments in summary file
    } > "$SUMMARY_FILE"
    echo "Summary written to $SUMMARY_FILE"
}

while read -r CRAWL || [[ -n "$CRAWL" ]]; do
    TOTAL_NODES=0
    TOTAL_EDGES=0
    TOTAL_STEPS=0
    TOTAL_FILES_DECOMPRESSED=0

    trap write_summary EXIT # on exit from error, write summary

    START_IDX=0
    if [ "$2" != "--keep" ]; then
        rm -rf "$DATA_DIR/$CRAWL/output"
    else
        if [[ "$3" =~ ^[0-9]+$ ]]; then
                START_IDX="$3"
        fi
    fi
    mkdir -p "$DATA_DIR/$CRAWL/output"

    wget -q "$BASE_URL/crawl-data/$CRAWL/wat.paths.gz" -O /tmp/wat.paths.gz
    TOTAL_FILES=$(gzip -dc /tmp/wat.paths.gz | wc -l)
    echo "Total WAT files: $TOTAL_FILES"

    STEP=300

    for (( start_idx=$START_IDX; start_idx<=TOTAL_FILES; start_idx+=STEP )); do

        end_idx=$((start_idx+STEP-1))

        TOTAL_STEPS=$((TOTAL_STEPS+1))
        slice_size=$((end_idx - start_idx + 1))
        TOTAL_FILES_DECOMPRESSED=$((TOTAL_FILES_DECOMPRESSED + slice_size))

        if [ "$end_idx" -gt "$TOTAL_FILES" ]; then # Clamp to TOTAL_FILES if end_idx exceeds it
            end_idx=$TOTAL_FILES
        fi

        echo "Running on: $start_idx-$end_idx"

        # clean the segments folder
        rm -rf "$DATA_DIR/$CRAWL/segments/"*

        # Run end-to-end on the given subset + aggregate to partial graph
        bash end-to-end.sh "$CRAWL" $start_idx $end_idx
        uv run python ../tgrag/construct_graph_scripts/construct_aggregate.py --source "$DATA_DIR/$CRAWL/output_text_dir" --target "$DATA_DIR/$CRAWL/output"

        for f in edges vertices; do
            target_file="$DATA_DIR/$CRAWL/output/${f}.txt.gz"
            if [ -f "$target_file" ]; then
                num_lines=$(gzip -dc "$target_file" | wc -l)
                echo "[INFO] After slice $start_idx-$end_idx: $f has $num_lines records"

                if [ "$f" == "edges" ]; then
                    TOTAL_EDGES=$num_lines
                else
                    TOTAL_NODES=$num_lines
                fi
            fi
        done
    done

    echo "Finished $CRAWL"

    trap - EXIT
    write_summary # when slice is done, write summary

done < "$CRAWL_LIST_FILE"
