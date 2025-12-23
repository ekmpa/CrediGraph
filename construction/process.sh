#!/usr/bin/env bash
set -euo pipefail

START_MONTH="$1"
END_MONTH="$2"
MIN_DEG=3

if [ -z "${SCRATCH:-}" ]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
fi

export PYTHONPATH="$(pwd)/.."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONSTRUCTION_DIR="$PROJECT_ROOT/construction"

get_cc_indices() {
    uv run python -c "
from tgrag.utils.data_loading import interval_to_CC_slices
indices = interval_to_CC_slices(\"$1\", \"$2\")
print(' '.join(indices))
"
}


process() {
    local CRAWL=$1
    local MIN_DEG=$2

    echo "[INFO] Starting crawl: $CRAWL"

    GRAPH_DIR="$DATA_DIR/crawl-data/$CRAWL/output1"
    SUBFOLDER_NAME="bash_scripts1"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    CONSTRUCTION_DIR="$PROJECT_ROOT/construction"
    TEMPLATE_SCRIPTS="$CONSTRUCTION_DIR/bash_scripts"
    CRAWL_DIR="$CONSTRUCTION_DIR/$CRAWL"
    TARGET_SCRIPTS="$CRAWL_DIR/$SUBFOLDER_NAME"

    mkdir -p "$CRAWL_DIR"

    if [ ! -d "$TARGET_SCRIPTS" ]; then
        tmp_dir="${TARGET_SCRIPTS}.tmp.$$"
        cp -r "$TEMPLATE_SCRIPTS" "$tmp_dir"
        mv "$tmp_dir" "$TARGET_SCRIPTS"
    fi

    echo $GRAPH_DIR
    echo "#####################  start process @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
    bash $TARGET_SCRIPTS/process_graph.sh $GRAPH_DIR $CRAWL $MIN_DEG "20%"
    echo "#####################  end process @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
    
}

CRAWL_INDICES=$(get_cc_indices "$START_MONTH" "$END_MONTH")

for CRAWL in $CRAWL_INDICES; do
    process $CRAWL $MIN_DEG \
      >"$CONSTRUCTION_DIR/logs/${CRAWL}_process_out.log" \
      2>"$CONSTRUCTION_DIR/logs/${CRAWL}_process_err.log" &
done

wait
echo "[INFO] Process done."
