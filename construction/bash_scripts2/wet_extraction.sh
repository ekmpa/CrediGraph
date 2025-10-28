#!/bin/bash

set -e

CRAWL_ID=$1
DOMAINS_FILE=$2  # optional
MAX_DOCS=${3:-100}

if [[ -z "$CRAWL_ID" ]]; then
  echo "Usage: $0 <CRAWL_ID> [domains_file_or_vertex_dir] [max_docs]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SEGMENT_DIR="$PROJECT_ROOT/data/crawl-data/$CRAWL_ID/segments"
OUTPUT_DIR="$PROJECT_ROOT/data/crawl-data/$CRAWL_ID/output_text_dir/articles"
VERTEX_DIR="$PROJECT_ROOT/data/crawl-data/$CRAWL_ID/output_text_dir/vertices"
PYTHON_SCRIPT="$PROJECT_ROOT/tgrag/utils/extract_WETs.py"

mkdir -p "$OUTPUT_DIR"

# Check if $DOMAINS_FILE is a file or fallback to vertex dir
USE_VERTEX_DIR=false
if [[ -n "$DOMAINS_FILE" && -f "$DOMAINS_FILE" ]]; then
  DOMAIN_ARG="--filter_domains $DOMAINS_FILE"
elif [[ -d "$VERTEX_DIR" ]]; then
  echo "⚠️  No valid domain file provided. Falling back to vertex dir: $VERTEX_DIR"
  DOMAIN_ARG="--vertex_dir $VERTEX_DIR"
  USE_VERTEX_DIR=true
else
  echo "Neither a domain file nor vertex dir found."
  exit 1
fi

# Loop through wet files
for WET_FILE in "$SEGMENT_DIR"/*/wet/*.warc.wet.gz; do
  SEG_ID=$(basename "$(dirname "$(dirname "$WET_FILE")")")
  OUT_FILE="$OUTPUT_DIR/articles_${SEG_ID}.jsonl"

  echo "▶ Extracting from: $WET_FILE"
  python "$PYTHON_SCRIPT" \
    --wet_file "$WET_FILE" \
    --output_jsonl "$OUT_FILE" \
    $DOMAIN_ARG \
    --max_docs "$MAX_DOCS"
done
