#!/bin/bash
set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: $0 <CRAWL-ID> <start_idx> <end_idx> <[data_type]> <subfolder_id>"
  exit 1
fi

CRAWL="$1"
start_idx="$2"
end_idx="$3"
data_type="${4:1:-1}" # remove brackets
subfolder_id="$5"
subfolder_name="$(basename "$(dirname "${BASH_SOURCE[0]}")")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCTION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
BASE_URL=https://data.commoncrawl.org

if [ -z "$SCRATCH" ]; then
    echo "[WARN] SCRATCH not set, using local data directory."
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
    echo "[INFO] Using SCRATCH directory: $DATA_DIR"
fi

mkdir -p "$DATA_DIR/crawl-data/$CRAWL/"
INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input$subfolder_id"
mkdir -p "$INPUT_DIR"

echo "[INFO] CC File Type= $data_type"
listing="$DATA_DIR/crawl-data/$CRAWL/$data_type.paths.gz"

echo "[INFO] Downloading sample ${data_type} file..."
sample_file="$(gzip -dc "$listing" | head -n 1 || true)"
if [ -n "$sample_file" ]; then
  full_path="$DATA_DIR/crawl-data/$sample_file"
  mkdir -p "$(dirname "$full_path")"
  (cd "$(dirname "$full_path")" && wget --timestamping "$BASE_URL/$sample_file") || \
    echo "[WARN] sample $data_type fetch failed; continuing"
fi

listing_content="$(gzip -dc "$listing")"
listing_count=$(wc -l <<< "$listing_content")
echo "[INFO] listing_FilesCount=$listing_count"

if [ "$listing_count" -lt "$end_idx" ]; then
  end_idx="$listing_count"
fi
FilesCount=$((end_idx - start_idx + 1))
echo "[INFO] To Process FilesCount=$FilesCount"

file="$(gzip -dc "$listing" | head -1 || true)"
if [ -n "$file" ]; then
  full_path="$DATA_DIR/$file"
  mkdir -p "$(dirname "$full_path")"
  ( cd "$(dirname "$full_path")" && wget --timestamping "$BASE_URL/$file" ) || \
    echo "[WARN] sample $data_type fetch failed; continuing"
fi

input_file="$INPUT_DIR/test_${data_type}.txt"
[ -f "$input_file" ] && rm "$input_file"
echo "[INFO] Writing input file listings to $input_file"

gzip -dc "$listing" | sed -n "$((start_idx + 1)),$((end_idx + 1))p" > "$input_file"

echo "[INFO] Done writing $FilesCount paths to $input_file"

echo "$listing_content" > "$INPUT_DIR/all_${data_type}_${CRAWL}_${subfolder_id}.txt"

fetch_with_retries() {
  local url="$1" out="$2" tries="${3:-10}"
  local tmp="${out}.part"
  mkdir -p "$(dirname "$out")"
  for ((i=1; i<=tries; i++)); do
    wget -c --tries=1 \
         --retry-connrefused \
         --retry-on-http-error=429,500,502,503,504 \
         --timeout=60 --read-timeout=60 \
         -O "$tmp" "$url" && gzip -t "$tmp" && mv "$tmp" "$out" && return 0
    echo "[WARN] attempt $i failed for $url; retrying in 2s..."
    rm -f "$tmp"
    sleep 2
  done
  echo "[ERROR] giving up on $url after $tries tries"
  return 8
}

echo "############Downloading Files @ $(date '+%Y-%m-%d %H:%M:%S') ############"

skipped=0
downloaded=0
fail_log="$DATA_DIR/crawl-data/$CRAWL/failed_${data_type}_${CRAWL}_${subfolder_id}.txt"
: > "$fail_log"

PARALLEL_JOBS=6
CHUNK_SIZE=$(( (FilesCount + PARALLEL_JOBS - 1) / PARALLEL_JOBS ))

echo "[INFO] Launching $PARALLEL_JOBS parallel download workers (chunk size: $CHUNK_SIZE)"

worker() {
    local worker_id="$1"
    local start=$(( (worker_id - 1) * CHUNK_SIZE + 1 ))
    local end=$(( worker_id * CHUNK_SIZE ))
    if [ "$start" -gt "$FilesCount" ]; then return; fi
    if [ "$end" -gt "$FilesCount" ]; then end="$FilesCount"; fi

    echo "[INFO] Worker $worker_id handling lines $start–$end"

    sed -n "${start},${end}p" "$input_file" |
    while IFS= read -r wat_file; do
        CUSTOM_SEGMENTS_DIR="segments${subfolder_id}"
        file_path="$DATA_DIR/crawl-data/$CRAWL/$CUSTOM_SEGMENTS_DIR/${wat_file#*/segments/}"
        target_file="$file_path"

        if [ -f "$target_file" ] && gzip -t "$target_file" 2>/dev/null; then
            echo "[Worker $worker_id] Already valid: $target_file"
            continue
        fi

        url="$BASE_URL/$wat_file"
        if fetch_with_retries "$url" "$target_file" 5; then
            echo "[Worker $worker_id] OK: $target_file"
        else
            echo "[Worker $worker_id] FAIL: $url"
            echo "$url" >> "$fail_log"
            skipped=$((skipped+1))
        fi
    done
}

pids=()
for w in $(seq 1 $PARALLEL_JOBS); do
    worker "$w" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[INFO] $CRAWL type=$data_type downloaded=$downloaded skipped=$skipped"
echo "[INFO] Failed log: $fail_log"
