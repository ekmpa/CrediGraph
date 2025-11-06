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
    echo "Using SCRATCH directory: $DATA_DIR"
fi

mkdir -p "$DATA_DIR/crawl-data/$CRAWL/"
INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input$subfolder_id"
mkdir -p "$INPUT_DIR"

echo "CC File Type= $data_type"
listing="$DATA_DIR/crawl-data/$CRAWL/$data_type.paths.gz"

echo "Downloading sample ${data_type} file..."
sample_file="$(gzip -dc "$listing" | head -n 1 || true)"
if [ -n "$sample_file" ]; then
  full_path="$DATA_DIR/crawl-data/$sample_file"
  mkdir -p "$(dirname "$full_path")"
  (cd "$(dirname "$full_path")" && wget --timestamping "$BASE_URL/$sample_file") || \
    echo "[WARN] sample $data_type fetch failed; continuing"
fi

listing_content="$(gzip -dc "$listing")"
listing_count=$(wc -l <<< "$listing_content")
echo "listing_FilesCount=$listing_count"

if [ "$listing_count" -lt "$end_idx" ]; then
  end_idx="$listing_count"
fi
FilesCount=$((end_idx - start_idx + 1))
echo "To Process FilesCount=$FilesCount"

file="$(gzip -dc "$listing" | head -1 || true)"
if [ -n "$file" ]; then
  full_path="$DATA_DIR/$file"
  mkdir -p "$(dirname "$full_path")"
  ( cd "$(dirname "$full_path")" && wget --timestamping "$BASE_URL/$file" ) || \
    echo "[WARN] sample $data_type fetch failed; continuing"
fi

input_file="$INPUT_DIR/test_${data_type}.txt"
[ -f "$input_file" ] && rm "$input_file"
echo "Writing input file listings to $input_file"

gzip -dc "$listing" | sed -n "$((start_idx + 1)),$((end_idx + 1))p" > "$input_file"

echo "Done writing $FilesCount paths to $input_file"

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

while IFS= read -r wat_file; do
  file_path="$DATA_DIR/crawl-data/$wat_file"
  target_file="$file_path"

  if [ -f "$target_file" ] && gzip -t "$target_file" 2>/dev/null; then
    echo "File '$target_file' already exists and is valid."
    downloaded=$((downloaded+1))
    continue
  fi

  url="$BASE_URL/$wat_file"
  if fetch_with_retries "$url" "$target_file" 5; then
    echo "[OK] Downloaded: $target_file"
    downloaded=$((downloaded+1))
  else
    echo "[SKIP] Failed: $url"
    echo "$url" >> "$fail_log"
    skipped=$((skipped+1))
  fi
done < "$input_file"

echo "[SUMMARY] $CRAWL type=$data_type downloaded=$downloaded skipped=$skipped"
echo "[SUMMARY] Failed log: $fail_log"
