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
subfolder_name="bash_scripts${subfolder_id}"

fetch_with_retries() {
  local url="$1" out="$2" tries="${3:-10}"
  local tmp="${out}.part" i status sleep_time
  mkdir -p "$(dirname "$out")"
  for ((i=1; i<=tries; i++)); do
    wget -c --tries=1 \
         --retry-connrefused \
         --retry-on-http-error=429,500,502,503,504 \
         --timeout=60 --read-timeout=60 \
         -O "$tmp" "$url" && status=0 || status=$?
    if [[ $status -eq 0 ]] && gzip -t "$tmp" 2>/dev/null; then
      mv -f "$tmp" "$out"
      return 0
    fi
    rm -f "$tmp" 2>/dev/null || true
    (( i == tries )) && break
    sleep_time=2 #$(( (2**i) < 60 ? (2**i) : 60 ))
    echo "[WARN] attempt $i failed for $url; retrying in ${sleep_time}s..."
    sleep "$sleep_time"
  done
  echo "[ERR] giving up on $url after $tries tries"
  return 8
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCTION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"

# echo "cc_file_types= ${cc_file_types[@]}"
# echo "start_idx=$start_idx end_idx=$end_idx"

# Get the root of the project (one level above this script's directory)
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# SPARK_WAREHOUSE="spark-warehouse"
BASE_URL=https://data.commoncrawl.org # Base URL used to download the path listings

if [ -z "$SCRATCH" ]; then
    echo "[WARN] SCRATCH not set, using local data directory."
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH/crawl-data" # was just $SCRATCH?
    echo "Using SCRATCH directory: $DATA_DIR"
fi

mkdir -p "$DATA_DIR/"
INPUT_DIR="$DATA_DIR/$CRAWL/input$subfolder_id"
mkdir -p "$INPUT_DIR"

# if [ -d "$SPARK_WAREHOUSE" ]; then this happens in end-to-end.sh now
#     rm -rf "$SPARK_WAREHOUSE"
# fi

#for data_type in warc wat wet; do
echo "CC File Type= $data_type"
echo "Downloading Common Crawl paths listings (${data_type} files of $CRAWL)..."

mkdir -p "$DATA_DIR/$CRAWL/"
listing="$DATA_DIR/$CRAWL/$data_type.paths.gz"
cd "$DATA_DIR/$CRAWL/"
wget --timestamping "$BASE_URL/crawl-data/$CRAWL/$data_type.paths.gz"
sleep 2
cd -

echo "Downloading sample ${data_type} file..."
# make sample fetch non-fatal so we reach the main loop even if it 503s
file="$(gzip -dc "$listing" | head -1 || true)"
if [ -n "$file" ]; then
  full_path="$DATA_DIR/$file"
  mkdir -p "$(dirname "$full_path")"
  ( cd "$(dirname "$full_path")" && wget --timestamping "$BASE_URL/$file" ) || \
    echo "[WARN] sample $data_type fetch failed; continuing"
fi

input="$INPUT_DIR/all_${data_type}_${CRAWL}.txt"
echo "All ${data_type} files of ${CRAWL}: $input"
listing_content=$(gzip -dc "$listing")
all_listing_content_path="$INPUT_DIR/test_all_${data_type}.txt"
echo "file:$listing_content" >>"$all_listing_content_path"

listing_FilesCount=$(wc -l <<< "$listing_content")
echo "listing_FilesCount=$listing_FilesCount"
if [ "$listing_FilesCount" -lt "$end_idx" ] ; then
  end_idx=$listing_FilesCount
fi
FilesCount=$((end_idx - start_idx + 1))
start_idx=$((start_idx + 1))
echo "To Process FilesCount=$FilesCount"

wat_files=$(echo "$listing_content" | tail -n +$start_idx | head -n $FilesCount)
echo "Writing input file listings..."
input="$INPUT_DIR/test_${data_type}.txt"
echo "Test file: $input"
if [ -e "$input" ]; then
  rm "$input"
  echo "File $input already existed. deleted it."
fi
while IFS= read -r wat_file; do
  echo "file:$DATA_DIR/$wat_file" >>"$input"
done <<< "$wat_files"

echo "############Downloading Files @ $(date '+%Y-%m-%d %H:%M:%S') ############"

# counters + skip log
skipped=0
downloaded=0
fail_log="$DATA_DIR/$CRAWL/failed-${data_type}.txt"
: > "$fail_log"

while IFS= read -r wat_file; do
  first=$(echo "$wat_file" | awk -F '/'$data_type'/' '{print $1}')
  file_path="$DATA_DIR/$wat_file"
  target_dir="$DATA_DIR/$first/$data_type/"
  target_file="${target_dir}$(basename "$wat_file")"

  if [ -f "$file_path" ] || [ -f "$target_file" ]; then
    # verify gzip; re-download if corrupt
    test -f "$file_path" && cand="$file_path" || cand="$target_file"
    if gzip -t "$cand" 2>/dev/null; then
      echo "File '$cand' exists."
      downloaded=$((downloaded+1))
      continue
    else
      echo "[WARN] corrupt '$cand' — re-downloading"
      rm -f "$cand"
    fi
  fi

  # retry this one file; skip if it keeps failing
  url="$BASE_URL/$wat_file"
  if fetch_with_retries "$url" "$target_file" 5; then
    #echo "[OK] $target_file"
    downloaded=$((downloaded+1))
  else
    echo "[SKIP] $url"
    echo "$url" >> "$fail_log"
    skipped=$((skipped+1))
    continue
  fi

done <<< "$wat_files"
echo "[SUMMARY] $CRAWL type=$data_type downloaded=$downloaded skipped=$skipped fail_log=$fail_log"
