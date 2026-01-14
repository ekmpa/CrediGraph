#!/bin/bash
set -e

fetch_with_retries() {
  local url="$1" out="$2" tries="${3:-10}"
  local tmp="${out}.part" i status sleep_time
  mkdir -p "$(dirname "$out")"
  for ((i=1; i<=tries; i++)); do
    wget -q -c --tries=1 \
         --retry-connrefused \
         --retry-on-http-error=429,500,502,503,504 \
         --timeout=60 --read-timeout=60 \
         -O "$tmp" "$url" && status=0 || status=$?
#    echo "status=$status tmp=$tmp"
    if [[ $status -eq 0 ]] && ([[ $url = *.parquet ]] || ([[ $url = *.gz ]] && gzip -t "$tmp" 2>/dev/null)); then
      echo "url=$url"
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

# args
if [ -z "$1" ]; then
  echo "Usage: $0 <CRAWL-ID>"
  echo "Example: $0 CC-MAIN-2017-13"
  exit 1
fi

CRAWL="$1"

if [ -z "$2" ]; then
      start_idx=1
else
      start_idx=$2
fi

if [ -z "$3" ]; then
      end_idx=10
else
      end_idx=$3
fi

if [ -z "$4" ]; then
      cc_file_types=('wat')
else
    cleaned=${4:1:-1}  # Removes the first and last character
    echo "cleaned $cleaned"
    IFS=',' read -ra cc_file_types <<< "$cleaned"  # 2. Convert comma-separated string to array0
fi


if [ -z "$5" ]; then
      listing="0"
else
      listing=$5
fi

echo "cc_file_types= ${cc_file_types[@]}"
echo "start_idx=$start_idx end_idx=$end_idx"
echo "listing path=$listing"

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SPARK_WAREHOUSE="spark-warehouse"
BASE_URL=https://data.commoncrawl.org # Base URL used to download the path listings

if [ -z "$SCRATCH" ]; then
    echo "[WARN] SCRATCH not set, using local data directory."
    DATA_DIR="$PROJECT_ROOT/data"
else
    DATA_DIR="$SCRATCH"
    echo "Using SCRATCH directory: $DATA_DIR"
fi

mkdir -p "$DATA_DIR/"
INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"
mkdir -p "$INPUT_DIR"

#if [ -d "$SPARK_WAREHOUSE" ]; then
#    rm -r "$SPARK_WAREHOUSE"/*
#fi

#for data_type in warc wat wet; do
for data_type in  "${cc_file_types[@]}" ; do
  echo "CC File Type= $data_type"
  echo "Downloading Common Crawl paths listings (${data_type} files of $CRAWL)..."

  mkdir -p "$DATA_DIR/crawl-data/$CRAWL/"
  all_listing_content_path="$INPUT_DIR/${CRAWL}_test_all_${data_type}.txt"
  echo "all_listing_content_path=$all_listing_content_path"
  
  if [[ "$listing" == "0" ]]; then
      echo "listing is not provided"
      listing="$DATA_DIR/crawl-data/$CRAWL/$data_type.paths.gz"
      if [ -e "$all_listing_content_path" ]; then ## listing paths has been downloaded in previouse batches
          echo "$all_listing_content_path exist."
          listing_content=$(<"$all_listing_content_path")  
      else  ## listing paths is to be downloaded
          cd "$DATA_DIR/crawl-data/$CRAWL/"
          wget  -q --timestamping "$BASE_URL/crawl-data/$CRAWL/$data_type.paths.gz"
          cd -
          echo "Downloading ${data_type} file paths..."
          file=$(gzip -dc "$listing" | head -1)
          full_path="$DATA_DIR/$file"
          mkdir -p "$(dirname "$full_path")"
          cd "$(dirname "$full_path")"
          wget -q --timestamping "$BASE_URL/$file"
          cd -
          input="$INPUT_DIR/all_${data_type}_${CRAWL}.txt"
          echo "All ${data_type} files of ${CRAWL}: $input"
          listing_content=$(gzip -dc "$listing")
          if [ -e "$all_listing_content_path" ]; then
              rm "$all_listing_content_path"
          fi
          echo "$listing_content" >>"$all_listing_content_path"
      fi
  else   ############### Listing paths is given in a certian order (index)
      echo "listing is provided"
      listing_content=$(<"$listing")    
      # echo "$listing_content"
      if [ -e "$all_listing_content_path" ]; then
              rm "$all_listing_content_path"
      fi
      echo "$listing_content" >>"$all_listing_content_path"
  fi    

  echo "Downloading sample ${data_type} file..."
  # make sample fetch non-fatal so we reach the main loop even if it 503s
  file="$(gzip -dc "$listing" | head -1 || true)"
  if [ -n "$file" ]; then
    full_path="$DATA_DIR/$file"
    mkdir -p "$(dirname "$full_path")"
    ( cd "$(dirname "$full_path")" && wget -q --timestamping "$BASE_URL/$file" ) || \
      echo "[WARN] sample $data_type fetch failed; continuing"
  fi

  input="$INPUT_DIR/all_${data_type}_${CRAWL}.txt"
  echo "All ${data_type} files of ${CRAWL}: $input"
  all_listing_content_path="$INPUT_DIR/${CRAWL}_test_all_${data_type}.txt"

  if [[ "$data_type" = "cc-index-table" ]]; then
    listing_content=$(gzip -dc "$listing" | grep -F "/subset=warc/")
    # echo "cc-index-table listing_content=$listing_content"
  elif ["$listing" = "0" ]; then
    listing_content=$(gzip -dc "$listing")
  fi

  echo "file:$listing_content" >>"$all_listing_content_path"
  listing_FilesCount=$(wc -l <<< "$listing_content")
  echo "listing_FilesCount=$listing_FilesCount"
  if [ "$listing_FilesCount" -lt "$end_idx" ] ; then
    end_idx=$listing_FilesCount
  fi
  echo "end_idx=$end_idx"
  FilesCount=$((end_idx - start_idx + 1))
  input="$INPUT_DIR/${CRAWL}_test_${data_type}_${start_idx}_${end_idx}.txt"
  start_idx=$((start_idx + 1))
  echo "To Process FilesCount=$FilesCount"
  wat_files=$(echo "$listing_content" | tail -n +$start_idx | head -n $FilesCount)
  echo "Writing input file listings..."  
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
  fail_log="$DATA_DIR/crawl-data/$CRAWL/failed-${data_type}.txt"
  : > "$fail_log"

  while IFS= read -r wat_file; do
    echo "wat_file=$wat_file"
    if [[ "$data_type" = "cc-index-table" ]]; then
      first=$(echo "$wat_file" | awk -F '/subset=warc/' '{print $1}')
    else
      first=$(echo "$wat_file" | awk -F '/'$data_type'/' '{print $1}')
    fi
    # echo "first=$first"
    file_path="$DATA_DIR/$wat_file"
    # echo "file_path=$file_path"
    if [[ "$data_type" = "cc-index-table" ]]; then
      target_dir="$DATA_DIR/$first/subset=warc/"
    else
      target_dir="$DATA_DIR/$first/$data_type/"
    fi
    # echo "target_dir=$target_dir"
    target_file="${target_dir}$(basename "$wat_file")"
    # echo "target_file=$target_file"

    if [ -f "$file_path" ] || [ -f "$target_file" ]; then
      # verify gzip; re-download if corrupt
      test -f "$file_path" && cand="$file_path" || cand="$target_file"
      echo "cand=$cand"
      if [[ $cand = *.parquet ]] || ([[ $cand = *.gz ]] && gzip -t "$cand" 2>/dev/null); then
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
done