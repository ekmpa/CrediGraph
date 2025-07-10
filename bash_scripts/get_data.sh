#!/bin/bash
set -e

# Check if CRAWL argument is provided
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

if [ -z "$2" ]; then
      end_idx=30
else
      end_idx=$3
fi
echo "start_idx=$start_idx end_idx=$end_idx"

# Get the root of the project (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Use SCRATCH if defined, else fallback to project-local data dir
# For cluster use
if [ -z "$SCRATCH" ]; then
    echo "[WARN] SCRATCH not set, using local data directory."
    DATA_DIR="$PROJECT_ROOT/data"
    SPARK_WAREHOUSE="spark-warehouse"
else
    DATA_DIR="$SCRATCH"
    SPARK_WAREHOUSE="$SCRATCH/CrediGraph/spark-warehouse"
    echo "Using SCRATCH directory: $DATA_DIR"
fi

# Base URL used to download the path listings
BASE_URL=https://data.commoncrawl.org


mkdir -p "$DATA_DIR/"
INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"
mkdir -p "$INPUT_DIR"


#for data_type in warc wat wet; do
for data_type in  wat ; do
#    echo "data_type= $data_type"
    echo "Downloading Common Crawl paths listings (${data_type} files of $CRAWL)..."

    mkdir -p "$DATA_DIR/crawl-data/$CRAWL/"
    listing="$DATA_DIR/crawl-data/$CRAWL/$data_type.paths.gz"
    cd "$DATA_DIR/crawl-data/$CRAWL/"
    wget --timestamping "$BASE_URL/crawl-data/$CRAWL/$data_type.paths.gz"
    cd -

    echo "Downloading sample ${data_type} file..."

    file=$(gzip -dc "$listing" | head -1)
    full_path="$DATA_DIR/$file"
    mkdir -p "$(dirname "$full_path")"
    cd "$(dirname "$full_path")"
    wget --timestamping "$BASE_URL/$file"
    cd -
    input="$INPUT_DIR/all_${data_type}_${CRAWL}.txt"
    echo "All ${data_type} files of ${CRAWL}: $input"
    listing_content=$(gzip -dc "$listing")
#    echo "listing_content=$listing_content"
    listing_FilesCount=$(wc -l <<< "$listing_content")
    echo "listing_FilesCount=$listing_FilesCount"
    if [ "$listing_FilesCount" -lt "$end_idx" ] ; then
      end_idx=listing_FilesCount
    fi
    FilesCount=$((end_idx - start_idx+1))
    start_idx=$((start_idx+1))
    echo "To Process FilesCount=$FilesCount"
#    echo " tail -n +$start_idx | head -n $FilesCount"
    wat_files=$(echo "$listing_content" | tail -n +$start_idx | head -n $FilesCount)
    echo "Writing input file listings..."
    input="$INPUT_DIR/test_${data_type}.txt"
    echo "Test file: $input"
    if [ -e "$input" ]; then
        rm "$input"
        echo "File $input already exists. delete it."
    fi
    while IFS= read -r wat_file; do
      echo "file:$DATA_DIR/$wat_file" >>"$input"
     done <<< "$wat_files"
    echo "############Downloading Files############"
    while IFS= read -r wat_file; do
      echo "$wat_file"
      # split file name by
      first=$(echo "$wat_file" | awk -F '$BASE_URL' '{print $1}')
      first=$(echo "$first" | awk -F '/'$data_type'/' '{print $1}')
#      echo "first=" "$first"
      file_path="../data/$wat_file"
      if [ -f "$file_path" ]; then
          echo "File '$file_path' exists."
      else
          wget --timestamping -P "../data/$first/$data_type/" "$BASE_URL/$wat_file"
      fi
    done <<< "$wat_files"

done
