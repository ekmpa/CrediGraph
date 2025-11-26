#!/bin/bash
set -e

# Check if crawl list file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <crawl-list.txt>"
    exit 1
fi

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


echo "cc_file_types= ${cc_file_types[@]}"
echo "start_idx=$start_idx end_idx=$end_idx"

CRAWL_LIST_FILE="$1"

if [ ! -f "$CRAWL_LIST_FILE" ]; then
    echo "File not found: $CRAWL_LIST_FILE"
    exit 1
fi

# Get the root of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$DATA_DIR/crawl-data/$CRAWL/input"
for data_type in  "${cc_file_types[@]}" ; do
  while read -r CRAWL || [[ -n "$CRAWL" ]]; do
      # Skip empty lines or comments
      [[ -z "$CRAWL" || "$CRAWL" =~ ^# ]] && continue
      echo "Processing $CRAWL..."
      echo "Removing previous $CRAWL spark-warehouse"
      # Use SCRATCH if defined, else fallback to project-local data dir
      # For cluster usage
#      if [ -z "$SCRATCH" ]; then
#          rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse" # Remove re-created directories before running
#      else
#          rm -rf "$SCRATCH/spark-warehouse" # Remove re-created directories before running
#      fi

      echo $CRAWL
      echo "################################### run get data ###################################"
      ./get_data.sh "$CRAWL" $start_idx $end_idx "[$data_type]"
      echo "Data Downloaded for $CRAWL."
      echo "################ Start Processing Processing $data_type Files ######################"
      lCRAWL="${CRAWL,,}" # to lower string
      if [ "$data_type" = "wat" ]; then
        if [ -z "$SCRATCH" ]; then
          rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse/wat_link_to_graph_batch_${lCRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
        else
            rm -rf "$SCRATCH/spark-warehouse/wat_link_to_graph_batch_${lCRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
        fi
        echo "#####################  run_wat_to_link #####################"
        ./run_wat_to_link.sh "$CRAWL"
        echo "wat_output_table constructed for $CRAWL."
        echo "#####################  run_link_to_graph #####################"
        ./run_link_to_graph.sh "$CRAWL"
        echo "Compressed graphs constructed for $CRAWL batch_${start_idx}_${end_idx}"
      elif [ "$data_type" = "wet" ]; then
        echo "#####################  run_wet_content_extraction #####################"
         if [ -z "$SCRATCH" ]; then
            spark_table_name="wet_domain_urls_batch_"
            rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse/${spark_table_name}${lCRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
         else
            rm -rf "$SCRATCH/spark-warehouse/${spark_table_name}${lCRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
         fi
        ./run_extract_wet_urls.sh "$CRAWL" "${spark_table_name}${lCRAWL//-/}_${start_idx}_${end_idx}"
        echo "wet_extract_content_table constructed for $CRAWL batch_${start_idx}_${end_idx}"
      fi
  echo "********************** End Of $data_type Task **********************"
  done
done < "$CRAWL_LIST_FILE"
