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


if [ -z "$5" ]; then
      seed_list='../data/dqr/domain_pc1.csv'
else
      seed_list=$5
fi

if [ -z "$6" ]; then
      spark_table_name='content_table'
else
      spark_table_name=$6
fi

echo "cc_file_types= ${cc_file_types[@]}"
echo "start_idx=$start_idx end_idx=$end_idx"
echo "seed_list=$seed_list"
echo "spark_table_name=$spark_table_name"

CRAWL_ARG="$1"

if [ -f "$CRAWL_ARG" ]; then
  echo "Crawl list file detected: $CRAWL_ARG"
  #CRAWLS=$(cat "$CRAWL_ARG")
  mapfile -t CRAWLS < "$CRAWL_ARG"
else
  echo "Single crawl ID detected: $CRAWL_ARG"
  CRAWLS="$CRAWL_ARG"
fi

# Get the root of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

for data_type in  "${cc_file_types[@]}" ; do
  #while read -r CRAWL || [[ -n "$CRAWL" ]]; do
  for CRAWL in "${CRAWLS[@]}"; do
      # Skip empty lines or comments
      [[ -z "$CRAWL" || "$CRAWL" =~ ^# ]] && continue
      echo "Processing $CRAWL..."
      echo "Removing previous $CRAWL spark-warehouse"
      rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse/$CRAWL"
      echo $CRAWL
      echo "################################### run get data @ $(date '+%Y-%m-%d %H:%M:%S') ###################################"
      echo "$SCRIPT_DIR/get_data.sh" "$CRAWL" $start_idx $end_idx "[$data_type]"
      "$SCRIPT_DIR/get_data.sh" "$CRAWL" $start_idx $end_idx "[$data_type]"
      echo "Data Downloaded for $CRAWL."
      if [ "$data_type" = "wat" ]; then
        echo "################ Start Processing Processing $data_type Files ######################"
        echo "#####################  run_wat_to_link @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
        "$SCRIPT_DIR/run_wat_to_link.sh" "$CRAWL"
        echo "wat_output_table constructed for $CRAWL."
        echo "#####################  run_link_to_graph @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
        "$SCRIPT_DIR/run_link_to_graph.sh" "$CRAWL"
        echo "Compressed graphs constructed for $CRAWL."
      elif [ "$data_type" = "wet" ]; then
         echo "#####################  run_wet_content_extraction @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
         if [ -z "$SCRATCH" ]; then
            rm -rf "$PROJECT_ROOT/bash_scripts/spark-warehouse/wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
         else
            rm -rf "$SCRATCH/spark-warehouse/wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
         fi
         echo "$SCRIPT_DIR/run_extract_wet_content.sh" "$CRAWL" "wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}" "$seed_list"
         "$SCRIPT_DIR/run_extract_wet_content.sh" "$CRAWL" "wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}" "$seed_list"
         echo "wet_extract_content_table constructed for $CRAWL batch_${start_idx}_${end_idx}"
      fi
  echo "********************** End Of $data_type Task @ $(date '+%Y-%m-%d %H:%M:%S') **********************"
  done
done #< "$CRAWLS"
