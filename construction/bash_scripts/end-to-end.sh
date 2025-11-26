#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <crawl> <start_idx> <end_idx> <subfolder_id> [data_type] [spark_table_name]"
    echo "Example: $0 CC-MAIN-2025-10 0 10000 6 <wat/wet> <content_table_name>"
    exit 1
fi

CRAWL="$1"
start_idx="$2"
end_idx="$3"
subfolder_id="$4"
data_type="${5:-wat}"
spark_table_name="${6:-content_table}"
#subfolder_name="bash_scripts${subfolder_id}"
subfolder_name="$(basename "$(dirname "${BASH_SOURCE[0]}")")"  # auto-detect parent dir

# set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBFOLDER_PATH="$SCRIPT_DIR"
CRAWL_FOLDER="$(dirname "$SUBFOLDER_PATH")"
CONSTRUCTION_DIR="$(dirname "$CRAWL_FOLDER")"
PROJECT_ROOT="$(dirname "$CONSTRUCTION_DIR")"
SPARK_WAREHOUSE_PATH="$SUBFOLDER_PATH/spark-warehouse"
seed_list="$PROJECT_ROOT/data/dqr/domain_pc1.csv"

echo "start_idx=$start_idx end_idx=$end_idx"
echo "seed_list=$seed_list"
echo "spark_table_name=$spark_table_name"
echo "subfolder_ID=$subfolder_id"

if [ -d "$SPARK_WAREHOUSE_PATH" ]; then
    echo "[INFO] Removing previous $CRAWL spark-warehouse"
    rm -rf "$SPARK_WAREHOUSE_PATH"
fi

echo "################################### run get data @ $(date '+%Y-%m-%d %H:%M:%S') ###################################"
"$SUBFOLDER_PATH/get_data.sh" "$CRAWL" "$start_idx" "$end_idx" "[$data_type]" "$subfolder_id"
echo "[INFO] Data downloaded for $CRAWL"


if [ "$data_type" = "wat" ]; then
    echo "#####################  run_wat_to_link @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
    "$SUBFOLDER_PATH/run_wat_to_link.sh" "$CRAWL" "$subfolder_id"
    echo "[INFO] wat_output_table constructed for $CRAWL in $SPARK_WAREHOUSE_PATH."
    echo "#####################  run_link_to_graph @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
    "$SUBFOLDER_PATH/run_link_to_graph.sh" "$CRAWL" "$subfolder_id"
    echo "[INFO] Compressed graphs constructed for $CRAWL."

elif [ "$data_type" = "wet" ]; then
      echo "#####################  run_wet_content_extraction @ $(date '+%Y-%m-%d %H:%M:%S') #####################"
      rm -rf "$SUBFOLDER_PATH/spark-warehouse/wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}/" # Remove re-created directories before running
      echo "$SUBFOLDER_PATH/run_extract_wet_content.sh" "$CRAWL" "wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}" "$seed_list"
      "$SUBFOLDER_PATH/run_extract_wet_content.sh" "$CRAWL" "wet_${spark_table_name}_${CRAWL//-/}_${start_idx}_${end_idx}" "$seed_list"
      echo "wet_extract_content_table constructed for $CRAWL batch_${start_idx}_${end_idx}"
fi
echo "********************** End Of $data_type Task @ $(date '+%Y-%m-%d %H:%M:%S') **********************"
