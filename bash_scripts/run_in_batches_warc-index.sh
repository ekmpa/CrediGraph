#!/bin/bash
set -e
batch_size=2
data_type="cc-index-table"
for ((i=0; i<300; i+=$batch_size)); do
    echo "#########################################################################################"
    echo "/end-to-end.sh  CC-Crawls/Feb2025.txt $i $((i+batch_size-1)) [$data_type]"
    ./end-to-end.sh  CC-Crawls/Feb2025.txt $i $((i+batch_size-1)) [$data_type]
    rm -r ../data/cc-index/table/cc-main/warc/crawl=CC-MAIN-2025-08/subset=warc
done

