#!/bin/bash
set -e
batch_size=25
for ((i=25; i<90001; i+=$batch_size)); do
    echo "#########################################################################################"
    echo "/end-to-end-content.sh  CC-Crawls/Feb2025 $i $((i+$batch_size-1)) [wet] ../data/Entire-Graph_Nov-2024/credigraph_Nov2024_wetFilesOrder.txt"
    ./end-to-end.sh  CC-Crawls/Feb2025.txt $i $((i+$batch_size-1)) [wet] ../data/Feb2025/Feb2025_domains.csv content_table spark-warehouse/credigraph_ccmain202508_wetFilesOrder.txt
    rm -r  ~/scratch/crawl-data/CC-MAIN-2025-08/segments
    #  rm -r ../data/crawl-data/CC-MAIN-2024-42/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-46/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-51/segments
    # rm -r /shared_mnt/tmp/*
done

