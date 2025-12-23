#!/bin/bash
set -e
for ((i=0; i<90001; i+=100)); do
    echo "#########################################################################################"
    echo "/end-to-end-content.sh  CC-Crawls/Dec2024txt $i $((i+99)) [wet]"
    ./end-to-end.sh  CC-Crawls/Dec2024.txt $i $((i+99)) [wet]
     rm -r ../data/crawl-data/CC-MAIN-2024-42/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-46/segments
    # rm -r ../data/crawl-data/CC-MAIN-2024-51/segments
    rm -r /shared_mnt/tmp/*
done
