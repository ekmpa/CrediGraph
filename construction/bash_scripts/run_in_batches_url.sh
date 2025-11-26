#!/bin/bash
set -e
for ((i=0; i<90001; i+=50)); do
    echo "#########################################################################################"
    echo "/end-to-end-url.sh  CC-Crawls/Oct2024.txt $i $((i+49)) [wet]"
    ./end-to-end-url.sh  CC-Crawls/Oct2024.txt $i $((i+49)) [wet]
    rm -r ../data/crawl-data/CC-MAIN-2024-42/segments
done
