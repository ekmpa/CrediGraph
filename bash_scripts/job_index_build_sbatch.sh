# CRAWL=ccmain202508
CRAWL=None
# Month=Feb2025
# Month=Jan2025
Month=Mar2025
batch_size=10
echo sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 240 300 $batch_size
sbatch ./run_in_batches_warc-index.sh $CRAWL $Month 240 300 $batch_size
