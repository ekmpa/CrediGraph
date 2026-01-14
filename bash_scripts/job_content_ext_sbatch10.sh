# CRAWL=ccmain202508
# Month=Feb2025

CRAWL=ccmain202505
Month=Jan2025

# CRAWL=ccmain202513
# Month=Mar2025



batch_size=5000
for ((i=50000; i<70000; i+=$batch_size)); do
    if ((i >= 20000)); then
        batch_size=10000
    fi
    end=$(($i+$batch_size))
    echo sbatch ./job_content_ext.sh $CRAWL $Month $i $end 50
    sbatch ./job_content_ext.sh $CRAWL $Month $i $end 50
done
# sbatch ./job_content_ext.sh ccmain202508 Feb2025 0 10000 50