# Bash scripts for data processing

> Note: *the slices available through the pipeline depend on the `collinfo.json` file. While we will try to update the repository regularly and thus recommend checking out the latest version of the repository (see below), if you are looking for more recent monthly slices, you can also fetch the json file from Common Crawl directly.*
> The current version of the repository has data from the following interval:
>
> - Start month: March 2014
> - End month: October 2025
>   Also, *Common Crawl (and CrediBench) technically support earlier indices than the start month indicated below. However, before March 2014, Common Crawl would deliver datasets on a seasonal (or, even earlier, yearly) basis. For consistency in the dataset we only consider the monthly granularity which starts in March 2014.*

## Set-up

To set up `jq`, needed for running the pipeline, you can

```bash
mkdir -p ~/bin
cd ~/bin
curl -L https://github.com/jqlang/jq/releases/latest/download/jq-linux-amd64 -o jq
chmod +x jq
```

Then in the job script, make sure to set `export PATH="$HOME/bin:$PATH"`

## Usage

### Building Temporal Graphs

```sh
bash pipeline.sh <start_month> <end_month> <number of subfolders>
# e.g,
bash pipeline.sh 'January 2020' 'February 2020' 9
```

where the format `<month name YYYY>` must be respected, and the number of subfolders designated the extent of parallelization of the graph construction. Note that not parallelizing the construction means a graph takes up to 600 hours to be processed and produced, whereas high level of parallelization can take this down to ... ?

> ***Note:***
> With a subfolder division factor set to 8 (with 8 CPUs), currently, the system uses, per batch:
>
> - 10 minute for downloading 300 WAT files
> - 3 hours for processing it into a graph
> - 15 minutes (on average, increases with size) to aggregate

> ***Note:***
>
> At times out of our control, the Common Crawl connection may be down. The CrediBench pipeline has safeguards against this (retries and wait periods), but in some cases, the server is just offline for a short amount of time. In the pipeline, this will appear most often as a `RuntimeError` in the high-level pipeline (before going into data download) that indicates the fetch from `https://index.commoncrawl.org/collinfo.json` failed.

#### (one batch)

<!--
TODO: add updated low-level run commands. -->

<!--
```sh
./end-to-end.sh ['COMMON-CRAWL-DATES', ...] start_idx end_idx [wat]
``` -->

### Extract Wet Content

#### Extract Wet Content in Batches

```sh
./end-to-end.sh ['COMMON-CRAWL-DATES', ...] start_idx end_idx [wet] ../data/dqr/domain_pc1.csv content_table
```

### Extract Wet Domain-URLs in Batches

```sh
./end-to-end-url.sh ['COMMON-CRAWL-DATES', ...] start_idx end_idx [wet] ../data/dqr/domain_pc1.csv urls_table
```

### Optionally running in parts:

#### Download data

```sh
./get_data.sh ['COMMON-CRAWL-DATES', ...] start_idx  end_idx  [list of Warc file types i.e, wat,wet]
```

#### Convert compressed WAT files to the compressed format in wat_output_tables

```sh
./run_wat_to_link.sh ['COMMON-CRAWL-DATES', ...]
```

#### wat_output_tables for each respective Common-Crawl date is converted to a graph in the form of (edges.txt.gz, vertices.txt.gz)

```sh
./run_link_to_graph.sh ['COMMON-CRAWL-DATES', ...]
```

#### Extract domains content from WET files (DomainURL,..., content)

```sh
./run_extract_wet_content.sh ['COMMON-CRAWL-DATES', ...]
```
