# Bash scripts for data processing

## Usage

### Building Temporal Graphs

```sh
bash pipeline.sh <start_month> <end_month> <number of subfolders>
# e.g,
bash pipeline.sh 'January 2020' 'February 2020' 3
```

where the format `<month name YYYY>` must be respected, and the number of subfolders designated the extent of parallelization of the graph construction. Note that not parallelizing the construction means a graph takes up to 600 hours to be processed and produced, whereas high level of parallelization can take this down to ... ?

> ***Note:***
>
> At times out of our control, the Common Crawl connection may be down. The CrediBench pipeline has safeguards against this (retries and wait periods), but in some cases, the server is just offline for a short amount of time. In the pipeline, this will appear most often as a `RuntimeError` in the high-level pipeline (before going into data download) that indicates the fetch from `https://index.commoncrawl.org/collinfo.json` failed.

#### (one batch)

TODO: add updated low-level run commands.

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
