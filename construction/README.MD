# Bash scripts for data processing

## Usage

### Build Custom Graph

```sh
./pipeline.sh file.txt [--keep idx]
```

where `file.txt` contains 1 crawl ID per line, and `idx` is an optional index on which to resume construction from.

#### (one batch)

```sh
./end-to-end.sh ['COMMON-CRAWL-DATES', ...] start_idx end_idx [wat]
```

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
