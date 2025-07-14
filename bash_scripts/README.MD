#Bash scripts for data processing

## Usage

### Running full data processing scripts

```sh
./end-to-end.sh ['COMMON-CRAWL-DATES', ...] start_idx end_idx [list of Warc file types i.e, wat,wet]
```

##Optionally running in parts:

### Get the data

```sh
./get_data.sh ['COMMON-CRAWL-DATES', ...] start_idx  end_idx  [list of Warc file types i.e, wat,wet]
```

### Convert compressed WAT files to the compressed format in wat_output_tables

```sh
./run_wat_to_link.sh ['COMMON-CRAWL-DATES', ...]
```

### wat_output_tables for each respective Common-Crawl date is converted to a graph in the form of (edges.txt.gz, vertices.txt.gz)

```sh
./run_link_to_graph.sh ['COMMON-CRAWL-DATES', ...]
```

### Extract domains content from WET files (DomainURL,..., content) 

```sh
./run_extract_wet_content.sh ['COMMON-CRAWL-DATES', ...]
```
