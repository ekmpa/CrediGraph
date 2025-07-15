#Bash scripts for data processing

## Usage

### Running full data processing scripts

The pipeline script runs the data loading pipeline repeatedly to gradually build a slice's on a slice, i.e runs on the entirety of the slice's WAT file links:

```sh
./pipeline.sh COMMON-CRAWL-DATES # where COMMON-CRAWL-DATES is a txt file with slice indices
```

The end-to-end script has the full loading pipeline, and runs it for a selection of WAT file links (this is necessary for heap space management):

```sh
./end-to-end.sh [EITHER: 'COMMON-CRAWL-DATES' OR 'CC-MAIN-ONE-DATE'] start_idx end_idx [list of Warc file types i.e, wat,wet]
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
