# CrediGraph

<img src="img/logo_silver.png" alt="CrediGraph Logo" style="width: 300px; height: auto;" />

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip`, you can invoke:

```sh
pip install uv
# or
brew install uv
```

### Installation

```sh
# Clone the repo
git@github.com:ekmpa/CrediGraph.git

# Enter the repo directory
cd CrediGraph

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv
source .venv/bin/activate
```

## Usage

### Running full graph data processing scripts

The `pipeline.sh` script iteratively runs the pipeline, with optional `--keep idx` to resume construction, skipping the first `idx` files:

```sh
cd bash_scripts

bash pipeline.sh CC-Crawls/CC-2024-nov.txt # --keep idx
```

where `CC-2024-nov.txt` is a `.txt` file with the slice names, e.g one `CC-MAIN-YYYY-WW` per line.

This will construct the graph in `$SCRATCH/crawl-data/CC-MAIN-YYYY-WW/output`.

#### Processing

For processing a graph, run the `main` script in `tgrag/construct_graph_scripts`, with the slice name and the degree threshold `k`.

```sh
cd ..

uv run python tgrag/construct_graph_scripts/main.py \
    --slices CC-MAIN-YYYY-WW \
    --min-deg k \
```

This will create a `processed-degk/` folder under the slice's `output/`.


### Running domain's content extraction 

The `end-to-end.sh` script runs a batch of content extraction from start_idx to end_idx

```sh
cd bash_scripts
bash end-to-end.sh CC-Crawls/Dec2024.txt <start_idx> <end_idx> [wet] <seed_list> <spark_table_name>
```
where 
- `Dec2024.txt` is a `.txt` file with the slice names, e.g one `CC-MAIN-YYYY-WW` per line.
- start_idx: the start index inclusive to process out of 90K wet files
- end_idx: the last index inclusive to process out of 90K wet files
- seed_list: the seed list of domains to extract thier content i.e, the dqr domains list at 'data/dqr/domain_pc1.csv'
- spark_table_name: the spark table name and output folder naming pattern i.e, content_table

This will generate parquet files per batch under `$SCRATCH/spark-warehouse/<spark_table_name>_batch_ccmain202451_<start_idx>_<end_idx>

#### Merging the extracted Content

Loop across the generated parquet files to collect text contents and merge them per domain\
simply use:  ```pandas.read_parquet(file_path, engine='pyarrow')```\
Each parquet file contains columns:
- Domain_Name: the domain name	
- WARC_Target_URI: the web page URI
- WARC_Identified_Content_Language: list of CC-identified content languagues	
- WARC_Date: the content scrap date	
- Content_Type: the content type i.e., text,csv,json
- Content_Length: the content length in bytes	
- wet_record_txt: the UTF-8 text content

### Running MLP Experiments
The required embedding files and dataset must be placed under the data directory. The expected file paths are as follows:
- Domain text embeddings: data/dqr/labeled_11k_scraped_text_emb.pkl
- Domain name embeddings: data/dqr/labeled_11k_domainname_emb.pkl
- DQR dataset (ratings): data/dqr/domain_ratings.csv
```sh
uv run python tgrag/experiments/mlp_experiments/main.py --target pc1 --embed_type text
```

### Running GNN Baseline Experiment

Given the size of our datasets we must leverage mini-batching in our GNN experiments. To do this we use PyG's neighbor_loader,
which requires additional libraries having undocumented build-time dependencies. As such, users are required to install them in their
own venv. seperate from `uv sync`.

pyg-lib:

```
uv pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

PyTorch Sparse:

```
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
```

For information on installations of these additional libraries see [pyg-lib](https://github.com/pyg-team/pyg-lib) and [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse).

To run our baseline static experimentation:

```sh
uv run tgrag/experiments/main.py
```

Alternatively, you can design you own configuration, updating the model paramaters:

```sh
uv run tgrag/experiments/main.py --config configs/your_config.yaml
```

To learn more about making a contribution to CrediGraph see our [contribution guide](./.github/CONTRIBUTION.md)
