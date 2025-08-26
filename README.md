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

### Running full data processing scripts

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
