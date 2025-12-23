
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <graph_path> <slice> <min_deg> [mem]"
    echo "Example: $0 /data/crawl-data/CC-MAIN-2025-08/output1 3 [60%]"
    exit 1
fi

GRAPH_PATH="$1"
SLICE="$2"
MIN_DEG="$3"
MEM="${4:-60%}"   

uv run python - <<EOF
from pathlib import Path
from tgrag.construct_graph_scripts.process import process_graph  

graph_path = "${GRAPH_PATH}"
slice_str = "${SLICE}"
min_deg = int("${MIN_DEG}")
mem = "${MEM}"

process_graph(graph_path, slice_str, min_deg, mem)
EOF