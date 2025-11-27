import argparse
import faulthandler
import logging
import pickle
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from tgrag.utils.args import parse_args
from tgrag.utils.logger import setup_logging
from tgrag.utils.path import discover_subfolders, get_root_dir, get_scratch
from tgrag.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Aggregate domain-to-ID mapping and rewrite all vertices/edges.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/tgl/base.yaml',
    help='Path to yaml configuration file',
)


def build_global_mapping(
    subfolders: List[Path], chunk_size: int = 1_000_000
) -> Dict[str, int]:
    """Scan all vertices.csv files and build one global domain→id mapping."""
    logging.info('Building global domain-to-id mapping...')

    domain_to_id = {}
    next_id = 0

    for folder in subfolders:
        node_csv = folder / 'vertices.csv'
        logging.info(f'Scanning domains in: {node_csv}')

        for chunk in tqdm(
            pd.read_csv(node_csv, chunksize=chunk_size),
            desc=f'Scanning {folder.name}',
            unit='chunk',
        ):
            for domain in chunk['domain'].astype(str):
                if domain not in domain_to_id:
                    domain_to_id[domain] = next_id
                    next_id += 1

    logging.info(f'Total unique domains: {len(domain_to_id):,}')
    return domain_to_id


def aggregate_rewrite(
    subfolders: List[Path],
    domain_to_id: Dict[str, int],
    aggregate_out: Path,
    chunk_size: int = 1_000_000,
) -> None:
    """Rewrite and append all subfolder vertices/edges into global outputs."""
    aggregate_out.mkdir(parents=True, exist_ok=True)

    out_nodes = aggregate_out / 'vertices_with_id.csv'
    out_edges = aggregate_out / 'edges_with_id.csv'

    with open(out_nodes, 'w') as f:
        f.write('id,ts\n')

    with open(out_edges, 'w') as f:
        f.write('src_id,dst_id,ts\n')

    for folder in subfolders:
        logging.info(f'Aggregating subfolder: {folder}')

        nodes_csv = folder / 'vertices.csv'
        edges_csv = folder / 'edges.csv'

        with open(out_nodes, 'a') as fout:
            for chunk in tqdm(
                pd.read_csv(nodes_csv, chunksize=chunk_size),
                desc=f'Vertices {folder.name}',
                unit='chunk',
            ):
                chunk['id'] = chunk['domain'].map(domain_to_id)
                chunk[['id', 'ts']].astype({'id': 'int64'}).to_csv(
                    fout, header=False, index=False
                )

        with open(out_edges, 'a') as fout:
            for chunk in tqdm(
                pd.read_csv(edges_csv, chunksize=chunk_size),
                desc=f'Edges {folder.name}',
                unit='chunk',
            ):
                chunk['src_id'] = chunk['src'].map(domain_to_id)
                chunk['dst_id'] = chunk['dst'].map(domain_to_id)
                chunk[['src_id', 'dst_id', 'ts']].astype(
                    {'src_id': 'int64', 'dst_id': 'int64'}
                ).to_csv(fout, header=False, index=False)

    logging.info(f'Aggregate outputs complete at: {aggregate_out}')


def main() -> None:
    faulthandler.enable()

    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()

    config_file_path = root / args.config_file
    meta_args, _ = parse_args(config_file_path)

    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    base_dir = scratch / cast(str, meta_args.database_folder)
    aggregate_out = base_dir / 'aggregate'
    aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f'Scanning base directory: {base_dir}')
    subfolders = discover_subfolders(base_dir)

    if not subfolders:
        raise RuntimeError(f'No valid subfolders found in {base_dir}')

    domain_to_id = build_global_mapping(subfolders)

    with open(aggregate_out / 'global_domain_to_id.pkl', 'wb') as f:
        pickle.dump(domain_to_id, f)

    np.save(
        aggregate_out / 'global_domain_ids.npy',
        np.arange(len(domain_to_id), dtype=np.int64),
    )

    aggregate_rewrite(subfolders, domain_to_id, aggregate_out)

    logging.info('Completed.')


if __name__ == '__main__':
    main()
