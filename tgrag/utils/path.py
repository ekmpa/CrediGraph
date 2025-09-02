import os
from glob import glob
from pathlib import Path
from typing import Tuple


def get_root_dir() -> Path:
    return Path(__file__).parent.parent.parent


def get_curr_parent() -> Path:
    return Path(__file__).parent.parent


def get_no_backup() -> Path:
    return Path('/NOBACKUP')


def get_data_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    for up in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        probe = os.path.join(up, 'data', 'dqr', 'domain_ratings.csv')
        if os.path.exists(probe):
            return up
    return os.getcwd()


def get_scratch() -> Path:
    home = Path.home()
    scratch_dir = home.parent / home.name / 'scratch'
    return scratch_dir


def get_cwd() -> Path:
    return Path.cwd()


def get_crawl_data_path(project_dir: Path) -> str:
    return os.path.join(
        os.environ.get('SCRATCH', os.path.join(project_dir, 'data')), 'crawl-data'
    )


def get_wet_file_path(slice_id: str, project_dir: str) -> str:
    scratch = os.environ.get('SCRATCH', project_dir)
    base_path = os.path.join(scratch, 'crawl-data', slice_id, 'segments')
    segment_dirs = glob(os.path.join(base_path, '*', 'wet', '*.warc.wet.gz'))

    if not segment_dirs:
        raise FileNotFoundError(
            f'No WET file found for slice {slice_id} in {base_path}'
        )

    return segment_dirs[0]


def get_data_paths(slice: str, crawl_path: str) -> Tuple[str, str]:
    vertices_path = os.path.join(f'{crawl_path}/{slice}/output/', 'vertices.txt.gz')
    edges_path = os.path.join(f'{crawl_path}/{slice}/output/', 'edges.txt.gz')
    return vertices_path, edges_path
