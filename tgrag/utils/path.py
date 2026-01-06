import os
from glob import glob
from pathlib import Path
from typing import Tuple


def get_root_dir() -> Path:
    """Return the project root directory.

    Defined as three levels above this file.
    """
    return Path(__file__).parent.parent.parent


def get_curr_parent() -> Path:
    """Return the parent directory two levels above this file."""
    return Path(__file__).parent.parent


def get_no_backup() -> Path:
    """Return the path to the system no-backup directory."""
    return Path('/NOBACKUP')


def get_data_root() -> str:
    """Return the nearest ancestor directory containing the DQR data file.

    Returns:
        Str
            The first directory for which that path exists, otherwise the current
            working directory.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    for up in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        probe = os.path.join(up, 'data', 'dqr', 'domain_ratings.csv')
        if os.path.exists(probe):
            return up
    return os.getcwd()


def get_scratch() -> Path:
    """Return the user's scratch directory."""
    home = Path.home()
    scratch_dir = home.parent / home.name / 'scratch'
    return scratch_dir


def get_cwd() -> Path:
    """Return the current working directory."""
    return Path.cwd()


def get_crawl_data_path(project_dir: Path) -> str:
    """Return the crawl-data directory path, if possible in scratch.

    Parameters:
        project_dir : Path
            Project root directory.

    Returns:
        str
            Path to the crawl-data directory.
    """
    return os.path.join(
        os.environ.get('SCRATCH', os.path.join(project_dir, 'data')), 'crawl-data'
    )


def get_wet_file_path(slice_id: str, project_dir: str) -> str:
    """Return the first WET file path for a given crawl slice.

    Parameters:
        slice_id : str
            Crawl slice identifier.
        project_dir : str
            Project root directory used if SCRATCH is not set.

    Returns:
        str
            Path to the first matching WET file.

    Raises:
        FileNotFoundError
            If no WET file is found for the given slice.
    """
    scratch = os.environ.get('SCRATCH', project_dir)
    base_path = os.path.join(scratch, 'crawl-data', slice_id, 'segments')
    segment_dirs = glob(os.path.join(base_path, '*', 'wet', '*.warc.wet.gz'))

    if not segment_dirs:
        raise FileNotFoundError(
            f'No WET file found for slice {slice_id} in {base_path}'
        )

    return segment_dirs[0]


def get_data_paths(slice: str, crawl_path: str) -> Tuple[str, str]:
    """Return the vertices and edges file paths for a given crawl slice.

    Parameters:
        slice : str
            Crawl slice identifier.
        crawl_path : str
            Root crawl-data directory.

    Returns:
        Tuple[str, str]
            (vertices_path, edges_path)
    """
    vertices_path = os.path.join(f'{crawl_path}/{slice}/output/', 'vertices.txt.gz')
    edges_path = os.path.join(f'{crawl_path}/{slice}/output/', 'edges.txt.gz')
    return vertices_path, edges_path
