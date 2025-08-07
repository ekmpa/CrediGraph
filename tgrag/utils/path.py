import os
from glob import glob
from pathlib import Path


def get_root_dir() -> Path:
    return Path(__file__).parent.parent.parent


def get_no_backup() -> Path:
    return Path('/NOBACKUP')


def get_scatch() -> Path:
    return Path('scratch/')


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
