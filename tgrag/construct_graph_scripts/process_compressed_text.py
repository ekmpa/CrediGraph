import glob
import gzip
import os
import shutil


def move_and_rename_compressed_outputs(source_base: str, target_base_root: str) -> None:
    os.makedirs(target_base_root, exist_ok=True)

    for subdir in ['edges', 'vertices']:
        source_dir = os.path.join(source_base, subdir)
        target_filename = f'{subdir}.txt.gz'

        matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))
        if not matches:
            print(f'[WARN] No .txt.gz file found in {source_dir}, skipping.')
            continue

        for source_file in matches:
            target_path = os.path.join(target_base_root, target_filename)

            if not os.path.exists(target_path):
                shutil.copy2(source_file, target_path)
                print(f'Copied {source_file} → {target_path}')
            else:
                print(f'Skipped: {target_path} already exists')
            break  # only process the first match


def move_and_append_compressed_outputs(source_base: str, target_base_root: str) -> None:
    """For each of ['edges', 'vertices']:
    - Find all .txt.gz files in the source subdir
    - Decompress and merge their contents
    - If the target exists, decompress it too and append
    - Write combined data back, compressed.
    """
    os.makedirs(target_base_root, exist_ok=True)

    for subdir in ['edges', 'vertices']:
        source_dir = os.path.join(source_base, subdir)
        target_filename = f'{subdir}.txt.gz'
        target_path = os.path.join(target_base_root, target_filename)

        matches = glob.glob(os.path.join(source_dir, '*.txt.gz'))
        if not matches:
            print(f'[WARN] No .txt.gz files found in {source_dir}, skipping.')
            continue

        combined_data = []

        # Add all new data from source files
        for source_file in matches:
            with gzip.open(source_file, 'rt', encoding='utf-8') as f:
                combined_data.extend(f.readlines())

        # Add existing target data if target exists (i.e continue build of previous runs)
        if os.path.exists(target_path):
            with gzip.open(target_path, 'rt', encoding='utf-8') as f:
                combined_data.extend(f.readlines())

        # Test: deduplicate lines
        # combined_data = list(set(combined_data))

        with gzip.open(target_path, 'wt', encoding='utf-8') as f:
            f.writelines(combined_data)

        print(f'[INFO] Aggregated {len(matches)} file(s) → {target_path}')
