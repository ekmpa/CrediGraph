import glob
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
