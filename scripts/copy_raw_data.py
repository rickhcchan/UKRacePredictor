# scripts/copy_raw_data.py

import shutil
from common import DATA_DIR, RPSCRAPE_DIR
from pathlib import Path

def copy_race_data():
    SOURCE_BASE_DIR = RPSCRAPE_DIR / 'data' / 'regions' / 'gb'
    DEST_BASE_DIR = DATA_DIR / 'training' / 'raw'
    race_types = ['flat', 'jumps']
    for race_type in race_types:
        source_dir = SOURCE_BASE_DIR / race_type
        dest_dir = DEST_BASE_DIR / race_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        files_copied = 0
        for csv_file in source_dir.glob('*.csv'):
            shutil.copy2(csv_file, dest_dir / csv_file.name)
            files_copied += 1
        print(f'Copied {files_copied} files from {source_dir} to {dest_dir}')

if __name__ == '__main__':
    copy_race_data()
