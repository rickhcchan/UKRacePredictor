# scripts/copy_raw_data.py

import shutil
from common import DATA_DIR, RPSCRAPE_DIR
from pathlib import Path

SOURCE_BASE_DIR = RPSCRAPE_DIR / 'data' / 'gb'
DEST_BASE_DIR = DATA_DIR / 'training' / 'raw'
race_types = ['flat', 'jumps']

for race_type in race_types:
    source_dir = SOURCE_BASE_DIR / race_type
    dest_dir = DEST_BASE_DIR / race_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        continue
        
    files_copied = 0
    for csv_file in source_dir.glob('*.csv'):
        shutil.copy2(csv_file, dest_dir / csv_file.name)
        files_copied += 1
