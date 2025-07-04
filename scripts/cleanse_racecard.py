# scripts/cleanse_racecard.py

import json
import pandas as pd
import numpy as np
import re

from common import map_hg
from mappings import pattern_map, going_map, sex_map
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'processed'
RACECARD_DIR = SCRIPT_DIR.parent / 'data' / 'racecard'

with open(DATA_DIR / 'type_mapping.json') as f:
    type_mapping = json.load(f)

racecard_file = RACECARD_DIR / '2025-06-28.json'
with open(racecard_file, encoding='utf-8') as f:
    racecard_data = json.load(f)

rows = []

courses = racecard_data['GB']
for course_name, races in courses.items():
    for race_time, race_info in races.items():
        class_match = re.search(r'(\d)', race_info.get('race_class', ''))
        class_id = int(class_match.group(1)) if class_match else -1
        
        runners = race_info.get("runners", [])
        for runner in runners:

            row = {
                'race_id': int(race_info.get('race_id')),
                'type': type_mapping.get(race_info['type'], -1),
                'class': class_id,
                'pattern': pattern_map.get(race_info.get('pattern'), 5),
                'dist_f': race_info.get('distance_f'),
                'going': going_map.get(race_info.get('going'), 3),
                'ran': len(race_info.get("runners", [])),
                'draw': int(race_info.get('draw')),
                'horse_id' : int(runner.get('horse_id', -1)),
                'age': int(runner.get('age')),
                'sex': sex_map.get(runner.get('sex_code'), -1),
                'hg': map_hg(runner.get('headgear')),
                'jockey_id': int(runner.get('jockey_id', -1)),
                'trainer_id': int(runner.get('trainer_id', -1)),
                'or': int(runner.get('ofr')),
                'rpr': int(runner.get('rpr')),
                'ts': int(runner.get('ts')),

            }
            rows.append(row)

df = pd.DataFrame(rows)

# distance in meters
df['dist_m'] = df['dist_f'] * 201.168

# cyclical month encoding
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# reconstruct track_id (course + distance_f)
df['track'] = df['course'] + '_' + df['dist_f'].astype(str)
# you will need to load your existing track mapping
track_mapping_path = Path('../data/processed/track_mapping.json')
if track_mapping_path.exists():
    import json
    with open(track_mapping_path) as f:
        track_map = json.load(f)
    df['track_id'] = df['track'].map(track_map).fillna(-1).astype(int)
else:
    df['track_id'] = -1

# going encoding: same logic as training
going_map = {
    'Firm': 1,
    'Good To Firm': 2,
    'Good': 3,
    'Good To Soft': 4,
    'Soft': 5,
    'Heavy': 6,
    'Standard': 3,
    'Standard To Slow': 4
}
df['going_encoded'] = df['going'].map(going_map).fillna(-1).astype(int)

# other features (rating_max, age_min, etc)
df['rating_max'] = 0
df['age_min'] = df['age']  # assume single
df['age_max'] = df['age']  # assume single
df['win'] = 0  # placeholder

# final columns reorder
final_cols = [
    'race_id','type','class','pattern','sex_rest','dist_m','going','ran','draw','ovr_btn',
    'horse_id','age','sex','lbs','hg','dec','jockey_id','trainer_id','or','rpr','ts',
    'sire_id','dam_id','damsire_id','owner_id','month_sin','month_cos','track_id',
    'rating_max','age_min','age_max','going_encoded','win'
]

df = df[final_cols]

# save the cleansed
output_file = Path('../data/predict/cleaned_racecard.csv')
output_file.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(output_file, index=False)

print(f'Saved cleansed racecard data to {output_file}')
