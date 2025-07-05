# scripts/cleanse_racecard.py

import json
import pandas as pd
import numpy as np
import re
import shutil
from datetime import datetime

from common import DATA_DIR, RPSCRAPE_DIR, map_hg, parse_age_band
from mappings import pattern_map, going_map, sex_map
from pathlib import Path

RACECARD_SOURCE_DIR = RPSCRAPE_DIR / 'racecards'
RACECARD_DEST_DIR = DATA_DIR / 'prediction' / 'raw'

RACECARD_DEST_DIR.mkdir(parents=True, exist_ok=True)

today = datetime.now().strftime('%Y-%m-%d')
source_file = RACECARD_SOURCE_DIR / f'{today}.json'
racecard_file = RACECARD_DEST_DIR / f'{today}.json'

if source_file.exists():
    shutil.copy2(source_file, racecard_file)
else:
    raise FileNotFoundError(f'Racecard file not found: {source_file}')

with open(DATA_DIR / 'mapping' / 'type_mapping.json') as f:
    type_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'sire_mapping.json') as f:
    sire_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'dam_mapping.json') as f:
    dam_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'damsire_mapping.json') as f:
    damsire_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'owner_mapping.json') as f:
    owner_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'track_mapping.json') as f:
    track_mapping = json.load(f)

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

            sire_name = runner.get('sire', '')
            sire_region = runner.get('sire_region', '')
            if sire_name and sire_region:
                sire_full = f"{sire_name} ({sire_region})"
            else:
                sire_full = sire_name
            
            dam_name = runner.get('dam', '')
            dam_region = runner.get('dam_region', '')
            if dam_name and dam_region:
                dam_full = f"{dam_name} ({dam_region})"
            else:
                dam_full = dam_name
            
            damsire_name = runner.get('damsire', '')
            damsire_region = runner.get('damsire_region', '')
            if damsire_name and damsire_region:
                damsire_full = f"{damsire_name} ({damsire_region})"
            else:
                damsire_full = damsire_name

            # Safe mapping lookup: try with region first, then without region, finally -1
            def safe_mapping_lookup(mapping, name_with_region, name_only):
                if name_with_region in mapping:
                    return mapping[name_with_region]
                elif name_only in mapping:
                    return mapping[name_only]
                else:
                    return -1

            row = {
                'race_id': int(race_info.get('race_id')),
                'type': type_mapping.get(race_info['type'], -1),
                'class': class_id,
                'pattern': pattern_map.get(race_info.get('pattern'), 5),
                'dist_f': race_info.get('distance_f'),
                'going': going_map.get(race_info.get('going'), 3),
                'ran': len(race_info.get("runners", [])),
                'draw': int(runner.get('draw', -1)),
                'horse_id': int(runner.get('horse_id', -1)),
                'age': int(runner.get('age')),
                'sex': sex_map.get(runner.get('sex_code'), -1),
                'lbs': int(runner.get('lbs', 0)),
                'hg': map_hg(runner.get('headgear')),
                'jockey_id': int(runner.get('jockey_id', -1)),
                'trainer_id': int(runner.get('trainer_id', -1)),
                'or': int(runner.get('ofr', -1)) if runner.get('ofr') is not None else -1,
                'rpr': int(runner.get('rpr', -1)) if runner.get('rpr') is not None else -1,
                'ts': int(runner.get('ts', -1)) if runner.get('ts') is not None else -1,
                'sire': safe_mapping_lookup(sire_mapping, sire_full, sire_name),
                'dam': safe_mapping_lookup(dam_mapping, dam_full, dam_name),
                'damsire': safe_mapping_lookup(damsire_mapping, damsire_full, damsire_name),
                'owner_id': owner_mapping.get(runner.get('owner', '').replace(' & ', ' ').replace('-', ' '), -1),
                'rating_band': race_info.get('rating_band', ''),
                'age_band': race_info.get('age_band', ''),
                'course': course_name,
                'date': today,

            }
            rows.append(row)

df = pd.DataFrame(rows)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['track_id'] = (df['course'] + '_' + df['dist_f'].apply(lambda x: str(int(x)) if x == int(x) else str(x)) + 'f').map(track_mapping).fillna(-1).astype(int)

df['rating_max'] = df['rating_band'].str.extract(r'-(\d+)').astype(float)

df[['age_min', 'age_max']] = df['age_band'].apply(parse_age_band).apply(pd.Series)

df.drop(columns=['course', 'date', 'month', 'rating_band', 'age_band'], inplace=True, errors='ignore')

training_cols = [
    'race_id', 'type', 'class', 'pattern', 'dist_f', 'going', 'ran', 'draw',
    'horse_id', 'age', 'sex', 'lbs', 'hg', 'jockey_id', 'trainer_id', 
    'or', 'rpr', 'ts', 'sire', 'dam', 'damsire', 'owner_id',
    'month_sin', 'month_cos', 'track_id', 'rating_max', 'age_min', 'age_max'
]

final_cols = [col for col in training_cols if col in df.columns]
df = df[final_cols]

output_file = DATA_DIR / 'prediction' / 'processed' / 'cleaned_racecard.csv'
output_file.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(output_file, index=False)

print(f'Saved cleansed racecard data to {output_file}')
print(f'Final columns: {list(df.columns)}')
