# scripts/encode_data.py

import json
import numpy as np
import pandas as pd

from common import DATA_DIR, map_hg, parse_age_band
from mappings import pattern_map, going_map, sex_rest_map, sex_map
from pathlib import Path

def save_mapping(mapping: dict, filename: str):
    mapping_path = DATA_DIR / 'mapping' / filename
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)

RAW_DATA_DIR = DATA_DIR / 'training' / 'raw'
csv_files = list(RAW_DATA_DIR.rglob('*.csv'))

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

# date, region
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df['month'] = merged_df['date'].dt.month
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df.drop(columns=['date', 'month', 'region'], inplace=True)

# course, course_id, dist, dist_f, dist_m
merged_df['track_id'] = merged_df['course'] + '_' + merged_df['dist_f'].astype(str)
unique_tracks = merged_df['track_id'].unique()
track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
merged_df['track_id'] = merged_df['track_id'].map(track_to_id).astype(int)

save_mapping(track_to_id, 'track_mapping.json')

merged_df['dist_f'] = merged_df['dist_f'].str.rstrip('f').astype(float)

merged_df.drop(columns=['course', 'course_id', 'dist', 'dist_m'], inplace=True)

# race_id, off, race_name
merged_df.drop(columns=['off', 'race_name'], inplace=True)

# type, class, pattern
unique_types = merged_df['type'].unique()
type_to_id = {type: idx for idx, type in enumerate(unique_types)}
merged_df['type'] = merged_df['type'].map(type_to_id).astype(int)

save_mapping(type_to_id, 'type_mapping.json')

merged_df['class'] = merged_df['class'].str.extract(r'(\d)').astype(int)

merged_df['pattern'] = merged_df['pattern'].map(pattern_map).fillna(5).astype(int)

# rating_band, age_band, sex_rest
merged_df['rating_max'] = merged_df['rating_band'].str.extract(r'-(\d+)').astype(float)
merged_df[['age_min', 'age_max']] = merged_df['age_band'].apply(parse_age_band).apply(pd.Series)

merged_df['sex_rest'] = merged_df['sex_rest'].map(sex_rest_map).fillna(-1).astype(int)

merged_df.drop(columns=['rating_band', 'age_band', 'sex_rest'], inplace=True)

# going, run
merged_df['going'] = merged_df['going'].map(going_map)

# num, pos, draw
merged_df['win'] = (merged_df['pos'] == '1').astype(int)

merged_df.drop(columns=['num', 'pos'], inplace=True)

# ovr_btn, btn
merged_df.drop(columns=['ovr_btn', 'btn'], inplace=True)

# horse_id, horse, age, sex

merged_df['sex'] = merged_df['sex'].map(sex_map)

merged_df.drop(columns=['horse'], inplace=True)

# lbs, hg
merged_df['hg'] = merged_df['hg'].apply(map_hg)

# time, secs
merged_df.drop(columns=['time', 'secs'], inplace=True)

# dec
merged_df.drop(columns=['dec'], inplace=True)

# jockey_id, jockey, trainer_id, trainer, owner_id, owner
owner_to_id = {}
for _, row in merged_df[['owner', 'owner_id']].dropna().iterrows():
    owner_name = row['owner']
    owner_id = int(row['owner_id'])
    if owner_name not in owner_to_id:
        owner_to_id[owner_name] = owner_id

save_mapping(owner_to_id, 'owner_mapping.json')

merged_df.drop(columns=['jockey', 'trainer', 'owner'], inplace=True)

# prize
merged_df.drop(columns=['prize'], inplace=True)

# or, rpr, ts
merged_df['or'] = pd.to_numeric(merged_df['or'], errors='coerce')
merged_df['rpr'] = pd.to_numeric(merged_df['rpr'], errors='coerce')
merged_df['ts'] = pd.to_numeric(merged_df['ts'], errors='coerce')

# sire_id, sire, dam_id, dam, damsire_id, damsire
unique_sires = merged_df['sire'].unique()
sire_to_id = {sire: idx for idx, sire in enumerate(unique_sires)}
merged_df['sire'] = merged_df['sire'].map(sire_to_id).fillna(-1).astype(int)

save_mapping(sire_to_id, 'sire_mapping.json')

unique_dams = merged_df['dam'].unique()
dam_to_id = {dam: idx for idx, dam in enumerate(unique_dams)}
merged_df['dam'] = merged_df['dam'].map(dam_to_id).fillna(-1).astype(int)

save_mapping(dam_to_id, 'dam_mapping.json')

unique_damsires = merged_df['damsire'].unique()
damsire_to_id = {ds: idx for idx, ds in enumerate(unique_damsires)}
merged_df['damsire'] = merged_df['damsire'].map(damsire_to_id).fillna(-1).astype(int)

save_mapping(damsire_to_id, 'damsire_mapping.json')

merged_df.drop(columns=['sire_id', 'dam_id', 'damsire_id'], inplace=True)

# comment
merged_df.drop(columns=['comment'], inplace=True)

# Save merged + encoded
output_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_file, index=False)
