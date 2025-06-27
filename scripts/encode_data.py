import json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"

top_hg = ['p', 't', 'b', 'h', 'v']
hg_map = {k: i+1 for i, k in enumerate(top_hg)}
hg_map[np.nan] = 0

def map_hg(val):
    if pd.isna(val):
        return 0
    elif val in hg_map:
        return hg_map[val]
    else:
        return len(top_hg) + 1

def save_mapping(mapping: dict, filename: str):
    mapping_path = SCRIPT_DIR.parent / "data" / "processed" / filename
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)

def parse_age_band(age_band):
    if pd.isna(age_band):
        return np.nan, np.nan
    age_band = age_band.replace('yo', '')
    if '+' in age_band:
        min_age = int(age_band.replace('+', ''))
        max_age = np.nan
    elif '-' in age_band:
        parts = age_band.split('-')
        min_age = int(parts[0])
        max_age = int(parts[1])
    else:
        min_age = max_age = int(age_band)
    return min_age, max_age



csv_files = list(DATA_DIR.rglob("*.csv"))

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
merged_df['track_id'] = merged_df['course'] + '_' + merged_df['dist'].astype(str)
unique_tracks = merged_df['track_id'].unique()
track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
merged_df['track_id'] = merged_df['track_id'].map(track_to_id).astype('int')

save_mapping(track_to_id, "track_mapping.json")

merged_df.drop(columns=['course', 'course_id', 'dist', 'dist_m'], inplace=True)

# race_id, off, race_name
merged_df.drop(columns=['off', 'race_name'], inplace=True)

# type, class, pattern
unique_types = merged_df['type'].unique()
type_to_id = {type: idx for idx, type in enumerate(unique_types)}
merged_df['type'] = merged_df['type'].map(type_to_id).astype('int')

save_mapping(type_to_id, "type_mapping.json")

merged_df['class'] = merged_df['class'].str.extract(r'(\d)').astype(int)

pattern_map = {
    'Group 1': 1,
    'Group 2': 2,
    'Group 3': 3,
    'Listed': 4,
    np.nan: 5
}
merged_df['pattern'] = merged_df['pattern'].map(pattern_map)

# rating_band, age_band, sex_rest
merged_df['rating_max'] = merged_df['rating_band'].str.extract(r'-(\d+)').astype(float)
merged_df[['age_min', 'age_max']] = merged_df['age_band'].apply(parse_age_band).apply(pd.Series)

sex_rest_map = {
    'F': 1,
    'M': 2,
    'C & G': 3,
    'F & M': 4
}
merged_df['sex_rest'] = merged_df['sex_rest'].map(sex_rest_map).fillna(0).astype(int)

merged_df.drop(columns=['rating_band', 'age_band'], inplace=True)

# going, run
going_map = {
    'Firm': 1,
    'Good To Firm': 2,
    'Good': 3,
    'Good To Soft': 4,
    'Standard': 3,
    'Standard To Slow': 4,
    'Soft': 5
}
merged_df['going_encoded'] = merged_df['going'].map(going_map)

# num, pos, draw
merged_df['win'] = (merged_df['pos'] == '1').astype(int)

merged_df.drop(columns=['num', 'pos'], inplace=True)

# ovr_btn, btn
merged_df.drop(columns=['btn'], inplace=True)

# horse_id, horse, age, sex
sex_map = {
    'G': 1,
    'F': 2,
    'M': 3,
    'C': 4,
    'H': 5
}
merged_df['sex'] = merged_df['sex'].map(sex_map)

merged_df.drop(columns=['horse'], inplace=True)

# lbs, hg
merged_df['hg'] = merged_df['hg'].apply(map_hg)

# time, secs
merged_df.drop(columns=['time', 'secs'], inplace=True)

# dec

# jockey_id, jockey, trainer_id, trainer, owner_id, owner
merged_df.drop(columns=['jockey', 'trainer', 'owner'], inplace=True)

# prize
merged_df.drop(columns=['prize'], inplace=True)

# or, rpr, ts

# sire_id, sire, dam_id, dam, damsire_id, damsire
merged_df.drop(columns=['sire', 'dam', 'damsire'], inplace=True)

# comment
merged_df.drop(columns=['comment'], inplace=True)

# Save merged + encoded
output_file = SCRIPT_DIR.parent / "data" / "processed" / "all_merged_encoded.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_file, index=False)
