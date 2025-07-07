# scripts/encode_data.py

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from common import (DATA_DIR, map_hg, parse_age_band, calculate_14d_win_pct_from_history, 
                   calculate_win_percentage, parse_race_time)
from mappings import pattern_map, going_map, sex_map
from pathlib import Path


def calculate_historical_win_pct(historical_records):
    total_races = len(historical_records)
    if total_races == 0:
        return -1.0
    wins = sum(1 for record in historical_records if record['win'])
    return calculate_win_percentage(total_races, wins)

def save_mapping(mapping: dict, filename: str):
    mapping_path = DATA_DIR / 'mapping' / filename
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=4)

def load_or_create_mapping(filename: str, new_keys: list):
    mapping_path = DATA_DIR / 'mapping' / filename
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            existing_mapping = json.load(f)
    else:
        existing_mapping = {}
    
    max_id = max(existing_mapping.values()) if existing_mapping else -1
    
    for key in new_keys:
        if key not in existing_mapping:
            max_id += 1
            existing_mapping[key] = max_id
    
    return existing_mapping

RAW_DATA_DIR = DATA_DIR / 'training' / 'raw'
csv_files = list(RAW_DATA_DIR.rglob('*.csv'))

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df['datetime'] = merged_df.apply(
    lambda row: parse_race_time(row['date'], row['off']), axis=1
)
merged_df = merged_df.sort_values(['datetime', 'race_id']).reset_index(drop=True)

horse_history = defaultdict(list)
jockey_history = defaultdict(list)
trainer_history = defaultdict(list)

win_pct_columns = [
    'horse_total_runs', 'horse_total_wins', 'horse_win_pct',
    'horse_course_runs', 'horse_course_wins', 'horse_course_win_pct', 
    'horse_distance_runs', 'horse_distance_wins', 'horse_distance_win_pct', 
    'horse_going_runs', 'horse_going_wins', 'horse_going_win_pct',
    'jockey_total_runs', 'jockey_total_wins', 'jockey_win_pct',
    'jockey_course_runs', 'jockey_course_wins', 'jockey_course_win_pct',
    'jockey_distance_runs', 'jockey_distance_wins', 'jockey_distance_win_pct',
    'jockey_going_runs', 'jockey_going_wins', 'jockey_going_win_pct',
    'trainer_total_runs', 'trainer_total_wins', 'trainer_win_pct',
    'trainer_course_runs', 'trainer_course_wins', 'trainer_course_win_pct',
    'trainer_distance_runs', 'trainer_distance_wins', 'trainer_distance_win_pct',
    'trainer_going_runs', 'trainer_going_wins', 'trainer_going_win_pct',
    'jockey_14d_runs', 'jockey_14d_wins', 'jockey_14d_win_pct',
    'trainer_14d_runs', 'trainer_14d_wins', 'trainer_14d_win_pct',
    'jockey_14d_type_runs', 'jockey_14d_type_wins', 'jockey_14d_type_win_pct',
    'trainer_14d_type_runs', 'trainer_14d_type_wins', 'trainer_14d_type_win_pct'
]

for col in win_pct_columns:
    merged_df[col] = -1.0

for idx, row in merged_df.iterrows():
    horse_id = row['horse_id']
    jockey_id = row['jockey_id']
    trainer_id = row['trainer_id']
    course = row['course']
    going = row['going']
    dist_f = row['dist_f']
    race_type = row['type']
    win = (row['pos'] == '1')
    
    horse_records = horse_history[horse_id]
    jockey_records = jockey_history[jockey_id]
    trainer_records = trainer_history[trainer_id]
    
    # Horse total (lifetime) statistics
    horse_total_runs = len(horse_records)
    horse_total_wins = sum(1 for record in horse_records if record['win'])
    horse_win_pct = calculate_historical_win_pct(horse_records)
    merged_df.at[idx, 'horse_total_runs'] = horse_total_runs
    merged_df.at[idx, 'horse_total_wins'] = horse_total_wins
    merged_df.at[idx, 'horse_win_pct'] = horse_win_pct
    
    # Horse course-specific statistics
    course_records = [r for r in horse_records if r['course'] == course]
    horse_course_runs = len(course_records)
    horse_course_wins = sum(1 for record in course_records if record['win'])
    course_win_pct = calculate_historical_win_pct(course_records)
    merged_df.at[idx, 'horse_course_runs'] = horse_course_runs
    merged_df.at[idx, 'horse_course_wins'] = horse_course_wins
    merged_df.at[idx, 'horse_course_win_pct'] = course_win_pct
    
    # Horse distance-specific statistics
    distance_records = [r for r in horse_records if r['dist_f'] == dist_f]
    horse_distance_runs = len(distance_records)
    horse_distance_wins = sum(1 for record in distance_records if record['win'])
    distance_win_pct = calculate_historical_win_pct(distance_records)
    merged_df.at[idx, 'horse_distance_runs'] = horse_distance_runs
    merged_df.at[idx, 'horse_distance_wins'] = horse_distance_wins
    merged_df.at[idx, 'horse_distance_win_pct'] = distance_win_pct
    
    # Horse going-specific statistics
    going_records = [r for r in horse_records if r['going'] == going]
    horse_going_runs = len(going_records)
    horse_going_wins = sum(1 for record in going_records if record['win'])
    going_win_pct = calculate_historical_win_pct(going_records)
    merged_df.at[idx, 'horse_going_runs'] = horse_going_runs
    merged_df.at[idx, 'horse_going_wins'] = horse_going_wins
    merged_df.at[idx, 'horse_going_win_pct'] = going_win_pct
    
    # Jockey lifetime statistics
    historical_jockey_records = [r for r in jockey_records]
    jockey_total_runs = len(historical_jockey_records)
    jockey_total_wins = sum(1 for r in historical_jockey_records if r['win'])
    jockey_win_pct = calculate_historical_win_pct(historical_jockey_records)
    merged_df.at[idx, 'jockey_total_runs'] = jockey_total_runs
    merged_df.at[idx, 'jockey_total_wins'] = jockey_total_wins
    merged_df.at[idx, 'jockey_win_pct'] = jockey_win_pct
    
    # Jockey course-specific statistics
    jockey_course_records = [r for r in jockey_records if r['course'] == course]
    jockey_course_runs = len(jockey_course_records)
    jockey_course_wins = sum(1 for r in jockey_course_records if r['win'])
    jockey_course_win_pct = calculate_historical_win_pct(jockey_course_records)
    merged_df.at[idx, 'jockey_course_runs'] = jockey_course_runs
    merged_df.at[idx, 'jockey_course_wins'] = jockey_course_wins
    merged_df.at[idx, 'jockey_course_win_pct'] = jockey_course_win_pct
    
    # Jockey distance-specific statistics
    jockey_distance_records = [r for r in jockey_records if r['dist_f'] == dist_f]
    jockey_distance_runs = len(jockey_distance_records)
    jockey_distance_wins = sum(1 for r in jockey_distance_records if r['win'])
    jockey_distance_win_pct = calculate_historical_win_pct(jockey_distance_records)
    merged_df.at[idx, 'jockey_distance_runs'] = jockey_distance_runs
    merged_df.at[idx, 'jockey_distance_wins'] = jockey_distance_wins
    merged_df.at[idx, 'jockey_distance_win_pct'] = jockey_distance_win_pct
    
    # Jockey going-specific statistics
    jockey_going_records = [r for r in jockey_records if r['going'] == going]
    jockey_going_runs = len(jockey_going_records)
    jockey_going_wins = sum(1 for r in jockey_going_records if r['win'])
    jockey_going_win_pct = calculate_historical_win_pct(jockey_going_records)
    merged_df.at[idx, 'jockey_going_runs'] = jockey_going_runs
    merged_df.at[idx, 'jockey_going_wins'] = jockey_going_wins
    merged_df.at[idx, 'jockey_going_win_pct'] = jockey_going_win_pct
    
    # Trainer lifetime statistics
    historical_trainer_records = [r for r in trainer_records]
    trainer_total_runs = len(historical_trainer_records)
    trainer_total_wins = sum(1 for r in historical_trainer_records if r['win'])
    trainer_win_pct = calculate_historical_win_pct(historical_trainer_records)
    merged_df.at[idx, 'trainer_total_runs'] = trainer_total_runs
    merged_df.at[idx, 'trainer_total_wins'] = trainer_total_wins
    merged_df.at[idx, 'trainer_win_pct'] = trainer_win_pct
    
    # Trainer course-specific statistics
    trainer_course_records = [r for r in trainer_records if r['course'] == course]
    trainer_course_runs = len(trainer_course_records)
    trainer_course_wins = sum(1 for r in trainer_course_records if r['win'])
    trainer_course_win_pct = calculate_historical_win_pct(trainer_course_records)
    merged_df.at[idx, 'trainer_course_runs'] = trainer_course_runs
    merged_df.at[idx, 'trainer_course_wins'] = trainer_course_wins
    merged_df.at[idx, 'trainer_course_win_pct'] = trainer_course_win_pct
    
    # Trainer distance-specific statistics
    trainer_distance_records = [r for r in trainer_records if r['dist_f'] == dist_f]
    trainer_distance_runs = len(trainer_distance_records)
    trainer_distance_wins = sum(1 for r in trainer_distance_records if r['win'])
    trainer_distance_win_pct = calculate_historical_win_pct(trainer_distance_records)
    merged_df.at[idx, 'trainer_distance_runs'] = trainer_distance_runs
    merged_df.at[idx, 'trainer_distance_wins'] = trainer_distance_wins
    merged_df.at[idx, 'trainer_distance_win_pct'] = trainer_distance_win_pct
    
    # Trainer going-specific statistics
    trainer_going_records = [r for r in trainer_records if r['going'] == going]
    trainer_going_runs = len(trainer_going_records)
    trainer_going_wins = sum(1 for r in trainer_going_records if r['win'])
    trainer_going_win_pct = calculate_historical_win_pct(trainer_going_records)
    merged_df.at[idx, 'trainer_going_runs'] = trainer_going_runs
    merged_df.at[idx, 'trainer_going_wins'] = trainer_going_wins
    merged_df.at[idx, 'trainer_going_win_pct'] = trainer_going_win_pct
    
    jockey_14d_win_pct, jockey_14d_runs, jockey_14d_wins = calculate_14d_win_pct_from_history(jockey_records, row['datetime'])
    merged_df.at[idx, 'jockey_14d_runs'] = jockey_14d_runs
    merged_df.at[idx, 'jockey_14d_wins'] = jockey_14d_wins
    merged_df.at[idx, 'jockey_14d_win_pct'] = jockey_14d_win_pct
    
    trainer_14d_win_pct, trainer_14d_runs, trainer_14d_wins = calculate_14d_win_pct_from_history(trainer_records, row['datetime'])
    merged_df.at[idx, 'trainer_14d_runs'] = trainer_14d_runs
    merged_df.at[idx, 'trainer_14d_wins'] = trainer_14d_wins
    merged_df.at[idx, 'trainer_14d_win_pct'] = trainer_14d_win_pct
    
    jockey_14d_type_win_pct, jockey_14d_type_runs, jockey_14d_type_wins = calculate_14d_win_pct_from_history(jockey_records, row['datetime'], race_type)
    merged_df.at[idx, 'jockey_14d_type_runs'] = jockey_14d_type_runs
    merged_df.at[idx, 'jockey_14d_type_wins'] = jockey_14d_type_wins
    merged_df.at[idx, 'jockey_14d_type_win_pct'] = jockey_14d_type_win_pct
    
    trainer_14d_type_win_pct, trainer_14d_type_runs, trainer_14d_type_wins = calculate_14d_win_pct_from_history(trainer_records, row['datetime'], race_type)
    merged_df.at[idx, 'trainer_14d_type_runs'] = trainer_14d_type_runs
    merged_df.at[idx, 'trainer_14d_type_wins'] = trainer_14d_type_wins
    merged_df.at[idx, 'trainer_14d_type_win_pct'] = trainer_14d_type_win_pct
    
    race_record = {
        'course': course,
        'going': going, 
        'dist_f': dist_f,
        'win': win,
        'datetime': row['datetime']
    }
    
    horse_history[horse_id].append(race_record)
    
    jockey_record = {
        'win': win,
        'datetime': row['datetime'],
        'type': race_type,
        'course': course,
        'dist_f': dist_f,
        'going': going
    }
    jockey_history[jockey_id].append(jockey_record)
    
    trainer_record = {
        'win': win,
        'datetime': row['datetime'],
        'type': race_type,
        'course': course,
        'dist_f': dist_f,
        'going': going
    }
    trainer_history[trainer_id].append(trainer_record)

# date, region
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df['month'] = merged_df['date'].dt.month
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df.drop(columns=['date', 'month', 'region'], inplace=True, errors='ignore')

# course, course_id, dist, dist_f, dist_m
unique_tracks = merged_df.apply(lambda row: f"{row['course']}_{row['dist_f']}", axis=1).unique().tolist()
track_mapping = load_or_create_mapping('track_mapping.json', unique_tracks)
merged_df['track_id'] = merged_df.apply(lambda row: f"{row['course']}_{row['dist_f']}", axis=1).map(track_mapping).fillna(-1).astype(int)

save_mapping(track_mapping, 'track_mapping.json')

merged_df['dist_f'] = merged_df['dist_f'].str.rstrip('f').astype(float)

merged_df.drop(columns=['course', 'course_id', 'dist', 'dist_m'], inplace=True, errors='ignore')

# race_id, off, race_name
merged_df.drop(columns=['off', 'race_name'], inplace=True, errors='ignore')

# type, class, pattern
unique_types = merged_df['type'].unique().tolist()
type_mapping = load_or_create_mapping('type_mapping.json', unique_types)
merged_df['type_id'] = merged_df['type'].map(type_mapping).fillna(-1).astype(int)
save_mapping(type_mapping, 'type_mapping.json')

merged_df.drop(columns=['type'], inplace=True, errors='ignore')

merged_df['class'] = merged_df['class'].str.extract(r'(\d)').astype(int)

merged_df['pattern'] = merged_df['pattern'].map(pattern_map).fillna(5).astype(int)

# rating_band, age_band
merged_df['rating_max'] = merged_df['rating_band'].str.extract(r'-(\d+)').astype(float)
merged_df[['age_min', 'age_max']] = merged_df['age_band'].apply(parse_age_band).apply(pd.Series)

merged_df.drop(columns=['rating_band', 'age_band', 'sex_rest'], inplace=True, errors='ignore')

# going, run
merged_df['going'] = merged_df['going'].map(going_map)

# num, pos, draw
merged_df['win'] = (merged_df['pos'] == '1').astype(int)

merged_df.drop(columns=['num', 'pos'], inplace=True, errors='ignore')

# ovr_btn, btn
merged_df.drop(columns=['ovr_btn', 'btn'], inplace=True, errors='ignore')

# horse_id, horse, age, sex

merged_df['sex'] = merged_df['sex'].map(sex_map)

merged_df.drop(columns=['horse'], inplace=True, errors='ignore')

# lbs, hg
merged_df['hg'] = merged_df['hg'].apply(map_hg)

# time, secs
merged_df.drop(columns=['time', 'secs'], inplace=True, errors='ignore')


# jockey_id, jockey, trainer_id, trainer, owner_id, owner
owner_mapping = {}
for _, row in merged_df[['owner', 'owner_id']].dropna().iterrows():
    owner_name = row['owner']
    owner_id = int(row['owner_id'])
    if owner_name not in owner_mapping:
        owner_mapping[owner_name] = owner_id

save_mapping(owner_mapping, 'owner_mapping.json')

merged_df.drop(columns=['jockey', 'trainer', 'owner'], inplace=True, errors='ignore')

# prize
merged_df.drop(columns=['prize'], inplace=True, errors='ignore')

# or, rpr, ts
merged_df['or'] = pd.to_numeric(merged_df['or'], errors='coerce')
merged_df['rpr'] = pd.to_numeric(merged_df['rpr'], errors='coerce')
merged_df['ts'] = pd.to_numeric(merged_df['ts'], errors='coerce')

# sire_id, sire, dam_id, dam, damsire_id, damsire
horse_bloodlines = {}
for _, row in merged_df[['horse_id', 'sire_id', 'dam_id', 'damsire_id']].iterrows():
    horse_id = int(row['horse_id'])
    sire_id = int(row['sire_id']) if pd.notna(row['sire_id']) else -1
    dam_id = int(row['dam_id']) if pd.notna(row['dam_id']) else -1
    damsire_id = int(row['damsire_id']) if pd.notna(row['damsire_id']) else -1
    
    if horse_id not in horse_bloodlines:
        horse_bloodlines[horse_id] = {
            'sire_id': sire_id,
            'dam_id': dam_id, 
            'damsire_id': damsire_id
        }
    else:
        existing = horse_bloodlines[horse_id]
        if (existing['sire_id'] != sire_id or 
            existing['dam_id'] != dam_id or 
            existing['damsire_id'] != damsire_id):
            print(f"Warning: Bloodline conflict for horse_id {horse_id}")
            print(f"  Existing: sire={existing['sire_id']}, dam={existing['dam_id']}, damsire={existing['damsire_id']}")
            print(f"  New: sire={sire_id}, dam={dam_id}, damsire={damsire_id}")
            print(f"  Keeping existing bloodline data")

save_mapping(horse_bloodlines, 'horse_bloodlines.json')

merged_df.drop(columns=['sire', 'dam', 'damsire'], inplace=True, errors='ignore')

merged_df.drop(columns=['comment'], inplace=True, errors='ignore')

print(f"Final columns in encoded data: {merged_df.columns.tolist()}")
print(f"Helper columns for cleansing: datetime, pos")
print(f"Training will exclude: datetime, pos, and identifier columns")

output_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_file, index=False)
