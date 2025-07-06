# scripts/encode_data.py

import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from common import DATA_DIR, map_hg, parse_age_band
from mappings import pattern_map, going_map, sex_map
from pathlib import Path

def parse_race_time(date_str, time_str):
    try:
        date_part = datetime.strptime(date_str, '%Y-%m-%d').date()
        time_part = datetime.strptime(time_str, '%H:%M').time()
        hour = time_part.hour
        if hour in [11, 12]:
            final_hour = hour
        else:
            final_hour = hour + 12 if hour <= 10 else hour
        return datetime.combine(date_part, time_part.replace(hour=final_hour))
    except:
        return datetime.strptime(date_str, '%Y-%m-%d')

def calculate_historical_win_pct(historical_records):
    total_races = len(historical_records)
    if total_races == 0:
        return -1
    wins = sum(1 for record in historical_records if record['win'])
    return (wins / total_races) * 100

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

merged_df['datetime'] = merged_df.apply(
    lambda row: parse_race_time(row['date'], row['off']), axis=1
)
merged_df = merged_df.sort_values(['datetime', 'race_id']).reset_index(drop=True)

horse_history = defaultdict(list)
jockey_history = defaultdict(list)
trainer_history = defaultdict(list)

win_pct_columns = [
    'horse_course_win_pct', 'horse_distance_win_pct', 'horse_going_win_pct',
    'jockey_win_pct', 'trainer_win_pct'
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
    win = (row['pos'] == '1')
    
    horse_records = horse_history[horse_id]
    jockey_records = jockey_history[jockey_id]
    trainer_records = trainer_history[trainer_id]
    
    # Course-specific win percentage
    course_records = [r for r in horse_records if r['course'] == course]
    course_win_pct = calculate_historical_win_pct(course_records)
    merged_df.at[idx, 'horse_course_win_pct'] = course_win_pct
    
    # Distance-specific win percentage
    distance_records = [r for r in horse_records if r['dist_f'] == dist_f]
    distance_win_pct = calculate_historical_win_pct(distance_records)
    merged_df.at[idx, 'horse_distance_win_pct'] = distance_win_pct
    
    # Going-specific win percentage
    going_records = [r for r in horse_records if r['going'] == going]
    going_win_pct = calculate_historical_win_pct(going_records)
    merged_df.at[idx, 'horse_going_win_pct'] = going_win_pct
    
    # Jockey overall win percentage
    jockey_win_pct = calculate_historical_win_pct(jockey_records)
    merged_df.at[idx, 'jockey_win_pct'] = jockey_win_pct
    
    # Trainer overall win percentage  
    trainer_win_pct = calculate_historical_win_pct(trainer_records)
    merged_df.at[idx, 'trainer_win_pct'] = trainer_win_pct
    
    # Add current race to horse history
    race_record = {
        'course': course,
        'going': going, 
        'dist_f': dist_f,
        'win': win,
        'datetime': row['datetime']
    }
    
    horse_history[horse_id].append(race_record)
    
    # Add current race to jockey history
    jockey_record = {
        'win': win,
        'datetime': row['datetime']
    }
    jockey_history[jockey_id].append(jockey_record)
    
    # Add current race to trainer history
    trainer_record = {
        'win': win,
        'datetime': row['datetime']
    }
    trainer_history[trainer_id].append(trainer_record)

# date, region
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df['month'] = merged_df['date'].dt.month
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df.drop(columns=['date', 'month', 'region'], inplace=True, errors='ignore')

# course, course_id, dist, dist_f, dist_m
merged_df['track_id'] = merged_df['course'] + '_' + merged_df['dist_f'].astype(str)
unique_tracks = merged_df['track_id'].unique()
track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
merged_df['track_id'] = merged_df['track_id'].map(track_to_id).astype(int)

save_mapping(track_to_id, 'track_mapping.json')

merged_df['dist_f'] = merged_df['dist_f'].str.rstrip('f').astype(float)

merged_df.drop(columns=['course', 'course_id', 'dist', 'dist_m'], inplace=True, errors='ignore')

# race_id, off, race_name
merged_df.drop(columns=['off', 'race_name'], inplace=True, errors='ignore')

# type, class, pattern
unique_types = merged_df['type'].unique()
type_to_id = {type: idx for idx, type in enumerate(unique_types)}
merged_df['type'] = merged_df['type'].map(type_to_id).astype(int)

save_mapping(type_to_id, 'type_mapping.json')

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

# dec
merged_df.drop(columns=['dec'], inplace=True, errors='ignore')

# jockey_id, jockey, trainer_id, trainer, owner_id, owner
owner_to_id = {}
for _, row in merged_df[['owner', 'owner_id']].dropna().iterrows():
    owner_name = row['owner']
    owner_id = int(row['owner_id'])
    if owner_name not in owner_to_id:
        owner_to_id[owner_name] = owner_id

save_mapping(owner_to_id, 'owner_mapping.json')

merged_df.drop(columns=['jockey', 'trainer', 'owner'], inplace=True, errors='ignore')

# prize
merged_df.drop(columns=['prize'], inplace=True, errors='ignore')

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

merged_df.drop(columns=['sire_id', 'dam_id', 'damsire_id'], inplace=True, errors='ignore')

# comment, datetime
merged_df.drop(columns=['comment', 'datetime'], inplace=True, errors='ignore')

output_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_file, index=False)
