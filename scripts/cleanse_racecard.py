# scripts/cleanse_racecard.py

import json
import pandas as pd
import numpy as np
import re
import shutil
from datetime import datetime, timedelta
from collections import defaultdict

from common import DATA_DIR, RPSCRAPE_DIR, map_hg, parse_age_band, calculate_14d_stats_from_encoded, calculate_win_percentage
from mappings import pattern_map, going_map, sex_map
from pathlib import Path

def parse_race_time(date_str, time_str):
    """Parse race date and time similar to encode_data.py"""
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

def load_encoded_data():
    try:
        encoded_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
        if not encoded_file.exists():
            return None
        
        encoded_df = pd.read_csv(encoded_file)
        encoded_df['datetime'] = pd.to_datetime(encoded_df['datetime'])
        
        return encoded_df
        
    except Exception as e:
        print(f"Error loading encoded data: {e}")
        return None


# File setup
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

# Load mappings
with open(DATA_DIR / 'mapping' / 'type_mapping.json') as f:
    type_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'owner_mapping.json') as f:
    owner_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'track_mapping.json') as f:
    track_mapping = json.load(f)

with open(DATA_DIR / 'mapping' / 'horse_bloodlines.json') as f:
    horse_bloodlines_raw = json.load(f)
    horse_bloodlines = {int(k): v for k, v in horse_bloodlines_raw.items()}

with open(racecard_file, encoding='utf-8') as f:
    racecard_data = json.load(f)

encoded_data = load_encoded_data()


rows = []

courses = racecard_data['GB']
for course_name, races in courses.items():
    for race_time, race_info in races.items():
        class_match = re.search(r'(\d)', race_info.get('race_class', ''))
        class_id = int(class_match.group(1)) if class_match else -1
        
        runners = race_info.get("runners", [])
        for runner in runners:

            # Initialize all win percentages
            horse_course_win_pct = -1.0
            horse_distance_win_pct = -1.0
            horse_going_win_pct = -1.0
            jockey_win_pct = -1.0
            trainer_win_pct = -1.0

            # Calculate all statistics from encoded data only
            if encoded_data is not None:
                horse_id = int(runner.get('horse_id', -1))
                jockey_id = int(runner.get('jockey_id', -1))
                trainer_id = int(runner.get('trainer_id', -1))
                race_type = type_mapping.get(race_info['type'], -1)
                
                # Horse-specific win percentages: calculate fresh for current race conditions
                if horse_id != -1:
                    horse_records = encoded_data[encoded_data['horse_id'] == horse_id]
                    
                    if len(horse_records) > 0:
                        # Course-specific win percentage - SKIP for now (would need course column in encoded data)
                        # track_id combines course+distance, so can't separate course-only stats
                        horse_course_win_pct = -1.0
                        
                        # Distance-specific win percentage for this specific distance
                        distance_records = horse_records[horse_records['dist_f'] == race_info.get('distance_f')]
                        if len(distance_records) > 0:
                            distance_wins = (distance_records['win'] == 1).sum()
                            horse_distance_win_pct = distance_wins / len(distance_records)
                        
                        # Going-specific win percentage for this specific going
                        going_id = going_map.get(race_info.get('going'), 3)
                        going_records = horse_records[horse_records['going'] == going_id]
                        if len(going_records) > 0:
                            going_wins = (going_records['win'] == 1).sum()
                            horse_going_win_pct = going_wins / len(going_records)
                
                # Jockey lifetime win percentage from encoded data only
                if jockey_id != -1:
                    jockey_records = encoded_data[encoded_data['jockey_id'] == jockey_id]
                    if len(jockey_records) > 0:
                        jockey_wins = (jockey_records['win'] == 1).sum()
                        jockey_win_pct = jockey_wins / len(jockey_records)
                
                # Trainer lifetime win percentage from encoded data only
                if trainer_id != -1:
                    trainer_records = encoded_data[encoded_data['trainer_id'] == trainer_id]
                    if len(trainer_records) > 0:
                        trainer_wins = (trainer_records['win'] == 1).sum()
                        trainer_win_pct = trainer_wins / len(trainer_records)
                
                # Calculate overall 14-day stats (all race types)
                stats_overall = calculate_14d_stats_from_encoded(
                    jockey_id, trainer_id, today, encoded_data, race_type=None
                )
                
                # Calculate race-type specific 14-day stats
                stats_bytype = calculate_14d_stats_from_encoded(
                    jockey_id, trainer_id, today, encoded_data, race_type=race_type
                )
                
                # Use overall stats as primary 14d features
                jockey_14d_runs = stats_overall['jockey_14d_runs']
                jockey_14d_wins = stats_overall['jockey_14d_wins']
                jockey_14d_win_pct = stats_overall['jockey_14d_win_pct']
                
                trainer_14d_runs = stats_overall['trainer_14d_runs']
                trainer_14d_wins = stats_overall['trainer_14d_wins']
                trainer_14d_win_pct = stats_overall['trainer_14d_win_pct']
                
                # Add race-type specific stats as additional features
                jockey_14d_type_runs = stats_bytype['jockey_14d_runs']
                jockey_14d_type_wins = stats_bytype['jockey_14d_wins']
                jockey_14d_type_win_pct = stats_bytype['jockey_14d_win_pct']
                
                trainer_14d_type_runs = stats_bytype['trainer_14d_runs']
                trainer_14d_type_wins = stats_bytype['trainer_14d_wins']
                trainer_14d_type_win_pct = stats_bytype['trainer_14d_win_pct']
                
            else:
                # If encoded data not available, all statistics remain -1.0 (no data)
                jockey_14d_runs = 0.0
                jockey_14d_wins = 0.0
                jockey_14d_win_pct = -1.0
                
                trainer_14d_runs = 0.0
                trainer_14d_wins = 0.0
                trainer_14d_win_pct = -1.0
                
                jockey_14d_type_runs = 0.0
                jockey_14d_type_wins = 0.0
                jockey_14d_type_win_pct = -1.0
                
                trainer_14d_type_runs = 0.0
                trainer_14d_type_wins = 0.0
                trainer_14d_type_win_pct = -1.0

            # Get bloodline IDs from horse_bloodlines mapping
            horse_id = int(runner.get('horse_id', -1))
            if horse_id in horse_bloodlines:
                bloodline = horse_bloodlines[horse_id]
                sire_id = bloodline['sire_id']
                dam_id = bloodline['dam_id']
                damsire_id = bloodline['damsire_id']
            else:
                sire_id = -1
                dam_id = -1
                damsire_id = -1

            row = {
                'race_id': int(race_info.get('race_id')),
                'type_id': type_mapping.get(race_info['type'], -1),
                'class': class_id,
                'pattern': pattern_map.get(race_info.get('pattern'), 5),
                'dist_f': race_info.get('distance_f'),
                'going': going_map.get(race_info.get('going'), 3),
                'ran': len(race_info.get("runners", [])),
                'draw': int(runner.get('draw', -1)),
                'horse_id': horse_id,
                'horse_name': runner.get('name', ''),
                'age': int(runner.get('age')),
                'sex': sex_map.get(runner.get('sex_code'), -1),
                'lbs': int(runner.get('lbs', 0)),
                'hg': map_hg(runner.get('headgear')),
                'jockey_id': int(runner.get('jockey_id', -1)),
                'trainer_id': int(runner.get('trainer_id', -1)),
                'or': int(runner.get('ofr', -1)) if runner.get('ofr') is not None else -1,
                'rpr': int(runner.get('rpr', -1)) if runner.get('rpr') is not None else -1,
                'ts': int(runner.get('ts', -1)) if runner.get('ts') is not None else -1,
                'sire_id': sire_id,
                'dam_id': dam_id,
                'damsire_id': damsire_id,
                'owner_id': owner_mapping.get(runner.get('owner', '').replace(' & ', ' ').replace('-', ' '), -1),
                'rating_band': race_info.get('rating_band', ''),
                'age_band': race_info.get('age_band', ''),
                'course': course_name,
                'date': today,
                'horse_course_win_pct': horse_course_win_pct,
                'horse_distance_win_pct': horse_distance_win_pct,
                'horse_going_win_pct': horse_going_win_pct,
                'jockey_win_pct': jockey_win_pct,
                'trainer_win_pct': trainer_win_pct,
                'jockey_14d_runs': jockey_14d_runs,
                'jockey_14d_wins': jockey_14d_wins,
                'jockey_14d_win_pct': jockey_14d_win_pct,
                'trainer_14d_runs': trainer_14d_runs,
                'trainer_14d_wins': trainer_14d_wins,
                'trainer_14d_win_pct': trainer_14d_win_pct,
                'jockey_14d_type_runs': jockey_14d_type_runs,
                'jockey_14d_type_wins': jockey_14d_type_wins,
                'jockey_14d_type_win_pct': jockey_14d_type_win_pct,
                'trainer_14d_type_runs': trainer_14d_type_runs,
                'trainer_14d_type_wins': trainer_14d_type_wins,
                'trainer_14d_type_win_pct': trainer_14d_type_win_pct,
            }
            
            rows.append(row)

df = pd.DataFrame(rows)

# Feature engineering
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['track_id'] = (df['course'] + '_' + df['dist_f'].apply(lambda x: str(int(x)) if x == int(x) else str(x)) + 'f').map(track_mapping).fillna(-1).astype(int)

df[['age_min', 'age_max']] = df['age_band'].apply(parse_age_band).apply(pd.Series)

# Extract rating_max from rating_band (same as in encode_data.py)
df['rating_max'] = df['rating_band'].str.extract(r'-(\d+)').astype(float)

df.drop(columns=['course', 'date', 'month', 'rating_band', 'age_band'], inplace=True, errors='ignore')

# Define final columns for model input
clean_prediction_cols = [
    'race_id', 'type_id', 'class', 'pattern', 'dist_f', 'going', 'ran', 'draw',
    'horse_id', 'horse_name', 'age', 'sex', 'lbs', 'hg', 'jockey_id', 'trainer_id', 
    'sire_id', 'dam_id', 'damsire_id', 'owner_id',
    'month_sin', 'month_cos', 'track_id', 'age_min', 'age_max', 'rating_max',
    'horse_course_win_pct', 'horse_distance_win_pct', 'horse_going_win_pct',
    'jockey_win_pct', 'trainer_win_pct',
    'jockey_14d_runs', 'jockey_14d_wins', 'jockey_14d_win_pct',
    'trainer_14d_runs', 'trainer_14d_wins', 'trainer_14d_win_pct',
    'jockey_14d_type_runs', 'jockey_14d_type_wins', 'jockey_14d_type_win_pct',
    'trainer_14d_type_runs', 'trainer_14d_type_wins', 'trainer_14d_type_win_pct'
]
# REMOVED: 'or', 'rpr', 'ts' (data leakage)

final_cols = [col for col in clean_prediction_cols if col in df.columns]
df = df[final_cols]

# Save output
output_file = DATA_DIR / 'prediction' / 'processed' / f"racecard_{datetime.now().strftime('%Y-%m-%d')}_cleansed.csv"
output_file.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(output_file, index=False)

print(f'Saved cleansed racecard data to {output_file}')
print(f'Final columns: {list(df.columns)}')
print(f'Total horses: {len(df)}')
print(f'Available 14d features: overall and by-type for both jockey and trainer')
