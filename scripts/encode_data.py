import json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "raw"
csv_files = list(DATA_DIR.rglob("*.csv"))

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # df['SourceFile'] = csv_file.name  # track file name
    # df['RaceType'] = csv_file.parent.name  # track 'flat' or 'jumps'
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

# date, region
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
merged_df['month'] = merged_df['date'].dt.month
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df.drop(columns=['date', 'month', 'region'], inplace=True)

# course
unique_courses = merged_df['course'].unique()
course_to_id = {course: idx for idx, course in enumerate(unique_courses)}
merged_df['course_id'] = merged_df['course'].map(course_to_id).astype('category')

mapping_path = SCRIPT_DIR.parent / "data" / "processed" / "course_mapping.json"
mapping_path.parent.mkdir(parents=True, exist_ok=True)
with open(mapping_path, 'w') as f:
    json.dump(course_to_id, f)

merged_df.drop(columns=['course'], inplace=True)

# course, distance
merged_df['track'] = merged_df['course'] + '_' + merged_df['dist'].astype(str)
unique_tracks = merged_df['track'].unique()
track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
merged_df['track_id'] = merged_df['track'].map(track_to_id).astype('category')

mapping_path = SCRIPT_DIR.parent / "data" / "processed" / "track_mapping.json"
mapping_path.parent.mkdir(parents=True, exist_ok=True)
with open(mapping_path, 'w') as f:
    json.dump(track_to_id, f)

merged_df.drop(columns=['course', 'dist', 'dist_f'], inplace=True)

going_order = ['Firm', 'Good to Firm', 'Good', 'Good to Soft', 'Soft', 'Heavy']
merged_df['going'] = pd.Categorical(
    merged_df['going'],
    categories=condition_order,
    ordered=True
)
merged_df['GroundCondition'] = merged_df['GroundCondition'].cat.codes

# Save merged + encoded
output_file = Path("../data/processed/all_merged_encoded.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
#merged_df.to_csv(output_file, index=False)

print(f"Encoded data saved to {output_file}")
