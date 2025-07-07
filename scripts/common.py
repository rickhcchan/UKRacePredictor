# scripts/common.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR  / 'data'
RPSCRAPE_DIR = PROJECT_DIR.parent.parent / 'joenano' / 'rpscrape'

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

def calculate_win_percentage(runs, wins):
    """Calculate win percentage, handling edge cases"""
    try:
        runs = int(runs) if not isinstance(runs, int) else runs
        wins = int(wins) if not isinstance(wins, int) else wins
        
        if runs == 0:
            return -1.0
        
        return (wins / runs) * 100.0
    except (ValueError, ZeroDivisionError, TypeError):
        return -1.0

def calculate_14d_stats_from_encoded(jockey_id, trainer_id, current_date, encoded_df, race_type=None):
    """
    Calculate 14-day statistics from encoded.csv data with optional race type filtering.
    
    This function is shared between encode_data.py and cleanse_racecard.py to ensure
    exactly the same calculation logic is used for both training and prediction.
    
    Args:
        jockey_id: ID of the jockey
        trainer_id: ID of the trainer  
        current_date: Current race date (string or datetime)
        encoded_df: DataFrame with encoded race data
        race_type: Optional race type to filter by (None for all races)
        
    Returns:
        dict: Dictionary with jockey and trainer 14-day stats
    """
    if isinstance(current_date, str):
        current_datetime = datetime.strptime(current_date, '%Y-%m-%d')
    else:
        current_datetime = current_date
        
    cutoff_date = current_datetime - timedelta(days=14)
    
    # Filter data before current date and within 14 days
    recent_data = encoded_df[
        (encoded_df['datetime'] >= cutoff_date) & 
        (encoded_df['datetime'] < current_datetime)
    ]
    
    # If race type is provided, filter by race type
    if race_type is not None:
        recent_data_filtered = recent_data[recent_data['type_id'] == race_type]
    else:
        recent_data_filtered = recent_data
    
    # Jockey 14-day stats
    jockey_recent = recent_data_filtered[recent_data_filtered['jockey_id'] == jockey_id]
    jockey_14d_runs = len(jockey_recent)
    jockey_14d_wins = len(jockey_recent[jockey_recent['win'] == 1])
    jockey_14d_win_pct = calculate_win_percentage(jockey_14d_runs, jockey_14d_wins)
    
    # Trainer 14-day stats
    trainer_recent = recent_data_filtered[recent_data_filtered['trainer_id'] == trainer_id]
    trainer_14d_runs = len(trainer_recent)
    trainer_14d_wins = len(trainer_recent[trainer_recent['win'] == 1])
    trainer_14d_win_pct = calculate_win_percentage(trainer_14d_runs, trainer_14d_wins)
    
    return {
        'jockey_14d_runs': float(jockey_14d_runs),
        'jockey_14d_wins': float(jockey_14d_wins),
        'jockey_14d_win_pct': jockey_14d_win_pct,
        'trainer_14d_runs': float(trainer_14d_runs),
        'trainer_14d_wins': float(trainer_14d_wins),
        'trainer_14d_win_pct': trainer_14d_win_pct
    }

def calculate_14d_win_pct_from_history(historical_records, current_datetime, race_type=None):
    """
    Calculate 14-day win percentage from historical records list.
    
    This is used by encode_data.py during the encoding process.
    
    Args:
        historical_records: List of race records
        current_datetime: Current race datetime
        race_type: Optional race type to filter by (for type-specific stats)
        
    Returns:
        tuple: (win_pct, total_runs, total_wins)
    """
    if not historical_records:
        return -1.0, 0.0, 0.0
    
    cutoff_date = current_datetime - timedelta(days=14)
    recent_records = [r for r in historical_records if r['datetime'] >= cutoff_date]
    
    # Filter by race type if specified
    if race_type is not None:
        recent_records = [r for r in recent_records if r.get('type_id') == race_type]
    
    total_runs = len(recent_records)
    if total_runs == 0:
        return -1.0, 0.0, 0.0
    
    total_wins = sum(1 for record in recent_records if record['win'])
    win_pct = calculate_win_percentage(total_runs, total_wins)
    return win_pct, float(total_runs), float(total_wins)

def convert_to_24h_time(time_str):
    """
    Convert race time to 24-hour format for proper chronological sorting.
    
    Logic:
    - Morning times (11:00, 12:00) stay as-is
    - Afternoon times (1:00-10:00) get converted to 13:00-22:00
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        24-hour format time string (HH:MM)
    """
    try:
        time_part = datetime.strptime(time_str, '%H:%M').time()
        hour = time_part.hour
        if hour in [11, 12]:
            final_hour = hour
        else:
            final_hour = hour + 12 if hour <= 10 else hour
        return f"{final_hour:02d}:{time_part.minute:02d}"
    except:
        return time_str  # Return original if conversion fails

def parse_race_time(date_str, time_str):
    """
    Parse race date and time into a datetime object with 24-hour conversion.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        time_str: Time string in HH:MM format
        
    Returns:
        datetime object with proper 24-hour time
    """
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
