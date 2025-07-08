"""
Incremental race data encoding script.

This script handles the complete incremental encoding workflow:
1. Check encoded_race_data table to get max date (last complete day)
2. Always delete the last day's encoded data (handle incomplete encoding)
3. Process raw race data from last_date to max available date
4. Encode features day-by-day in chronological order
5. Store encoded features directly in encoded_race_data table

The script automatically processes from the last encoded date to the latest raw data.
Maintains historical context for accurate feature engineering.

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python encode_incremental.py [--dry-run]
    
Examples:
    # Encode all new data since last encoded date
    python encode_incremental.py
    
    # Test run without making changes
    python encode_incremental.py --dry-run
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

try:
    from common import setup_logging
except ImportError:
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

class IncrementalEncoder:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Set database path from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.encoded_table = self._get_config_value('encode_incremental', 'encoded_table', 'encoded_race_data')
        self.batch_size = int(self._get_config_value('encode_incremental', 'batch_size', '100'))
        self.process_mode = self._get_config_value('encode_incremental', 'process_mode', 'daily')
        
        self.logger.info(f"Using database: {self.db_path}")
        self.logger.info(f"Encoded table: {self.encoded_table}")
        self.logger.info(f"Process mode: {self.process_mode}")
        
        # Initialize historical tracking for features
        self.horse_history = defaultdict(list)
        self.jockey_history = defaultdict(list)
        self.trainer_history = defaultdict(list)
        
        # Feature mappings
        self.mappings = {
            'going': {},
            'pattern': {},
            'sex': {},
            'type': {},
            'track': {}
        }

    def _get_config_value(self, section: str, key: str, default: str = None) -> str:
        """Get a configuration value with fallback to default."""
        if self.config and self.config.has_option(section, key):
            value = self.config.get(section, key)
            # Handle relative paths
            if key.endswith('_path') or key.endswith('_dir'):
                if not os.path.isabs(value):
                    project_root = Path(__file__).parent.parent
                    return str(project_root / value)
            return value
        return default

    def _load_config(self) -> Optional[configparser.ConfigParser]:
        """Load configuration from user_settings.conf or default_settings.conf."""
        project_root = Path(__file__).parent.parent
        
        # Try user settings first, fallback to default settings
        user_config = project_root / "config" / "user_settings.conf"
        default_config = project_root / "config" / "default_settings.conf"
        
        config = configparser.ConfigParser()
        
        # Always load default settings first (if available)
        if default_config.exists():
            config.read(str(default_config))
            self.logger.info(f"Loaded default configuration from: {default_config}")
        
        # Override with user settings if they exist
        if user_config.exists():
            config.read(str(user_config))
            self.logger.info(f"Loaded user configuration from: {user_config}")
        elif default_config.exists():
            self.logger.info("No user_settings.conf found, using default_settings.conf only")
        else:
            self.logger.warning("No configuration files found, using built-in defaults")
            return None
        
        return config

    def init_encoded_table(self):
        """Initialize the encoded race data table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create encoded features table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.encoded_table} (
                    race_id INTEGER,
                    horse_id INTEGER,
                    date TEXT,
                    
                    -- Basic race features
                    course TEXT,
                    race_name TEXT,
                    type_id INTEGER,
                    class TEXT,
                    pattern_id INTEGER,
                    rating_band TEXT,
                    age_band TEXT,
                    sex_rest TEXT,
                    dist_f REAL,
                    going_id INTEGER,
                    ran INTEGER,
                    track_id INTEGER,
                    
                    -- Time features
                    month_sin REAL,
                    month_cos REAL,
                    
                    -- Horse features
                    horse_name TEXT,
                    age INTEGER,
                    sex_id INTEGER,
                    lbs INTEGER,
                    hg INTEGER,
                    draw INTEGER,
                    
                    -- Horse historical features
                    horse_total_runs INTEGER,
                    horse_total_wins INTEGER,
                    horse_win_pct REAL,
                    horse_course_runs INTEGER,
                    horse_course_wins INTEGER,
                    horse_course_win_pct REAL,
                    horse_distance_runs INTEGER,
                    horse_distance_wins INTEGER,
                    horse_distance_win_pct REAL,
                    horse_going_runs INTEGER,
                    horse_going_wins INTEGER,
                    horse_going_win_pct REAL,
                    
                    -- Jockey features
                    jockey_id INTEGER,
                    jockey_total_runs INTEGER,
                    jockey_total_wins INTEGER,
                    jockey_win_pct REAL,
                    jockey_course_runs INTEGER,
                    jockey_course_wins INTEGER,
                    jockey_course_win_pct REAL,
                    jockey_distance_runs INTEGER,
                    jockey_distance_wins INTEGER,
                    jockey_distance_win_pct REAL,
                    jockey_going_runs INTEGER,
                    jockey_going_wins INTEGER,
                    jockey_going_win_pct REAL,
                    jockey_14d_runs INTEGER,
                    jockey_14d_wins INTEGER,
                    jockey_14d_win_pct REAL,
                    jockey_14d_type_runs INTEGER,
                    jockey_14d_type_wins INTEGER,
                    jockey_14d_type_win_pct REAL,
                    
                    -- Trainer features
                    trainer_id INTEGER,
                    trainer_total_runs INTEGER,
                    trainer_total_wins INTEGER,
                    trainer_win_pct REAL,
                    trainer_course_runs INTEGER,
                    trainer_course_wins INTEGER,
                    trainer_course_win_pct REAL,
                    trainer_distance_runs INTEGER,
                    trainer_distance_wins INTEGER,
                    trainer_distance_win_pct REAL,
                    trainer_going_runs INTEGER,
                    trainer_going_wins INTEGER,
                    trainer_going_win_pct REAL,
                    trainer_14d_runs INTEGER,
                    trainer_14d_wins INTEGER,
                    trainer_14d_win_pct REAL,
                    trainer_14d_type_runs INTEGER,
                    trainer_14d_type_wins INTEGER,
                    trainer_14d_type_win_pct REAL,
                    
                    -- Ratings and performance
                    or_rating INTEGER,
                    rpr INTEGER,
                    ts INTEGER,
                    
                    -- Bloodline features
                    sire_id INTEGER,
                    dam_id INTEGER,
                    damsire_id INTEGER,
                    owner_id INTEGER,
                    
                    -- Target variable (for training)
                    win INTEGER,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (race_id, horse_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_date ON {self.encoded_table}(date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_horse ON {self.encoded_table}(horse_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_race ON {self.encoded_table}(race_id)")
            
            conn.commit()
            self.logger.info(f"Encoded table {self.encoded_table} initialized successfully")

    def get_last_encoded_date(self) -> Optional[str]:
        """Get the last complete encoded date."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT MAX(date) FROM {self.encoded_table}")
            result = cursor.fetchone()
            
            if result and result[0]:
                self.logger.info(f"Last encoded date: {result[0]}")
                return result[0]
            else:
                self.logger.info("No encoded data found")
                return None

    def get_max_raw_date(self) -> Optional[str]:
        """Get the maximum date available in raw race data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM race_data")
            result = cursor.fetchone()
            
            if result and result[0]:
                self.logger.info(f"Max raw data date: {result[0]}")
                return result[0]
            else:
                self.logger.info("No raw race data found")
                return None

    def delete_encoded_date_data(self, date_str: str):
        """Delete all encoded data for a specific date."""
        if self.dry_run:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {self.encoded_table} WHERE date = ?", (date_str,))
                count = cursor.fetchone()[0]
                self.logger.info(f"[DRY RUN] Would delete {count} encoded records for date {date_str}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"DELETE FROM {self.encoded_table} WHERE date = ?", (date_str,))
            deleted_count = cursor.rowcount
            conn.commit()
            self.logger.info(f"Deleted {deleted_count} encoded records for date {date_str}")

    def load_historical_context(self, up_to_date: str):
        """Load historical context up to a specific date for accurate feature calculation."""
        self.logger.info(f"Loading historical context up to {up_to_date}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Load all historical race data up to the date (exclusive)
            query = """
            SELECT horse_id, jockey_id, trainer_id, course, going, dist_f, type, 
                   pos, date, race_id
            FROM race_data 
            WHERE date < ? 
            ORDER BY date, race_id
            """
            
            df = pd.read_sql_query(query, conn, params=(up_to_date,))
            
        self.logger.info(f"Loaded {len(df)} historical records for context")
        
        # Build historical tracking
        for _, row in df.iterrows():
            horse_id = row['horse_id']
            jockey_id = row['jockey_id']
            trainer_id = row['trainer_id']
            win = (row['pos'] == 1 or row['pos'] == '1')
            
            # Build horse history
            self.horse_history[horse_id].append({
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date']
            })
            
            # Build jockey history
            self.jockey_history[jockey_id].append({
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date']
            })
            
            # Build trainer history
            self.trainer_history[trainer_id].append({
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date']
            })

    def encode_daily_races(self, date_str: str) -> int:
        """Encode all races for a specific date."""
        self.logger.info(f"Encoding races for {date_str}")
        
        # Load raw data for this date
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT * FROM race_data 
            WHERE date = ? 
            ORDER BY race_id, pos
            """
            df = pd.read_sql_query(query, conn, params=(date_str,))
        
        if len(df) == 0:
            self.logger.info(f"No races found for {date_str}")
            return 0
        
        self.logger.info(f"Found {len(df)} race records for {date_str}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would encode {len(df)} records for {date_str}")
            # Update historical context with this day's results
            self._update_historical_context(df)
            return len(df)
        
        # Encode features for all races on this date
        encoded_features = []
        
        for _, row in df.iterrows():
            features = self._encode_single_record(row)
            encoded_features.append(features)
        
        # Save encoded features to database
        if encoded_features:
            encoded_df = pd.DataFrame(encoded_features)
            with sqlite3.connect(self.db_path) as conn:
                encoded_df.to_sql(self.encoded_table, conn, if_exists='append', index=False)
                conn.commit()
        
        # Update historical context with this day's results
        self._update_historical_context(df)
        
        self.logger.info(f"✓ Encoded {len(encoded_features)} records for {date_str}")
        return len(encoded_features)

    def _encode_single_record(self, row) -> Dict:
        """Encode features for a single race record."""
        horse_id = row['horse_id']
        jockey_id = row['jockey_id']
        trainer_id = row['trainer_id']
        course = row['course']
        going = row['going']
        dist_f = row['dist_f']
        race_type = row['type']
        race_date = row['date']
        
        # Get historical records
        horse_records = self.horse_history[horse_id]
        jockey_records = self.jockey_history[jockey_id]
        trainer_records = self.trainer_history[trainer_id]
        
        # Calculate features
        features = {
            'race_id': row['race_id'],
            'horse_id': horse_id,
            'date': race_date,
            
            # Basic race features
            'course': course,
            'race_name': row['race_name'],
            'type_id': self._get_or_create_mapping('type', race_type),
            'class': row['class'],
            'pattern_id': self._get_or_create_mapping('pattern', row.get('pattern', 'Unknown')),
            'rating_band': row['rating_band'],
            'age_band': row['age_band'],
            'sex_rest': row['sex_rest'],
            'dist_f': float(str(dist_f).rstrip('f')) if pd.notna(dist_f) else 0.0,
            'going_id': self._get_or_create_mapping('going', going),
            'ran': row['ran'],
            'track_id': self._get_or_create_mapping('track', f"{course}_{dist_f}"),
            
            # Time features
            'month_sin': np.sin(2 * np.pi * pd.to_datetime(race_date).month / 12),
            'month_cos': np.cos(2 * np.pi * pd.to_datetime(race_date).month / 12),
            
            # Horse features
            'horse_name': row['horse'],
            'age': row['age'],
            'sex_id': self._get_or_create_mapping('sex', row['sex']),
            'lbs': row['lbs'],
            'hg': self._map_hg(row.get('hg')),
            'draw': row['draw'],
            
            # Jockey features
            'jockey_id': jockey_id,
            
            # Trainer features
            'trainer_id': trainer_id,
            
            # Ratings
            'or_rating': row.get('or_rating') or row.get('or'),
            'rpr': row['rpr'],
            'ts': row['ts'],
            
            # Bloodline
            'sire_id': row['sire_id'],
            'dam_id': row['dam_id'],
            'damsire_id': row['damsire_id'],
            'owner_id': row['owner_id'],
            
            # Target
            'win': 1 if (row['pos'] == 1 or row['pos'] == '1') else 0
        }
        
        # Calculate historical statistics
        features.update(self._calculate_horse_stats(horse_records, course, going, dist_f))
        features.update(self._calculate_jockey_stats(jockey_records, course, going, dist_f, race_type, race_date))
        features.update(self._calculate_trainer_stats(trainer_records, course, going, dist_f, race_type, race_date))
        
        return features

    def _calculate_horse_stats(self, records: List[Dict], course: str, going: str, dist_f: str) -> Dict:
        """Calculate horse historical statistics."""
        total_runs = len(records)
        total_wins = sum(1 for r in records if r['win'])
        
        course_records = [r for r in records if r['course'] == course]
        course_runs = len(course_records)
        course_wins = sum(1 for r in course_records if r['win'])
        
        distance_records = [r for r in records if r['dist_f'] == dist_f]
        distance_runs = len(distance_records)
        distance_wins = sum(1 for r in distance_records if r['win'])
        
        going_records = [r for r in records if r['going'] == going]
        going_runs = len(going_records)
        going_wins = sum(1 for r in going_records if r['win'])
        
        return {
            'horse_total_runs': total_runs,
            'horse_total_wins': total_wins,
            'horse_win_pct': (total_wins / total_runs * 100) if total_runs > 0 else -1.0,
            'horse_course_runs': course_runs,
            'horse_course_wins': course_wins,
            'horse_course_win_pct': (course_wins / course_runs * 100) if course_runs > 0 else -1.0,
            'horse_distance_runs': distance_runs,
            'horse_distance_wins': distance_wins,
            'horse_distance_win_pct': (distance_wins / distance_runs * 100) if distance_runs > 0 else -1.0,
            'horse_going_runs': going_runs,
            'horse_going_wins': going_wins,
            'horse_going_win_pct': (going_wins / going_runs * 100) if going_runs > 0 else -1.0,
        }

    def _calculate_jockey_stats(self, records: List[Dict], course: str, going: str, dist_f: str, race_type: str, race_date: str) -> Dict:
        """Calculate jockey historical statistics."""
        total_runs = len(records)
        total_wins = sum(1 for r in records if r['win'])
        
        course_records = [r for r in records if r['course'] == course]
        course_runs = len(course_records)
        course_wins = sum(1 for r in course_records if r['win'])
        
        distance_records = [r for r in records if r['dist_f'] == dist_f]
        distance_runs = len(distance_records)
        distance_wins = sum(1 for r in distance_records if r['win'])
        
        going_records = [r for r in records if r['going'] == going]
        going_runs = len(going_records)
        going_wins = sum(1 for r in going_records if r['win'])
        
        # 14-day statistics
        cutoff_date = (pd.to_datetime(race_date) - timedelta(days=14)).strftime('%Y-%m-%d')
        recent_records = [r for r in records if r['date'] >= cutoff_date]
        recent_runs = len(recent_records)
        recent_wins = sum(1 for r in recent_records if r['win'])
        
        recent_type_records = [r for r in recent_records if r['type'] == race_type]
        recent_type_runs = len(recent_type_records)
        recent_type_wins = sum(1 for r in recent_type_records if r['win'])
        
        return {
            'jockey_total_runs': total_runs,
            'jockey_total_wins': total_wins,
            'jockey_win_pct': (total_wins / total_runs * 100) if total_runs > 0 else -1.0,
            'jockey_course_runs': course_runs,
            'jockey_course_wins': course_wins,
            'jockey_course_win_pct': (course_wins / course_runs * 100) if course_runs > 0 else -1.0,
            'jockey_distance_runs': distance_runs,
            'jockey_distance_wins': distance_wins,
            'jockey_distance_win_pct': (distance_wins / distance_runs * 100) if distance_runs > 0 else -1.0,
            'jockey_going_runs': going_runs,
            'jockey_going_wins': going_wins,
            'jockey_going_win_pct': (going_wins / going_runs * 100) if going_runs > 0 else -1.0,
            'jockey_14d_runs': recent_runs,
            'jockey_14d_wins': recent_wins,
            'jockey_14d_win_pct': (recent_wins / recent_runs * 100) if recent_runs > 0 else -1.0,
            'jockey_14d_type_runs': recent_type_runs,
            'jockey_14d_type_wins': recent_type_wins,
            'jockey_14d_type_win_pct': (recent_type_wins / recent_type_runs * 100) if recent_type_runs > 0 else -1.0,
        }

    def _calculate_trainer_stats(self, records: List[Dict], course: str, going: str, dist_f: str, race_type: str, race_date: str) -> Dict:
        """Calculate trainer historical statistics."""
        total_runs = len(records)
        total_wins = sum(1 for r in records if r['win'])
        
        course_records = [r for r in records if r['course'] == course]
        course_runs = len(course_records)
        course_wins = sum(1 for r in course_records if r['win'])
        
        distance_records = [r for r in records if r['dist_f'] == dist_f]
        distance_runs = len(distance_records)
        distance_wins = sum(1 for r in distance_records if r['win'])
        
        going_records = [r for r in records if r['going'] == going]
        going_runs = len(going_records)
        going_wins = sum(1 for r in going_records if r['win'])
        
        # 14-day statistics
        cutoff_date = (pd.to_datetime(race_date) - timedelta(days=14)).strftime('%Y-%m-%d')
        recent_records = [r for r in records if r['date'] >= cutoff_date]
        recent_runs = len(recent_records)
        recent_wins = sum(1 for r in recent_records if r['win'])
        
        recent_type_records = [r for r in recent_records if r['type'] == race_type]
        recent_type_runs = len(recent_type_records)
        recent_type_wins = sum(1 for r in recent_type_records if r['win'])
        
        return {
            'trainer_total_runs': total_runs,
            'trainer_total_wins': total_wins,
            'trainer_win_pct': (total_wins / total_runs * 100) if total_runs > 0 else -1.0,
            'trainer_course_runs': course_runs,
            'trainer_course_wins': course_wins,
            'trainer_course_win_pct': (course_wins / course_runs * 100) if course_runs > 0 else -1.0,
            'trainer_distance_runs': distance_runs,
            'trainer_distance_wins': distance_wins,
            'trainer_distance_win_pct': (distance_wins / distance_runs * 100) if distance_runs > 0 else -1.0,
            'trainer_going_runs': going_runs,
            'trainer_going_wins': going_wins,
            'trainer_going_win_pct': (going_wins / going_runs * 100) if going_runs > 0 else -1.0,
            'trainer_14d_runs': recent_runs,
            'trainer_14d_wins': recent_wins,
            'trainer_14d_win_pct': (recent_wins / recent_runs * 100) if recent_runs > 0 else -1.0,
            'trainer_14d_type_runs': recent_type_runs,
            'trainer_14d_type_wins': recent_type_wins,
            'trainer_14d_type_win_pct': (recent_type_wins / recent_type_runs * 100) if recent_type_runs > 0 else -1.0,
        }

    def _get_or_create_mapping(self, mapping_type: str, value: str) -> int:
        """Get or create a mapping for categorical values."""
        if mapping_type not in self.mappings:
            self.mappings[mapping_type] = {}
        
        if value not in self.mappings[mapping_type]:
            self.mappings[mapping_type][value] = len(self.mappings[mapping_type])
        
        return self.mappings[mapping_type][value]

    def _map_hg(self, hg_value) -> int:
        """Map headgear values to integers."""
        if pd.isna(hg_value) or hg_value == '':
            return 0
        
        top_hg = ['p', 't', 'b', 'h', 'v']
        if hg_value in top_hg:
            return top_hg.index(hg_value) + 1
        else:
            return len(top_hg) + 1

    def _update_historical_context(self, df: pd.DataFrame):
        """Update historical context with the day's race results."""
        for _, row in df.iterrows():
            horse_id = row['horse_id']
            jockey_id = row['jockey_id']
            trainer_id = row['trainer_id']
            win = (row['pos'] == 1 or row['pos'] == '1')
            
            record = {
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date']
            }
            
            self.horse_history[horse_id].append(record)
            self.jockey_history[jockey_id].append(record)
            self.trainer_history[trainer_id].append(record)

    def run_incremental_encoding(self):
        """Main method to run incremental encoding."""
        self.logger.info("Starting incremental race data encoding")
        
        # Initialize encoded table
        self.init_encoded_table()
        
        # Get date ranges
        last_encoded_date = self.get_last_encoded_date()
        max_raw_date = self.get_max_raw_date()
        
        if max_raw_date is None:
            self.logger.error("No raw race data found")
            return
        
        if last_encoded_date is None:
            # No encoded data yet, start from configured historical start date
            historical_start = self._get_config_value('common', 'historical_start_date', '2016-01-01')
            start_date = historical_start
            self.logger.info(f"No encoded data found, starting from configured historical start date: {start_date}")
        else:
            # Delete last encoded date (might be incomplete) and resume from there
            self.delete_encoded_date_data(last_encoded_date)
            start_date = last_encoded_date
        
        self.logger.info(f"Encoding from {start_date} to {max_raw_date}")
        
        # Load historical context up to start date
        self.load_historical_context(start_date)
        
        # Process day by day
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(max_raw_date)
        total_encoded = 0
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            encoded_count = self.encode_daily_races(date_str)
            total_encoded += encoded_count
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"Incremental encoding complete. Total records encoded: {total_encoded}")
        
        # Show summary
        if not self.dry_run:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {self.encoded_table}")
                count, min_date, max_date = cursor.fetchone()
                self.logger.info(f"Encoded table now contains {count} records from {min_date} to {max_date}")

def main():
    parser = argparse.ArgumentParser(description='Incremental race data encoding')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    try:
        encoder = IncrementalEncoder(dry_run=args.dry_run)
        
        # Ensure db directory exists
        db_dir = os.path.dirname(encoder.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        encoder.run_incremental_encoding()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
