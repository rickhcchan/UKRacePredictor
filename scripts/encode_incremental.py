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
    python encode_incremental.py [--dry-run] [--force-rebuild]
    
Examples:
    # Encode all new data since last encoded date
    python encode_incremental.py
    
    # Force rebuild from scratch (after schema changes)
    python encode_incremental.py --force-rebuild
    
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
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging, convert_to_24h_time

class IncrementalEncoder:
    def __init__(self, dry_run: bool = False, force_rebuild: bool = False):
        self.dry_run = dry_run
        self.force_rebuild = force_rebuild
        self.logger = setup_logging()
        
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE: No data will be modified")
        
        # Load configuration
        self.config = self._load_config()
        
        # Set database path from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.encoded_table = 'encoded_race_data'
        
        self.logger.info(f"Using database: {self.db_path}")
        self.logger.info(f"Encoded table: {self.encoded_table}")
        
        # Initialize historical tracking for features
        self.horse_history = defaultdict(list)
        self.jockey_history = defaultdict(list)
        self.trainer_history = defaultdict(list)
        
        # Feature mappings (loaded from/saved to database)
        # Note: pattern and going use hardcoded ordinal mappings, not stored in DB
        self.mappings = {
            'sex': {},
            'type': {},
            'course': {}
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

    def init_encoded_table(self, force_recreate: bool = False):
        """Initialize the encoded race data table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # If force_recreate is True, drop and recreate the table to ensure schema matches
            if force_recreate:
                self.logger.info(f"Force recreating {self.encoded_table} table with new schema")
                conn.execute(f"DROP TABLE IF EXISTS {self.encoded_table}")
                conn.commit()
            
            # Create encoded features table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.encoded_table} (
                    race_id INTEGER,
                    horse_id INTEGER,
                    date TEXT,
                    off_time_12h TEXT,
                    off_time_24h TEXT,
                    
                    -- Basic race features
                    course_id INTEGER,
                    type_id INTEGER,
                    class_num INTEGER,
                    pattern_id INTEGER,
                    max_rating INTEGER,
                    min_age INTEGER,
                    max_age INTEGER,
                    sex_rest_id INTEGER,
                    dist_f REAL,
                    going_id INTEGER,
                    ran INTEGER,
                    
                    -- Time features
                    month_sin REAL,
                    month_cos REAL,
                    
                    -- Horse features
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
                    horse_days_since_last_run INTEGER,
                    horse_last_or_rating INTEGER,
                    horse_avg_or_90d REAL,
                    horse_or_trend_direction INTEGER,
                    horse_or_sample_size INTEGER,
                    horse_rating_vs_field_avg REAL,
                    horse_avg_vs_field_avg REAL,
                    horse_rating_percentile REAL,
                    stronger_horses_count INTEGER,
                    
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
                    
                    -- Same-day jockey performance features
                    jockey_day_runs INTEGER,
                    jockey_day_wins INTEGER,
                    jockey_day_win_pct REAL,
                    jockey_day_avg_finish REAL,
                    
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
                    
                    -- Target variables (for training)
                    target_win INTEGER,     -- Position == 1 (win)
                    target_top3 INTEGER,    -- Position <= 3 (place)
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (race_id, horse_id)
                )
            """)
            
            # Create mappings table for categorical features
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_mappings (
                    mapping_type TEXT,
                    value TEXT,
                    mapped_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (mapping_type, value)
                )
            """)
            
            # Create indexes for performance
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_date ON {self.encoded_table}(date)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_horse ON {self.encoded_table}(horse_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.encoded_table}_race ON {self.encoded_table}(race_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_mappings_type ON feature_mappings(mapping_type)")
            
            conn.commit()
            if force_recreate:
                self.logger.info(f"Encoded table {self.encoded_table} recreated with new schema")
            else:
                self.logger.info(f"Encoded table {self.encoded_table} initialized successfully")
            self.logger.info("Feature mappings table initialized successfully")

    def load_mappings_from_db(self):
        """Load existing mappings from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT mapping_type, value, mapped_id FROM feature_mappings ORDER BY mapping_type, mapped_id")
            rows = cursor.fetchall()
            
        # Build mappings dictionary
        mapping_counts = {}
        for mapping_type, value, mapped_id in rows:
            if mapping_type not in self.mappings:
                self.mappings[mapping_type] = {}
            self.mappings[mapping_type][value] = mapped_id
            mapping_counts[mapping_type] = mapping_counts.get(mapping_type, 0) + 1
            
        if mapping_counts:
            self.logger.info(f"Loaded existing mappings: {mapping_counts}")
        else:
            self.logger.info("No existing mappings found, starting fresh")

    def save_mappings_to_db(self):
        """Save current mappings to database."""
        if self.dry_run:
            self.logger.info("[DRY RUN] Would save mappings to database")
            return
            
        new_mappings_inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            for mapping_type, mapping_dict in self.mappings.items():
                for value, mapped_id in mapping_dict.items():
                    # Use INSERT OR IGNORE to avoid conflicts with existing mappings
                    cursor = conn.execute("""
                        INSERT OR IGNORE INTO feature_mappings (mapping_type, value, mapped_id) 
                        VALUES (?, ?, ?)
                    """, (mapping_type, value, mapped_id))
                    # rowcount tells us if the INSERT actually inserted a row (1) or was ignored (0)
                    new_mappings_inserted += cursor.rowcount
            conn.commit()
            
        total_mappings = sum(len(mapping_dict) for mapping_dict in self.mappings.values())
        if new_mappings_inserted > 0:
            self.logger.info(f"Saved {new_mappings_inserted} new mappings to database ({total_mappings} total mappings)")
        else:
            self.logger.info(f"No new mappings to save (all {total_mappings} mappings already existed)")

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
                   pos, date, race_id, or_rating, off, ran
            FROM race_data 
            WHERE date < ? 
            ORDER BY date, race_id
            """
            
            df = pd.read_sql_query(query, conn, params=(up_to_date,))
            
        self.logger.info(f"Loaded {len(df)} historical records for context")
        
        # Build historical tracking
        self.logger.info("Building historical tracking for horses, jockeys, and trainers...")
        progress_interval = 50000  # Log every 50k records
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx > 0 and idx % progress_interval == 0:
                self.logger.info(f"Processed {idx:,} / {len(df):,} historical records ({idx/len(df)*100:.1f}%)")
            
            horse_id = row['horse_id']
            jockey_id = row['jockey_id']
            trainer_id = row['trainer_id']
            win = (row['pos'] == 1 or row['pos'] == '1')
            race_time_24h = convert_to_24h_time(row.get('off', ''))
            
            # Build horse history
            self.horse_history[horse_id].append({
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date'],
                'or_rating': row.get('or_rating')
            })
            
            # Build jockey history (include additional fields for same-day tracking)
            self.jockey_history[jockey_id].append({
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date'],
                'race_time_24h': race_time_24h,
                'pos': row.get('pos'),
                'ran': row.get('ran', 0)
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
        
        self.logger.info(f"Historical tracking complete: {len(self.horse_history)} horses, {len(self.jockey_history)} jockeys, {len(self.trainer_history)} trainers")

    def encode_daily_races(self, date_str: str) -> int:
        """Encode all races for a specific date."""
        self.logger.info(f"Encoding races for {date_str}")
        
        # Load raw data for this date
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT * FROM race_data 
            WHERE date = ? 
            ORDER BY pos
            """
            df = pd.read_sql_query(query, conn, params=(date_str,))
        
        if len(df) == 0:
            self.logger.info(f"No races found for {date_str}")
            return 0
        
        # Convert times to 24H format and sort races chronologically within the day
        df['off_time_24h'] = df['off'].apply(convert_to_24h_time)
        df['off_time_12h'] = df['off']  # Keep original 12H time
        
        # Sort by 24H time first, then by race_id, then by pos
        df = df.sort_values(['off_time_24h', 'race_id', 'pos'])
        
        self.logger.info(f"Found {len(df)} race records for {date_str}, sorted by 24H time for chronological processing")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would encode {len(df)} records for {date_str}")
            # Still simulate the encoding process without saving
            for race_id in df['race_id'].unique():
                race_df = df[df['race_id'] == race_id]
                # Collect field ratings for this race
                field_ratings = self._collect_field_ratings(race_df, date_str)
                
                for _, row in race_df.iterrows():
                    # Just simulate the encoding without saving
                    features = self._encode_single_record(row, field_ratings)
            
            # Update historical context with this day's results
            self._update_historical_context(df)
            return len(df)
        
        # Encode features for all races on this date
        encoded_features = []
        
        # Process each race separately to calculate field ratings
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id]
            
            # Collect field ratings for this race (historical ratings for all horses)
            field_ratings = self._collect_field_ratings(race_df, date_str)
            
            # Encode each horse in this race
            for _, row in race_df.iterrows():
                features = self._encode_single_record(row, field_ratings)
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

    def _encode_single_record(self, row, field_ratings: List[int] = None) -> Dict:
        """Encode features for a single race record."""
        horse_id = row['horse_id']
        jockey_id = row['jockey_id']
        trainer_id = row['trainer_id']
        course = row['course']
        going = row['going']
        dist_f = row['dist_f']
        race_type = row['type']
        race_date = row['date']
        
        # Get historical records (filter to exclude current race date to prevent data leakage)
        horse_records = [r for r in self.horse_history[horse_id] if r['date'] < race_date]
        jockey_records = [r for r in self.jockey_history[jockey_id] if r['date'] < race_date]
        trainer_records = [r for r in self.trainer_history[trainer_id] if r['date'] < race_date]
        
        # Engineer features from raw fields
        class_num = self._extract_class_number(row['class'])
        max_rating = self._extract_max_rating(row['rating_band'])
        min_age, max_age = self._extract_age_range(row['age_band'])
        sex_rest_id = self._get_sex_rest_mapping(row['sex_rest'])  # Use special mapping for sex_rest
        
        # Calculate features
        features = {
            'race_id': row['race_id'],
            'horse_id': horse_id,
            'date': race_date,
            'off_time_12h': row.get('off_time_12h', row.get('off', '')),
            'off_time_24h': row.get('off_time_24h', ''),
            
            # Basic race features
            'course_id': self._get_or_create_mapping('course', course),
            'type_id': self._get_or_create_mapping('type', race_type),
            'class_num': class_num,
            'pattern_id': self._get_or_create_ordinal_mapping('pattern', row.get('pattern', 'Unknown')),
            'max_rating': max_rating,
            'min_age': min_age,
            'max_age': max_age,
            'sex_rest_id': sex_rest_id,
            'dist_f': float(str(dist_f).rstrip('f')) if pd.notna(dist_f) else 0.0,
            'going_id': self._get_or_create_ordinal_mapping('going', going),
            'ran': row['ran'],
            
            # Time features
            'month_sin': np.sin(2 * np.pi * pd.to_datetime(race_date).month / 12),
            'month_cos': np.cos(2 * np.pi * pd.to_datetime(race_date).month / 12),
            
            # Horse features
            'age': row['age'],
            'sex_id': self._get_or_create_mapping('sex', self._normalize_sex(row['sex'])),
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
        }
        
        # Calculate targets
        win_result = 1 if (row['pos'] == 1 or row['pos'] == '1') else 0
        place_result = 1 if (row['pos'] in [1, 2, 3] or row['pos'] in ['1', '2', '3']) else 0
        
        features.update({
            'target_win': win_result,
            'target_top3': place_result
        })
        
        # Calculate historical statistics
        features.update(self._calculate_horse_stats(horse_records, course, going, dist_f, race_date))
        features.update(self._calculate_jockey_stats(jockey_records, course, going, dist_f, race_type, race_date))
        features.update(self._calculate_jockey_day_stats(jockey_id, race_date, row.get('off_time_24h', '')))
        features.update(self._calculate_trainer_stats(trainer_records, course, going, dist_f, race_type, race_date))
        features.update(self._calculate_horse_rating_features(horse_records, race_date, field_ratings))
        
        return features

    def _collect_field_ratings(self, race_df: pd.DataFrame, race_date: str) -> Dict[str, List[float]]:
        """Collect historical ratings for all horses in a race to calculate field strength.
        
        Returns dict with:
        - 'last_ratings': List of last OR ratings for horses that have them
        - 'avg_90d_ratings': List of 90-day average ratings for horses that have them
        """
        last_ratings = []
        avg_90d_ratings = []
        
        for _, row in race_df.iterrows():
            horse_id = row['horse_id']
            
            # Get historical records for this horse (excluding current race date)
            horse_records = [r for r in self.horse_history[horse_id] if r['date'] < race_date]
            
            # Filter to records with valid ratings
            rating_records = [
                r for r in horse_records 
                if r.get('or_rating') is not None 
                and r['or_rating'] != -1 
                and str(r['or_rating']).isdigit()
            ]
            
            if rating_records:
                # Sort by date (most recent first)
                sorted_records = sorted(rating_records, key=lambda x: x['date'], reverse=True)
                
                # Last OR rating (always available if rating_records exists)
                last_rating = int(sorted_records[0]['or_rating'])
                last_ratings.append(float(last_rating))
                
                # 90-day average (only if we have recent data)
                race_date_dt = pd.to_datetime(race_date)
                recent_records = [
                    r for r in sorted_records 
                    if (race_date_dt - pd.to_datetime(r['date'])).days <= 90
                ]
                
                if recent_records:
                    # Calculate 90-day average
                    ratings = [int(r['or_rating']) for r in recent_records]
                    avg_rating = sum(ratings) / len(ratings)
                    avg_90d_ratings.append(avg_rating)
                # Note: If no 90-day data, we don't add anything to avg_90d_ratings
                # This horse will be excluded from 90-day field average calculation
        
        return {
            'last_ratings': last_ratings,
            'avg_90d_ratings': avg_90d_ratings
        }

    def _calculate_horse_rating_features(self, records: List[Dict], race_date: str, field_ratings: Dict[str, List[float]] = None) -> Dict:
        """Calculate horse rating trend features from historical races."""
        # Filter to only races BEFORE current race date (no same-day races)
        race_date_dt = pd.to_datetime(race_date)
        prior_records = [
            r for r in records 
            if pd.to_datetime(r['date']) < race_date_dt 
            and r.get('or_rating') is not None 
            and r['or_rating'] != -1
            and str(r['or_rating']).isdigit()
        ]
        
        if not prior_records:
            base_features = {
                'horse_last_or_rating': 0,
                'horse_avg_or_90d': 0.0,
                'horse_or_trend_direction': 0,
                'horse_or_sample_size': 0
            }
            
            # Add race-contextual features (all zeros for horses with no rating history)
            if field_ratings:
                last_field_ratings = field_ratings.get('last_ratings', [])
                stronger_horses = len(last_field_ratings) if last_field_ratings else 0
                base_features.update({
                    'horse_rating_vs_field_avg': 0.0,
                    'horse_avg_vs_field_avg': 0.0,
                    'horse_rating_percentile': 0.0,
                    'stronger_horses_count': stronger_horses
                })
            else:
                base_features.update({
                    'horse_rating_vs_field_avg': 0.0,
                    'horse_avg_vs_field_avg': 0.0,
                    'horse_rating_percentile': 0.0,
                    'stronger_horses_count': 0
                })
            
            return base_features
        
        # Sort by date (most recent first)
        sorted_records = sorted(prior_records, key=lambda x: x['date'], reverse=True)
        
        # Most recent rating (convert to int)
        last_or = int(sorted_records[0]['or_rating'])
        
        # 90-day window for recent form
        recent_records = [
            r for r in sorted_records 
            if (race_date_dt - pd.to_datetime(r['date'])).days <= 90
        ]
        
        if len(recent_records) == 0:
            avg_90d = 0.0
            trend_direction = 0
            sample_size = 0
        elif len(recent_records) == 1:
            avg_90d = float(int(recent_records[0]['or_rating']))
            trend_direction = 0  # Can't determine trend with 1 race
            sample_size = 1
        else:
            # Multiple races available (convert to int)
            ratings = [int(r['or_rating']) for r in recent_records]
            avg_90d = sum(ratings) / len(ratings)
            
            # Trend: compare most recent vs older average
            recent_rating = ratings[0]  # Most recent
            older_ratings = ratings[1:]  # Rest
            older_avg = sum(older_ratings) / len(older_ratings)
            
            # Trend direction (2-point threshold for significance)
            if recent_rating > older_avg + 2:
                trend_direction = 1  # Improving
            elif recent_rating < older_avg - 2:
                trend_direction = -1  # Declining
            else:
                trend_direction = 0  # Stable
            
            sample_size = len(recent_records)
        
        base_features = {
            'horse_last_or_rating': last_or,
            'horse_avg_or_90d': avg_90d,
            'horse_or_trend_direction': trend_direction,
            'horse_or_sample_size': sample_size
        }
        
        # Add race-contextual features if field ratings are provided
        if field_ratings:
            # Calculate field averages separately for each rating type
            last_field_ratings = field_ratings.get('last_ratings', [])
            avg_90d_field_ratings = field_ratings.get('avg_90d_ratings', [])
            
            # Feature 1: Last OR rating vs field average of last OR ratings (apples to apples)
            if last_field_ratings and last_or > 0:
                last_field_avg = sum(last_field_ratings) / len(last_field_ratings)
                last_or_vs_field = last_or - last_field_avg
                
                # Calculate percentile using last OR rating against other last OR ratings
                weaker_count = sum(1 for r in last_field_ratings if last_or > r)
                percentile = (weaker_count / len(last_field_ratings)) * 100
                
                # Count of horses with higher last OR ratings
                stronger_horses = sum(1 for r in last_field_ratings if r > last_or)
            else:
                last_or_vs_field = 0.0
                percentile = 0.0
                stronger_horses = len(last_field_ratings) if last_field_ratings else 0
            
            # Feature 2: 90-day average vs field average of 90-day averages (apples to apples)
            if avg_90d_field_ratings and avg_90d > 0:
                avg_90d_field_avg = sum(avg_90d_field_ratings) / len(avg_90d_field_ratings)
                avg_90d_vs_field = avg_90d - avg_90d_field_avg
            else:
                avg_90d_vs_field = 0.0
            
            base_features.update({
                'horse_rating_vs_field_avg': last_or_vs_field,
                'horse_avg_vs_field_avg': avg_90d_vs_field,
                'horse_rating_percentile': percentile,
                'stronger_horses_count': stronger_horses
            })
        else:
            # No field ratings provided (shouldn't happen in normal encoding)
            base_features.update({
                'horse_rating_vs_field_avg': 0.0,
                'horse_avg_vs_field_avg': 0.0,
                'horse_rating_percentile': 0.0,
                'stronger_horses_count': 0
            })
        
        return base_features

    def _calculate_horse_stats(self, records: List[Dict], course: str, going: str, dist_f: str, race_date: str) -> Dict:
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
        
        # Calculate days since last run
        # Sort records by date descending to get most recent first
        if records:
            sorted_records = sorted(records, key=lambda x: x['date'], reverse=True)
            last_run_date = sorted_records[0]['date']  # Most recent run
            race_date_dt = pd.to_datetime(race_date)
            last_run_date_dt = pd.to_datetime(last_run_date)
            days_since_last_run = (race_date_dt - last_run_date_dt).days
        else:
            # First run for this horse
            days_since_last_run = -1
        
        # Calculate rating trend features
        rating_features = self._calculate_horse_rating_features(records, race_date)
        
        return {
            'horse_total_runs': total_runs,
            'horse_win_pct': (total_wins / total_runs * 100) if total_runs > 0 else -1.0,
            'horse_course_runs': course_runs,
            'horse_course_win_pct': (course_wins / course_runs * 100) if course_runs > 0 else -1.0,
            'horse_distance_runs': distance_runs,
            'horse_distance_win_pct': (distance_wins / distance_runs * 100) if distance_runs > 0 else -1.0,
            'horse_going_runs': going_runs,
            'horse_going_win_pct': (going_wins / going_runs * 100) if going_runs > 0 else -1.0,
            'horse_days_since_last_run': days_since_last_run,
            **rating_features,  # Add rating features
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
        
        # 14-day statistics (exclude current race date)
        cutoff_date = (pd.to_datetime(race_date) - timedelta(days=14)).strftime('%Y-%m-%d')
        recent_records = [r for r in records if r['date'] >= cutoff_date and r['date'] < race_date]
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

    def _calculate_jockey_day_stats(self, jockey_id: int, race_date: str, race_time_24h: str) -> Dict:
        """Calculate jockey's performance in earlier races on the same day."""
        
        # Get all races for this jockey on this date with earlier times
        same_day_earlier_races = [
            r for r in self.jockey_history[jockey_id] 
            if r['date'] == race_date and r.get('race_time_24h', '') < race_time_24h
        ]
        
        if not same_day_earlier_races:
            return {
                'jockey_day_runs': 0,
                'jockey_day_wins': 0,
                'jockey_day_win_pct': -1.0,
                'jockey_day_avg_finish': -1.0
            }
        
        runs = len(same_day_earlier_races)
        wins = sum(1 for r in same_day_earlier_races if r['win'])
        
        # Calculate average finish position (only for races where position is valid)
        positions = []
        for r in same_day_earlier_races:
            pos = r.get('pos')
            if pos is not None and str(pos).isdigit():
                positions.append(int(pos))
        
        avg_finish = sum(positions) / len(positions) if positions else -1.0
        
        return {
            'jockey_day_runs': runs,
            'jockey_day_wins': wins,
            'jockey_day_win_pct': (wins / runs * 100) if runs > 0 else -1.0,
            'jockey_day_avg_finish': avg_finish
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
        
        # 14-day statistics (exclude current race date)
        cutoff_date = (pd.to_datetime(race_date) - timedelta(days=14)).strftime('%Y-%m-%d')
        recent_records = [r for r in records if r['date'] >= cutoff_date and r['date'] < race_date]
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

    def _get_or_create_ordinal_mapping(self, mapping_type: str, value: str) -> int:
        """Get or create an ordinal mapping for values with inherent ordering."""
        if mapping_type == 'pattern':
            return self._map_pattern(value)
        elif mapping_type == 'going':
            return self._map_going(value)
        else:
            # Fallback to categorical mapping
            return self._get_or_create_mapping(mapping_type, value)

    def _map_pattern(self, pattern_value) -> int:
        """Map pattern values to ordinal integers (1=highest prestige, 5=lowest)."""
        if pd.isna(pattern_value) or pattern_value == '' or pattern_value == 'Unknown':
            return 5
        
        # Handle both Group and Grade races with same ordinal values
        pattern_ordinal_map = {
            'Group 1': 1,    'Grade 1': 1,     # Highest prestige
            'Group 2': 2,    'Grade 2': 2,     # Second tier
            'Group 3': 3,    'Grade 3': 3,     # Third tier
            'Listed': 4,                       # Listed races
        }
        
        return pattern_ordinal_map.get(pattern_value, 5)  # Default to 5 for unknown

    def _map_going(self, going_value) -> int:
        """Map going values to ordinal integers (1=firm, higher=softer, -1=unknown)."""
        if pd.isna(going_value) or going_value == '':
            return -1
        
        # Ordinal mapping from firm to soft ground
        going_ordinal_map = {
            'Firm': 1,
            'Good To Firm': 2,
            'Good': 3,
            'Standard': 3,           # Equivalent to Good for AW
            'Good To Soft': 4,
            'Standard To Slow': 4,   # Equivalent to Good To Soft for AW
            'Soft': 5,
            'Heavy': 6,
            'Slow': 5,               # Equivalent to Soft for AW
            'Standard To Fast': 2,   # Equivalent to Good To Firm for AW
            'None': -1
        }
        
        return going_ordinal_map.get(going_value, -1)  # Default to -1 for unknown

    def _map_hg(self, hg_value) -> int:
        """Map headgear values to integers."""
        if pd.isna(hg_value) or hg_value == '':
            return 0
        
        top_hg = ['p', 't', 'b', 'h', 'v']
        if hg_value in top_hg:
            return top_hg.index(hg_value) + 1
        else:
            return len(top_hg) + 1

    def _normalize_sex(self, sex_value) -> str:
        """Normalize sex values to single character codes."""
        if pd.isna(sex_value) or sex_value == '':
            return 'Unknown'
        
        sex_str = str(sex_value).strip().lower()
        
        # Handle full word to single character mapping
        sex_normalization_map = {
            # Single characters (already normalized)
            'g': 'G',
            'f': 'F', 
            'm': 'M',
            'c': 'C',
            'h': 'H',
            'r': 'R',
            
            # Full words from racecard
            'gelding': 'G',
            'filly': 'F',
            'mare': 'F',  # Mare is also female
            'colt': 'M',
            'horse': 'M',  # Male horse
            'stallion': 'M',
            'colt or gelding': 'C',  # Could be either
            'colt or filly': 'C',    # Could be either
            'rig': 'R'
        }
        
        return sex_normalization_map.get(sex_str, 'Unknown')

    def _extract_class_number(self, class_value) -> int:
        """Extract numeric class value from class string."""
        if pd.isna(class_value) or class_value == '':
            return -1
        
        # Handle different class formats
        class_str = str(class_value).strip()
        
        # Extract number from class string (e.g., "Class 4" -> 4, "3" -> 3)
        match = re.search(r'(\d+)', class_str)
        if match:
            return int(match.group(1))
        
        return -1

    def _extract_max_rating(self, rating_band) -> int:
        """Extract maximum rating from rating band (e.g., '0-100' -> 100)."""
        if pd.isna(rating_band) or rating_band == '':
            return -1
        
        # Most rating bands are in format "0-100"
        if '-' in str(rating_band):
            parts = str(rating_band).split('-')
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        
        return -1

    def _extract_age_range(self, age_band) -> tuple:
        """Extract min and max age from age band (e.g., '3-5yo' -> (3, 5), '3yo+' -> (3, 999))."""
        if pd.isna(age_band) or age_band == '':
            return -1, -1
        
        age_str = str(age_band).strip().lower()
        
        # Handle "3yo+" format (means 3 or above)
        if '+' in age_str:
            match = re.search(r'(\d+)', age_str)
            if match:
                min_age = int(match.group(1))
                return min_age, 999  # Use 999 for unlimited
        
        # Handle range format like "3-5yo"
        if '-' in age_str:
            match = re.search(r'(\d+)-(\d+)', age_str)
            if match:
                min_age = int(match.group(1))
                max_age = int(match.group(2))
                return min_age, max_age
        
        # Handle single age like "3yo"
        match = re.search(r'(\d+)', age_str)
        if match:
            age = int(match.group(1))
            return age, age
        
        return -1, -1

    def _update_historical_context(self, df: pd.DataFrame):
        """Update historical context with the day's race results."""
        for _, row in df.iterrows():
            horse_id = row['horse_id']
            jockey_id = row['jockey_id']
            trainer_id = row['trainer_id']
            win = (row['pos'] == 1 or row['pos'] == '1')
            race_time_24h = row.get('off_time_24h', convert_to_24h_time(row.get('off', '')))
            
            horse_record = {
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date'],
                'or_rating': row.get('or_rating')
            }
            
            jockey_record = {
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date'],
                'race_time_24h': race_time_24h,
                'pos': row.get('pos'),
                'ran': row.get('ran', 0)
            }
            
            trainer_record = {
                'course': row['course'],
                'going': row['going'],
                'dist_f': row['dist_f'],
                'type': row['type'],
                'win': win,
                'date': row['date'],
                'or_rating': row.get('or_rating')
            }
            
            self.horse_history[horse_id].append(horse_record)
            self.jockey_history[jockey_id].append(jockey_record)
            self.trainer_history[trainer_id].append(trainer_record)

    def run_incremental_encoding(self):
        """Main method to run incremental encoding."""
        self.logger.info("Starting incremental race data encoding")
        
        # Initialize encoded table and load existing mappings
        self.init_encoded_table(force_recreate=self.force_rebuild)
        
        self.load_mappings_from_db()
        
        # Get date ranges
        last_encoded_date = self.get_last_encoded_date()
        max_raw_date = self.get_max_raw_date()
        
        if max_raw_date is None:
            self.logger.error("No raw race data found")
            return
        
        if last_encoded_date is None or self.force_rebuild:
            # No encoded data yet or force rebuild, start from configured historical start date
            historical_start = self._get_config_value('common', 'historical_start_date', '2016-01-01')
            start_date = historical_start
            if self.force_rebuild:
                self.logger.info(f"Force rebuild requested, starting fresh from configured historical start date: {start_date}")
            else:
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
        
        # Save mappings to database
        self.save_mappings_to_db()
        
        self.logger.info(f"Incremental encoding complete. Total records encoded: {total_encoded}")
        
        # Show summary
        if not self.dry_run:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {self.encoded_table}")
                count, min_date, max_date = cursor.fetchone()
                self.logger.info(f"Encoded table now contains {count} records from {min_date} to {max_date}")
                
                # Show mapping statistics
                cursor = conn.execute("SELECT mapping_type, COUNT(*) FROM feature_mappings GROUP BY mapping_type")
                mapping_stats = cursor.fetchall()
                if mapping_stats:
                    self.logger.info("Final mapping counts:")
                    for mapping_type, count in mapping_stats:
                        self.logger.info(f"  {mapping_type}: {count} unique values")

    def _get_sex_rest_mapping(self, sex_rest_value) -> int:
        """Map sex_rest values using related sex IDs since they share the same values."""
        if pd.isna(sex_rest_value) or sex_rest_value == '':
            return self._get_or_create_mapping('sex', 'Unknown')
        
        sex_rest_str = str(sex_rest_value).strip()
        
        # Handle compound values like "F & M" or "C & G"
        if '&' in sex_rest_str:
            # For compound values, use the first sex type for mapping consistency
            first_sex = sex_rest_str.split('&')[0].strip()
            return self._get_or_create_mapping('sex', first_sex)
        else:
            # Single sex value - use the same sex mapping
            return self._get_or_create_mapping('sex', sex_rest_str)

def main():
    parser = argparse.ArgumentParser(description='Incremental race data encoding')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--force-rebuild', 
                       action='store_true',
                       help='Force rebuild: truncate encoded table and start fresh (use after schema changes)')
    
    args = parser.parse_args()
    
    try:
        encoder = IncrementalEncoder(
            dry_run=args.dry_run,
            force_rebuild=args.force_rebuild
        )
        
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
