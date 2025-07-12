"""
Racecard preparation script with feature engineering.

This script handles the complete racecard preparation workflow:
1. Download today's racecard using rpscrape (saved to raw/ subdirectory)
2. Load historical context from database for feature engineering
3. Apply same feature engineering as training data
4. Save prepared racecard ready for prediction
5. Keep raw racecard file for debugging and analysis

The script requires historical data to be available for accurate 14-day features.
Run update_race_data.py first to ensure yesterday's results are available.

Raw files are saved to: data/prediction/raw/{date}.json
Processed files are saved to: data/prediction/racecard_{date}_prepared.csv

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python prepare_racecard.py
    
Examples:
    # Prepare today's racecard
    python prepare_racecard.py
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import argparse
import configparser
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging, convert_to_24h_time

class RacecardPreparer:
    def __init__(self, date: str = None, dry_run: bool = False):
        self.target_date = date or datetime.now().strftime('%Y-%m-%d')
        self.dry_run = dry_run
        self.logger = setup_logging()
        
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE: No files will be modified")
        
        # Load configuration
        self.config = self._load_config()
        
        # Set paths from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.rpscrape_dir = self._get_config_value('common', 'rpscrape_dir')
        self.timeout = int(self._get_config_value('common', 'timeout', '30'))
        self.prediction_dir = Path(self._get_config_value('common', 'data_dir', 'data')) / 'prediction'
        
        self.logger.info(f"Preparing racecard for date: {self.target_date}")
        self.logger.info(f"Using database: {self.db_path}")
        self.logger.info(f"Using rpscrape at: {self.rpscrape_dir}")
        self.logger.info(f"Using timeout: {self.timeout}s")
        
        # Create directory if it doesn't exist
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize historical tracking for features
        self.horse_history = defaultdict(list)
        self.jockey_history = defaultdict(list)
        self.trainer_history = defaultdict(list)
        
        # Feature mappings (loaded from database)
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

    def check_historical_data_availability(self) -> bool:
        """Check if sufficient historical data is available for feature engineering."""
        # Need at least yesterday's data for 14-day features
        yesterday = (pd.to_datetime(self.target_date) - timedelta(days=1)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM race_data")
            max_date = cursor.fetchone()[0]
            
        if max_date is None:
            self.logger.error("No historical race data found in database")
            return False
        
        if max_date < yesterday:
            self.logger.warning(f"Historical data only available up to {max_date}, need data up to {yesterday}")
            self.logger.warning("Consider running: python update_race_data.py")
            return False
        
        self.logger.info(f"Historical data available up to {max_date} ✓")
        return True

    def _auto_update_historical_data(self, target_date: str) -> bool:
        """Automatically update historical data if auto_update is enabled."""
        try:
            self.logger.info("Running update_race_data.py to get latest historical data...")
            result = subprocess.run([
                sys.executable, "update_race_data.py", "--end-date", target_date
            ], cwd=script_dir, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode == 0:
                self.logger.info("Successfully updated historical data")
                return True
            else:
                self.logger.error(f"Failed to update historical data: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while updating historical data")
            return False
        except Exception as e:
            self.logger.error(f"Error running update_race_data.py: {e}")
            return False

    def download_racecard(self) -> bool:
        """Download today's racecard using rpscrape racecards.py."""
        racecard_file = self.prediction_dir / f"{self.target_date}.json"
        
        if racecard_file.exists():
            self.logger.info(f"Racecard file already exists: {racecard_file}")
            return True
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would download racecard for {self.target_date} to: {racecard_file}")
            return True
        
        try:
            # Get the racecards.py script (not racecard.py)
            rpscrape_dir = Path(self.rpscrape_dir)
            racecards_script = rpscrape_dir / "scripts" / "racecards.py"
            scripts_dir = rpscrape_dir / "scripts"
            
            if not racecards_script.exists():
                self.logger.error(f"Could not find racecards.py in {racecards_script}")
                return False
            
            # Determine whether to fetch today or tomorrow's racecard
            from datetime import datetime, timedelta
            today = datetime.today().strftime('%Y-%m-%d')
            tomorrow = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if self.target_date == today:
                day_arg = "today"
            elif self.target_date == tomorrow:
                day_arg = "tomorrow"
            else:
                self.logger.error(f"rpscrape racecards.py only supports 'today' or 'tomorrow', not {self.target_date}")
                return False
            
            self.logger.info(f"Running rpscrape racecards.py to download racecard for {self.target_date} (using '{day_arg}' argument)")
            
            # Run rpscrape racecards script with day parameter
            # Run from scripts subdirectory so it can find relative imports and files
            result = subprocess.run([
                sys.executable, "racecards.py", day_arg
            ], cwd=scripts_dir, capture_output=True, text=True, timeout=self.timeout)
            
            # Display the console output for debugging
            if result.stdout:
                self.logger.info(f"racecards.py stdout:\n{result.stdout}")
            if result.stderr:
                self.logger.info(f"racecards.py stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                self.logger.error(f"rpscrape racecards.py failed with return code {result.returncode}")
                return False
            
            # Copy downloaded file to our prediction directory
            # rpscrape saves racecards in racecards/ subdirectory
            source_file = rpscrape_dir / "racecards" / f"{self.target_date}.json"
            if source_file.exists():
                shutil.copy2(source_file, racecard_file)
                self.logger.info(f"Racecard downloaded and saved to: {racecard_file}")
                return True
            else:
                self.logger.error(f"Expected racecard file not found: {source_file}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while downloading racecard")
            return False
        except Exception as e:
            self.logger.error(f"Error downloading racecard: {e}")
            return False

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
            self.logger.warning("No existing mappings found - may need to run encode_incremental.py first")

    def load_historical_context(self):
        """Load historical context up to target date for accurate feature calculation."""
        self.logger.info(f"Loading historical context up to {self.target_date}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Load all historical race data up to the target date (exclusive)
            query = """
            SELECT horse_id, jockey_id, trainer_id, course, going, dist_f, type, 
                   pos, date, race_id
            FROM race_data 
            WHERE date < ? 
            ORDER BY date, race_id
            """
            
            df = pd.read_sql_query(query, conn, params=(self.target_date,))
            
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
        
        self.logger.info(f"Historical tracking complete: {len(self.horse_history)} horses, {len(self.jockey_history)} jockeys, {len(self.trainer_history)} trainers")

    def engineer_features(self, racecard_data: dict) -> pd.DataFrame:
        """Apply feature engineering to racecard data using database mappings and historical context."""
        self.logger.info("Engineering features for racecard data")
        
        rows = []
        courses = racecard_data.get('GB', {})
        
        # Collect all races with their times for proper sorting
        all_races = []
        for course_name, races in courses.items():
            for race_time, race_info in races.items():
                # Get the actual off_time from race_info, fallback to race_time key
                actual_time = race_info.get('off_time', race_time)
                converted_time = convert_to_24h_time(actual_time)
                all_races.append((course_name, race_time, race_info, converted_time))
        
        # Sort races by converted 24-hour time to ensure chronological processing
        all_races.sort(key=lambda x: x[3])  # Sort by converted_time
        self.logger.info(f"Processing {len(all_races)} races in chronological order")
        
        for course_name, race_time, race_info, converted_time in all_races:
            runners = race_info.get("runners", [])
            
            for runner in runners:
                # Engineer features using same logic as encode_incremental.py
                features = self._engineer_single_runner(runner, race_info, course_name, race_time)
                if features:
                    rows.append(features)
        
        if not rows:
            self.logger.warning("No runners found in racecard data")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        self.logger.info(f"Engineered features for {len(df)} runners from {df.get('race_id', pd.Series()).nunique()} races")
        
        return df

    def _engineer_single_runner(self, runner: dict, race_info: dict, course_name: str, race_time: str) -> Optional[Dict]:
        """Engineer features for a single runner - similar to encode_incremental._encode_single_record."""
        try:
            horse_id = int(runner.get('horse_id', -1))
            jockey_id = int(runner.get('jockey_id', -1))
            trainer_id = int(runner.get('trainer_id', -1))
            
            if horse_id == -1:
                self.logger.warning(f"Missing horse_id for runner: {runner.get('name', 'Unknown')}")
                return None
            
            # Get historical records (filter to exclude current date to prevent data leakage)
            horse_records = [r for r in self.horse_history.get(horse_id, []) if r['date'] < self.target_date]
            jockey_records = [r for r in self.jockey_history.get(jockey_id, []) if r['date'] < self.target_date]
            trainer_records = [r for r in self.trainer_history.get(trainer_id, []) if r['date'] < self.target_date]
            
            # Basic race information
            course = course_name
            going = race_info.get('going', '')
            dist_f = race_info.get('distance_f', 0)
            race_type = race_info.get('type', '')
            
            # Engineer features from raw fields using same methods as encode_incremental
            class_num = self._extract_class_number(race_info.get('race_class', ''))
            max_rating = self._extract_max_rating(race_info.get('rating_band', ''))
            min_age, max_age = self._extract_age_range(race_info.get('age_band', ''))
            sex_rest_id = self._get_sex_rest_mapping(race_info.get('sex_rest', ''))
            
            # Calculate features
            features = {
                'race_id': int(race_info.get('race_id', 0)),
                'horse_id': horse_id,
                'date': self.target_date,
                
                # Basic race features
                'course_id': self._get_mapping_value('course', course),
                'type_id': self._get_mapping_value('type', race_type),
                'class_num': class_num,
                'pattern_id': self._map_pattern(race_info.get('pattern', 'Unknown')),
                'max_rating': max_rating,
                'min_age': min_age,
                'max_age': max_age,
                'sex_rest_id': sex_rest_id,
                'dist_f': float(str(dist_f).rstrip('f')) if pd.notna(dist_f) else 0.0,
                'going_id': self._map_going(going),
                'ran': len(race_info.get("runners", [])),
                
                # Time features
                'month_sin': np.sin(2 * np.pi * pd.to_datetime(self.target_date).month / 12),
                'month_cos': np.cos(2 * np.pi * pd.to_datetime(self.target_date).month / 12),
                
                # Horse features
                'age': int(runner.get('age', 0)),
                'sex_id': self._get_mapping_value('sex', self._normalize_sex(runner.get('sex_code', ''))),
                'lbs': int(runner.get('lbs', 0)),
                'hg': self._map_hg(runner.get('headgear', '')),
                'draw': int(runner.get('draw', -1)),
                
                # Jockey features
                'jockey_id': jockey_id,
                
                # Trainer features
                'trainer_id': trainer_id,
                
                # Ratings
                'or_rating': int(runner.get('ofr', -1)) if runner.get('ofr') is not None else -1,
                'rpr': int(runner.get('rpr', -1)) if runner.get('rpr') is not None else -1,
                'ts': int(runner.get('ts', -1)) if runner.get('ts') is not None else -1,
                
                # Bloodline features (would need bloodline mapping - for now set to -1)
                'sire_id': -1,  # TODO: Add bloodline lookup
                'dam_id': -1,
                'damsire_id': -1,
                'owner_id': -1,  # TODO: Add owner mapping
                
                # Additional metadata for display - use off_time from race_info and convert to 24h
                'horse_name': runner.get('name', ''),
                'course': course,
                'time': convert_to_24h_time(race_info.get('off_time', race_time)),
            }
            
            # Calculate historical statistics
            features.update(self._calculate_horse_stats(horse_records, course, going, dist_f, self.target_date))
            features.update(self._calculate_jockey_stats(jockey_records, course, going, dist_f, race_type, self.target_date))
            features.update(self._calculate_trainer_stats(trainer_records, course, going, dist_f, race_type, self.target_date))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features for runner {runner.get('name', 'Unknown')}: {e}")
            return None

    def save_prepared_racecard(self, df: pd.DataFrame):
        """Save prepared racecard with engineered features."""
        output_file = self.prediction_dir / f"racecard_{self.target_date}_prepared.csv"
        raw_file = self.prediction_dir / f"{self.target_date}.json"
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would save {len(df)} prepared records to: {output_file}")
            self.logger.info(f"[DRY RUN] Raw racecard would be preserved at: {raw_file}")
        else:
            df.to_csv(output_file, index=False)
            self.logger.info(f"✓ Saved {len(df)} prepared records to: {output_file}")
            self.logger.info(f"✓ Raw racecard will be preserved at: {raw_file}")

    def cleanup_raw_racecard(self):
        """Keep raw racecard file for debugging and analysis."""
        raw_file = self.prediction_dir / f"{self.target_date}.json"
        
        if raw_file.exists():
            self.logger.info(f"✓ Raw racecard kept for analysis: {raw_file}")
        else:
            self.logger.warning(f"Raw racecard file not found: {raw_file}")

    def prepare_racecard(self):
        """Main method to run racecard preparation."""
        self.logger.info(f"Starting racecard preparation for {self.target_date}")
        
        # Check historical data availability
        if not self.check_historical_data_availability():
            self.logger.error("Insufficient historical data for feature engineering")
            return False
        
        # Load mappings from database
        self.load_mappings_from_db()
        
        # Download racecard
        if not self.download_racecard():
            self.logger.error("Failed to download racecard")
            return False
        
        # Load racecard data
        racecard_file = self.prediction_dir / f"{self.target_date}.json"
        try:
            with open(racecard_file, 'r', encoding='utf-8') as f:
                racecard_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load racecard data: {e}")
            return False
        
        # Load historical context
        self.load_historical_context()
        
        # Engineer features
        df = self.engineer_features(racecard_data)
        if df.empty:
            self.logger.error("No features engineered from racecard")
            return False
        
        # Save prepared racecard
        self.save_prepared_racecard(df)
        
        # Confirm raw file preservation (don't delete it)
        self.cleanup_raw_racecard()
        
        self.logger.info(f"✓ Racecard preparation complete for {self.target_date}")
        return True

    # Feature engineering methods (same as encode_incremental.py)
    def _get_mapping_value(self, mapping_type: str, value: str) -> int:
        """Get mapping value from database mappings."""
        if mapping_type not in self.mappings:
            return -1
        return self.mappings[mapping_type].get(value, -1)

    def _get_sex_rest_mapping(self, sex_rest_value) -> int:
        """Map sex_rest values using related sex IDs since they share the same values."""
        if pd.isna(sex_rest_value) or sex_rest_value == '':
            return self._get_mapping_value('sex', 'Unknown')
        
        sex_rest_str = str(sex_rest_value).strip()
        
        # Handle compound values like "F & M" or "C & G"
        if '&' in sex_rest_str:
            # For compound values, use the first sex type for mapping consistency
            first_sex = sex_rest_str.split('&')[0].strip()
            return self._get_mapping_value('sex', first_sex)
        else:
            # Single sex value - use the same sex mapping
            return self._get_mapping_value('sex', sex_rest_str)

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
        if records:
            sorted_records = sorted(records, key=lambda x: x['date'], reverse=True)
            last_run_date = sorted_records[0]['date']  # Most recent run
            race_date_dt = pd.to_datetime(race_date)
            last_run_date_dt = pd.to_datetime(last_run_date)
            days_since_last_run = (race_date_dt - last_run_date_dt).days
        else:
            days_since_last_run = -1
        
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
            'horse_days_since_last_run': days_since_last_run,
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

    # Same helper methods as encode_incremental.py
    def _map_pattern(self, pattern_value) -> int:
        """Map pattern values to ordinal integers (1=highest prestige, 5=lowest)."""
        if pd.isna(pattern_value) or pattern_value == '' or pattern_value == 'Unknown':
            return 5
        
        pattern_ordinal_map = {
            'Group 1': 1,    'Grade 1': 1,
            'Group 2': 2,    'Grade 2': 2,
            'Group 3': 3,    'Grade 3': 3,
            'Listed': 4,
        }
        
        return pattern_ordinal_map.get(pattern_value, 5)

    def _map_going(self, going_value) -> int:
        """Map going values to ordinal integers (1=firm, higher=softer, -1=unknown)."""
        if pd.isna(going_value) or going_value == '':
            return -1
        
        going_ordinal_map = {
            'Firm': 1,
            'Good To Firm': 2,
            'Good': 3,
            'Standard': 3,
            'Good To Soft': 4,
            'Standard To Slow': 4,
            'Soft': 5,
            'Heavy': 6,
            'Slow': 5,
            'Standard To Fast': 2,
            'None': -1
        }
        
        return going_ordinal_map.get(going_value, -1)

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
        
        sex_normalization_map = {
            'g': 'G', 'f': 'F', 'm': 'M', 'c': 'C', 'h': 'H', 'r': 'R',
            'gelding': 'G', 'filly': 'F', 'mare': 'F', 'colt': 'M',
            'horse': 'M', 'stallion': 'M', 'colt or gelding': 'C',
            'colt or filly': 'C', 'rig': 'R'
        }
        
        return sex_normalization_map.get(sex_str, 'Unknown')

    def _extract_class_number(self, class_value) -> int:
        """Extract numeric class value from class string."""
        if pd.isna(class_value) or class_value == '':
            return -1
        
        import re
        match = re.search(r'(\d+)', str(class_value).strip())
        if match:
            return int(match.group(1))
        
        return -1

    def _extract_max_rating(self, rating_band) -> int:
        """Extract maximum rating from rating band (e.g., '0-100' -> 100)."""
        if pd.isna(rating_band) or rating_band == '':
            return -1
        
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
        
        import re
        if '+' in age_str:
            match = re.search(r'(\d+)', age_str)
            if match:
                min_age = int(match.group(1))
                return min_age, 999
        
        if '-' in age_str:
            match = re.search(r'(\d+)-(\d+)', age_str)
            if match:
                min_age = int(match.group(1))
                max_age = int(match.group(2))
                return min_age, max_age
        
        match = re.search(r'(\d+)', age_str)
        if match:
            age = int(match.group(1))
            return age, age
        
        return -1, -1

def main():
    parser = argparse.ArgumentParser(description='Prepare racecard with feature engineering')
    parser.add_argument('--date', 
                       type=str,
                       help='Target date for racecard preparation (YYYY-MM-DD). Only today or tomorrow supported, defaults to today')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without making actual changes')
    
    args = parser.parse_args()
    
    try:
        preparer = RacecardPreparer(date=args.date, dry_run=args.dry_run)
        
        success = preparer.prepare_racecard()
        
        if success:
            print(f"✓ Racecard preparation completed successfully for {preparer.target_date}")
        else:
            print(f"✗ Racecard preparation failed for {preparer.target_date}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
