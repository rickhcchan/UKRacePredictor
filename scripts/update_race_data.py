"""
Incremental race data update script.

This script handles the complete workflow:
1. Check DB to get max date (last complete day)
2. Always delete the last day's data (handle incomplete downloads)
3. Download data from last_date to current_date using rpscrape
4. Parse and insert new CSV data into DB

The script automatically processes from the last date in DB to today.
Can handle years of data efficiently in a single execution.

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python update_race_data.py [--dry-run]
    
Examples:
    # Update to current date
    python update_race_data.py
    
    # Test run without making changes
    python update_race_data.py --dry-run
"""

import os
import sys
import sqlite3
import pandas as pd
import subprocess
import argparse
import configparser
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Tuple

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging

class RaceDataUpdater:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Set paths from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.rpscrape_dir = self._get_config_value('common', 'rpscrape_dir')
        self.timeout = int(self._get_config_value('common', 'timeout', '30'))
        self.rpscrape_path = Path(self.rpscrape_dir) / 'scripts' / 'rpscrape.py'
        
        self.logger.info(f"Using database: {self.db_path}")
        self.logger.info(f"Using rpscrape directory: {self.rpscrape_dir}")
        self.logger.info(f"Using timeout: {self.timeout}s")
        self.logger.info(f"Using rpscrape script: {self.rpscrape_path}")

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

    def init_database(self):
        """Initialize the database with race data table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create table based on the CSV structure we've seen
            conn.execute("""
                CREATE TABLE IF NOT EXISTS race_data (
                    date TEXT,
                    region TEXT,
                    course_id INTEGER,
                    course TEXT,
                    race_id INTEGER,
                    off TEXT,
                    race_name TEXT,
                    type TEXT,
                    class TEXT,
                    pattern TEXT,
                    rating_band TEXT,
                    age_band TEXT,
                    sex_rest TEXT,
                    dist TEXT,
                    dist_f TEXT,
                    dist_m REAL,
                    going TEXT,
                    ran INTEGER,
                    num INTEGER,
                    pos INTEGER,
                    draw INTEGER,
                    ovr_btn REAL,
                    btn REAL,
                    horse_id INTEGER,
                    horse TEXT,
                    age INTEGER,
                    sex TEXT,
                    lbs INTEGER,
                    hg TEXT,
                    time TEXT,
                    secs REAL,
                    dec REAL,
                    jockey_id INTEGER,
                    jockey TEXT,
                    trainer_id INTEGER,
                    trainer TEXT,
                    prize REAL,
                    or_rating INTEGER,
                    rpr INTEGER,
                    ts INTEGER,
                    sire_id INTEGER,
                    sire TEXT,
                    dam_id INTEGER,
                    dam TEXT,
                    damsire_id INTEGER,
                    damsire TEXT,
                    owner_id INTEGER,
                    owner TEXT,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (race_id, horse_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_race_data_date ON race_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_race_data_course ON race_data(course)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_race_data_horse ON race_data(horse)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_race_data_type ON race_data(type)")
            
            conn.commit()
            self.logger.info("Database initialized successfully")

    def get_last_complete_date(self) -> Optional[str]:
        """Get the last complete date in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM race_data")
            result = cursor.fetchone()
            
            if result and result[0]:
                self.logger.info(f"Last date in database: {result[0]}")
                return result[0]
            else:
                self.logger.info("No data in database")
                return None

    def delete_date_data(self, date_str: str):
        """Delete all data for a specific date."""
        if self.dry_run:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM race_data WHERE date = ?", (date_str,))
                count = cursor.fetchone()[0]
                self.logger.info(f"[DRY RUN] Would delete {count} records for date {date_str}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM race_data WHERE date = ?", (date_str,))
            deleted_count = cursor.rowcount
            conn.commit()
            self.logger.info(f"Deleted {deleted_count} records for date {date_str}")

    def download_and_process_data(self, start_date: str, end_date: str) -> int:
        """Download and immediately process race data, cleaning up files after each import."""
        total_records = 0
        
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            rpscrape_date = current_date.strftime('%Y/%m/%d')
            
            self.logger.info(f"Processing data for {date_str}")
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would download: python {self.rpscrape_path} -d {rpscrape_date} -r gb")
                # Simulate processing
                self.logger.info(f"[DRY RUN] Would insert 100 dummy records for {date_str}")
                total_records += 100
            else:
                # Run rpscrape command
                cmd = [
                    sys.executable, self.rpscrape_path,
                    '-d', rpscrape_date,
                    '-r', 'gb'
                ]
                
                try:
                    # Change to rpscrape directory to ensure correct output location
                    rpscrape_dir = os.path.dirname(self.rpscrape_path)
                    result = subprocess.run(
                        cmd, 
                        cwd=rpscrape_dir,
                        capture_output=True, 
                        text=True, 
                        timeout=self.timeout
                    )
                    
                    if result.returncode != 0:
                        self.logger.error(f"rpscrape failed for {date_str}: {result.stderr}")
                        current_date += timedelta(days=1)
                        continue
                    
                    # Find the output file (rpscrape saves to data/dates/gb/ relative to rpscrape repo root)
                    rpscrape_repo_dir = os.path.dirname(rpscrape_dir) if os.path.basename(rpscrape_dir) == 'scripts' else rpscrape_dir
                    expected_file = os.path.join(rpscrape_repo_dir, 'data', 'dates', 'gb', f"{date_str.replace('-', '_')}.csv")
                    
                    if os.path.exists(expected_file):
                        self.logger.info(f"✓ Downloaded {date_str}")
                        
                        # Process the file immediately
                        try:
                            records = self.process_csv_file(expected_file)
                            total_records += records
                            
                            if records == 0:
                                # Empty file (no races) - this is normal for holidays/non-racing days
                                self.logger.info(f"✓ No races on {date_str} - normal for holidays/non-racing days")
                            else:
                                self.logger.info(f"✓ Processed {records} records from {os.path.basename(expected_file)}")
                            
                            # Clean up the file after processing (whether empty or not)
                            try:
                                os.remove(expected_file)
                                self.logger.info(f"✓ Cleaned up file: {os.path.basename(expected_file)}")
                            except Exception as e:
                                self.logger.error(f"✗ CRITICAL: Could not delete file {expected_file}: {e}")
                                raise RuntimeError(f"File cleanup failed: {e}")
                                
                        except Exception as e:
                            # Real processing error - this is a critical failure, stop the script
                            self.logger.error(f"✗ CRITICAL: Error processing {expected_file}: {e}")
                            self.logger.error(f"Keeping problematic file for debugging: {expected_file}")
                            raise RuntimeError(f"Processing failed for {date_str}: {e}")
                                
                    else:
                        error_msg = f"✗ CRITICAL: Expected file not found: {expected_file}"
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
                
                except subprocess.TimeoutExpired:
                    self.logger.error(f"Timeout downloading data for {date_str}")
                except Exception as e:
                    self.logger.error(f"Error downloading data for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        return total_records

    def process_csv_file(self, csv_file: str) -> int:
        """Process a CSV file and insert data into database."""
        try:
            # Try normal CSV reading first
            try:
                df = pd.read_csv(csv_file)
            except Exception as csv_error:
                self.logger.warning(f"Standard CSV parsing failed for {csv_file}: {csv_error}")
                self.logger.info(f"Attempting to repair malformed CSV data...")
                
                # Handle malformed CSV with unescaped newlines in comment field
                df = self._repair_and_read_csv(csv_file)
            
            if len(df) == 0:
                self.logger.info(f"No races found in {csv_file} (normal for holidays/non-racing days)")
                return 0  # Return 0 records processed, this is normal
            
            # Validate race types (should have correct types now)
            type_counts = df['type'].value_counts()
            self.logger.info(f"Race types in {os.path.basename(csv_file)}: {type_counts.to_dict()}")
            
            # Rename 'or' column to 'or_rating' to avoid SQL keyword conflict
            if 'or' in df.columns:
                df = df.rename(columns={'or': 'or_rating'})
            
            # Convert dist_f to float for consistency
            if 'dist_f' in df.columns:
                df['dist_f'] = df['dist_f'].astype(str)
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would insert {len(df)} records from {os.path.basename(csv_file)}")
                return len(df)
            
            # Insert data into database
            with sqlite3.connect(self.db_path) as conn:
                records_inserted = df.to_sql('race_data', conn, if_exists='append', index=False)
                conn.commit()
                
                # Always validate data integrity - critical for data quality
                date_from_file = os.path.basename(csv_file).replace('.csv', '').replace('_', '-')
                cursor = conn.execute("SELECT COUNT(*) FROM race_data WHERE date = ?", (date_from_file,))
                db_count = cursor.fetchone()[0]
                
                if db_count >= len(df):  # Allow for duplicates in case of re-runs
                    self.logger.info(f"✓ Validated: {len(df)} records inserted, {db_count} total in DB for {date_from_file}")
                    return len(df)
                else:
                    error_msg = f"✗ CRITICAL: Data validation failed! CSV has {len(df)} records, DB has {db_count} for {date_from_file}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
        
        except Exception as e:
            self.logger.error(f"Error processing {csv_file}: {e}")
            raise e  # Re-raise the exception to stop the script

    def _repair_and_read_csv(self, csv_file: str) -> pd.DataFrame:
        """Repair malformed CSV files with unescaped newlines in comment field."""
        import io
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get header to determine expected number of columns
        header = lines[0].strip()
        expected_columns = len(header.split(','))
        
        repaired_lines = [header]
        current_line = ""
        
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue
                
            current_line += line
            
            # Count columns by splitting on comma (rough estimate)
            # This isn't perfect but works for our data
            columns_count = len(current_line.split(','))
            
            # If we have enough columns or if the line starts with a date pattern
            # (new row), then we consider this line complete
            if (columns_count >= expected_columns or 
                (current_line and len(current_line.split(',')) > 0 and 
                 self._looks_like_new_row_start(current_line.split(',')[0]))):
                
                # Clean up the line and add it
                repaired_lines.append(current_line)
                current_line = ""
            else:
                # This line continues the previous row, add a space
                current_line += " "
        
        # Add any remaining line
        if current_line.strip():
            repaired_lines.append(current_line)
        
        # Create DataFrame from repaired lines
        repaired_csv = '\n'.join(repaired_lines)
        df = pd.read_csv(io.StringIO(repaired_csv))
        
        self.logger.info(f"✓ Repaired malformed CSV: {len(lines)} original lines → {len(df)} data rows")
        return df
    
    def _looks_like_new_row_start(self, first_field: str) -> bool:
        """Check if a field looks like the start of a new data row (date field)."""
        try:
            # Check if it's a date pattern YYYY-MM-DD
            if len(first_field) == 10 and first_field.count('-') == 2:
                parts = first_field.split('-')
                if (len(parts[0]) == 4 and parts[0].isdigit() and
                    len(parts[1]) == 2 and parts[1].isdigit() and
                    len(parts[2]) == 2 and parts[2].isdigit()):
                    return True
        except:
            pass
        return False

    def update_data(self, end_date: str = None):
        """Main method to update race data."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"Starting race data update to {end_date}")
        
        # Initialize database
        self.init_database()
        
        # Get last complete date
        last_date = self.get_last_complete_date()
        
        if last_date is None:
            # No data in database, start from configured historical start date
            start_date = self._get_config_value('common', 'historical_start_date', '2016-01-01')
            self.logger.info(f"No existing data, starting from configured historical start date: {start_date}")
        else:
            # Delete last day's data (might be incomplete)
            self.delete_date_data(last_date)
            start_date = last_date
        
        # Download and process data (with immediate cleanup)
        total_records = self.download_and_process_data(start_date, end_date)
        
        self.logger.info(f"Update complete. Total records processed: {total_records}")
        
        # Show summary
        if not self.dry_run:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM race_data")
                count, min_date, max_date = cursor.fetchone()
                self.logger.info(f"Database now contains {count} records from {min_date} to {max_date}")

def main():
    parser = argparse.ArgumentParser(description='Update race data incrementally')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    try:
        updater = RaceDataUpdater(
            dry_run=args.dry_run
        )
        
        # Ensure db directory exists
        db_dir = os.path.dirname(updater.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        updater.update_data()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
