"""
Race prediction script using prepared racecard data.

This script handles the complete race prediction workflow:
1. Load prepared racecard with engineered features
2. Load trained model and calibrator
3. Make predictions and apply calibration
4. Format and display results with race context
5. Save predictions to file

The script requires a prepared racecard from prepare_racecard.py.

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python predict_races.py [--date YYYY-MM-DD] [--dry-run] [--model-version v1]
    
Examples:
    # Predict today's races
    python predict_races.py
    
    # Predict races for specific date
    python predict_races.py --date 2025-07-08
    
    # Test run without saving predictions
    python predict_races.py --dry-run
    
    # Use specific model version
    python predict_races.py --model-version v2
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import configparser
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging, convert_to_24h_time

class RacePredictor:
    def __init__(self, date: str = None, dry_run: bool = False, model_version: str = "v1"):
        self.target_date = date or datetime.now().strftime('%Y-%m-%d')
        self.dry_run = dry_run
        self.model_version = model_version
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Set paths from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.prediction_processed_dir = Path(self._get_config_value('common', 'data_dir', 'data')) / 'prediction' / 'processed'
        self.models_dir = Path(self._get_config_value('common', 'models_dir', 'models'))
        
        self.logger.info(f"Predicting races for date: {self.target_date}")
        self.logger.info(f"Using model version: {self.model_version}")
        self.logger.info(f"Models directory: {self.models_dir}")
        
        # Model components
        self.base_model = None
        self.calibrator = None
        self.feature_list = None

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

    def load_models(self) -> bool:
        """Load the trained model, calibrator, and feature list."""
        try:
            model_dir = self.models_dir / self.model_version
            
            if not model_dir.exists():
                # Fallback to root models directory for backward compatibility
                model_dir = self.models_dir
                self.logger.warning(f"Model version directory not found, using root models directory: {model_dir}")
            
            # Load base model
            model_file = model_dir / f"lightgbm_model_{self.model_version}.pkl"
            if not model_file.exists():
                # Try alternative naming
                model_file = model_dir / "lightgbm_model_clean.pkl"
            
            if not model_file.exists():
                self.logger.error(f"Model file not found: {model_file}")
                return False
            
            self.base_model = joblib.load(model_file)
            self.logger.info(f"✓ Loaded base model from: {model_file}")
            
            # Load calibrator
            calibrator_file = model_dir / f"probability_calibrator_{self.model_version}.pkl"
            if not calibrator_file.exists():
                # Try alternative naming
                calibrator_file = model_dir / "probability_calibrator.pkl"
            
            if not calibrator_file.exists():
                self.logger.error(f"Calibrator file not found: {calibrator_file}")
                return False
            
            self.calibrator = joblib.load(calibrator_file)
            self.logger.info(f"✓ Loaded calibrator from: {calibrator_file}")
            
            # Load feature list
            feature_file = model_dir / f"feature_list_{self.model_version}.txt"
            if not feature_file.exists():
                # Try alternative naming
                feature_file = model_dir / "feature_list_clean.txt"
            
            if not feature_file.exists():
                self.logger.error(f"Feature list file not found: {feature_file}")
                return False
            
            with open(feature_file, 'r') as f:
                self.feature_list = [line.strip() for line in f]
            
            self.logger.info(f"✓ Loaded feature list with {len(self.feature_list)} features from: {feature_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def load_prepared_racecard(self) -> Optional[pd.DataFrame]:
        """Load prepared racecard with engineered features."""
        racecard_file = self.prediction_processed_dir / f"racecard_{self.target_date}_prepared.csv"
        
        if not racecard_file.exists():
            self.logger.error(f"Prepared racecard not found: {racecard_file}")
            self.logger.error("Run prepare_racecard.py first to create prepared racecard")
            return None
        
        try:
            df = pd.read_csv(racecard_file)
            self.logger.info(f"✓ Loaded prepared racecard: {len(df)} runners from {df['race_id'].nunique()} races")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading prepared racecard: {e}")
            return None

    def make_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the prepared racecard data."""
        self.logger.info("Making predictions...")
        
        # Check feature alignment
        missing_features = [f for f in self.feature_list if f not in df.columns]
        if missing_features:
            self.logger.warning(f"Missing features in racecard: {missing_features}")
            # Fill missing features with defaults
            for feature in missing_features:
                df[feature] = 0
        
        # Prepare features for prediction
        X = df[self.feature_list]
        
        # Make base predictions
        base_predictions = self.base_model.predict(X, num_iteration=self.base_model.best_iteration)
        
        # Apply calibration
        calibrated_predictions = self.calibrator.predict(base_predictions)
        
        # Add predictions to dataframe
        df = df.copy()
        df['base_probability'] = base_predictions
        df['win_probability'] = calibrated_predictions
        
        self.logger.info(f"Base probability range: {base_predictions.min():.3f} - {base_predictions.max():.3f}")
        self.logger.info(f"Calibrated probability range: {calibrated_predictions.min():.3f} - {calibrated_predictions.max():.3f}")
        
        return df

    def normalize_race_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize probabilities within each race to sum to 100%."""
        self.logger.info("Normalizing probabilities within races...")
        
        def normalize_race_group(race_group):
            total_prob = race_group['win_probability'].sum()
            if total_prob > 0:
                race_group['win_probability_normalized'] = (race_group['win_probability'] / total_prob) * 100
            else:
                race_group['win_probability_normalized'] = 100 / len(race_group)  # Equal probability
            return race_group
        
        df = df.groupby('race_id').apply(normalize_race_group).reset_index(drop=True)
        
        return df

    def format_and_display_results(self, df: pd.DataFrame):
        """Format and display prediction results using betting criteria logic."""
        self.logger.info("Formatting prediction results...")
        
        # Betting criteria from original predict.py
        MIN_CALIB_THRESHOLD = 0.20  # 20% calibrated probability minimum (X)
        MIN_HORSES_PER_RACE = 3     # Show at least 3 horses when race qualifies
        
        # Calculate dynamic normalized threshold (Y) based on number of runners
        def calculate_norm_threshold(num_runners):
            # Y = 1.5 × (100/num_runners) - 50% above random chance
            factor = 1.5
            threshold = factor * (100 / num_runners) / 100  # Convert to decimal
            return threshold
        
        qualifying_races = []
        
        # Process each race to find qualifying ones
        for race_id, race_group in df.groupby('race_id'):
            # Sort by normalized probability (highest first)
            race_sorted = race_group.sort_values('win_probability_normalized', ascending=False)
            
            # Calculate dynamic normalized threshold for this race
            num_runners = len(race_group)
            min_norm_threshold = calculate_norm_threshold(num_runners)
            
            # Check if race has any qualifying horses
            # Horse qualifies if it meets: Calibrated >20% OR Normalized >dynamic%
            qualifying_horses = race_sorted[
                (race_sorted['win_probability'] >= MIN_CALIB_THRESHOLD) |
                (race_sorted['win_probability_normalized'] >= (min_norm_threshold * 100))
            ]
            
            race_qualifies = len(qualifying_horses) > 0
            
            if race_qualifies:
                # Display Logic: Show at least 3 horses (qualifying + reference for context)
                horses_to_show = race_sorted.head(MIN_HORSES_PER_RACE)
                
                # Add any additional qualifying horses beyond the top 3
                additional_qualifying = qualifying_horses[~qualifying_horses.index.isin(horses_to_show.index)]
                if len(additional_qualifying) > 0:
                    horses_to_show = pd.concat([horses_to_show, additional_qualifying])
                
                # Handle ties with the last shown horse
                if len(race_sorted) > len(horses_to_show):
                    last_horse_norm_prob = horses_to_show.iloc[-1]['win_probability_normalized']
                    
                    # Find all remaining horses with same probability as last horse
                    remaining_horses = race_sorted[~race_sorted.index.isin(horses_to_show.index)]
                    same_prob_horses = remaining_horses[
                        remaining_horses['win_probability_normalized'] == last_horse_norm_prob
                    ]
                    
                    # Include horses with same probability as last horse
                    if len(same_prob_horses) > 0:
                        horses_to_show = pd.concat([horses_to_show, same_prob_horses])
                
                qualifying_races.append(horses_to_show)
        
        # Combine qualifying races for display
        if qualifying_races:
            significant_horses = pd.concat(qualifying_races, ignore_index=True)
        else:
            significant_horses = pd.DataFrame()
        
        print("\n🏇 UK HORSE RACING PREDICTIONS")
        print("=" * 80)
        
        if len(significant_horses) == 0:
            print(f"❌ No races found with top horse meeting EITHER criteria:")
            print(f"   • Calibrated probability >{MIN_CALIB_THRESHOLD:.0%} OR")
            print(f"   • Normalized probability >dynamic threshold (1.5 × random chance)")
            print("💡 No qualifying races to display.")
            return
        
        # Sort by time (if available), then course, then normalized probability
        if 'time' in significant_horses.columns:
            try:
                significant_horses['time_24h'] = significant_horses['time'].apply(convert_to_24h_time)
                significant_horses = significant_horses.sort_values(['time_24h', 'course', 'win_probability_normalized'], ascending=[True, True, False])
            except:
                significant_horses = significant_horses.sort_values(['course', 'win_probability_normalized'], ascending=[True, False])
        else:
            significant_horses = significant_horses.sort_values(['course', 'win_probability_normalized'], ascending=[True, False])
        
        print(f"🎯 QUALIFYING RACES (At least 1 horse: Calibrated >{MIN_CALIB_THRESHOLD:.0%} OR normalized >dynamic threshold):")
        print("💡 BET LEGEND: ✅ BET = Qualifies (Calibrated >20% OR normalized >dynamic%), 📋 REF = Reference only")
        print("=" * 90)
        
        current_race = None
        for _, horse in significant_horses.iterrows():
            race_key = f"{horse.get('course', 'Unknown')} {horse.get('time', 'Unknown')}"
            
            if race_key != current_race:
                if current_race is not None:
                    print()  # Add space between races
                
                # Count total horses in this race
                total_horses_in_race = len(df[df['race_id'] == horse['race_id']])
                
                # Calculate and display the dynamic threshold for this race
                dynamic_threshold = calculate_norm_threshold(total_horses_in_race) * 100  # Convert to percentage
                
                print(f"\n📍 {horse.get('course', 'Unknown')} - {horse.get('time', 'Unknown')} ({total_horses_in_race} horses total)")
                print(f"   Qualifies: Calibrated >{MIN_CALIB_THRESHOLD:.0%} OR normalized >{dynamic_threshold:.1f}% (1.5 × {100/total_horses_in_race:.1f}%)")
                print("-" * 80)
                print(f"{'Horse':25} | {'Base%':>6} | {'Calib%':>7} | {'Norm%':>6} | {'BET':>6}")
                print("-" * 80)
                current_race = race_key
            
            # Format probability display with all three percentages
            base_prob = horse['base_probability'] * 100
            calib_prob = horse['win_probability'] * 100
            norm_prob = horse['win_probability_normalized']
            
            # Determine if this is a bet (horse qualifies: above X OR above Y threshold)
            dynamic_threshold = calculate_norm_threshold(len(df[df['race_id'] == horse['race_id']])) * 100
            meets_calib_threshold = calib_prob >= (MIN_CALIB_THRESHOLD * 100)
            meets_norm_threshold = norm_prob >= dynamic_threshold
            is_bet = meets_calib_threshold or meets_norm_threshold
            bet_indicator = "✅ BET" if is_bet else "📋 REF"
            
            print(f"{horse.get('horse_name', 'Unknown'):25} | {base_prob:5.1f}% | {calib_prob:6.1f}% | {norm_prob:5.1f}% | {bet_indicator:>6}")
        
        # Calculate summary statistics
        total_predictions = len(significant_horses)
        races_with_predictions = significant_horses['race_id'].nunique() if len(significant_horses) > 0 else 0
        total_races = df['race_id'].nunique()
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total races analyzed: {total_races}")
        print(f"Races with predictions: {races_with_predictions}")
        print(f"Total predictions: {total_predictions}")
        print(f"Coverage: {races_with_predictions/total_races*100:.1f}% of races")
        print(f"Min calibrated threshold: {MIN_CALIB_THRESHOLD:.0%}")
        print(f"Dynamic normalized threshold: varies by race (1.5x random chance)")
        print(f"Min horses per qualifying race: {MIN_HORSES_PER_RACE}")
        if total_predictions > 0:
            print(f"Average calibrated prob: {significant_horses['win_probability'].mean()*100:.1f}%")
            print(f"Average normalized prob: {significant_horses['win_probability_normalized'].mean():.1f}%")
            print(f"Max calibrated prob: {significant_horses['win_probability'].max()*100:.1f}%")
            print(f"Max normalized prob: {significant_horses['win_probability_normalized'].max():.1f}%")

    def save_predictions(self, df: pd.DataFrame):
        """Save predictions to file."""
        output_file = self.prediction_processed_dir / f"predictions_{self.target_date}_{self.model_version}.csv"
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would save {len(df)} predictions to: {output_file}")
            return
        
        # Select key columns for output
        output_columns = [
            'race_id', 'horse_id', 'horse_name', 'course', 'time',
            'win_probability', 'win_probability_normalized', 'base_probability'
        ]
        
        # Add additional columns if they exist
        for col in ['age', 'draw', 'lbs', 'jockey_id', 'trainer_id']:
            if col in df.columns:
                output_columns.append(col)
        
        output_df = df[output_columns].copy()
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"✓ Saved {len(output_df)} predictions to: {output_file}")

    def run_prediction(self):
        """Main method to run race prediction."""
        self.logger.info(f"Starting race prediction for {self.target_date}")
        
        # Load models
        if not self.load_models():
            self.logger.error("Failed to load models")
            return False
        
        # Load prepared racecard
        df = self.load_prepared_racecard()
        if df is None:
            return False
        
        # Make predictions
        df_with_predictions = self.make_predictions(df)
        
        # Normalize probabilities within races
        df_normalized = self.normalize_race_probabilities(df_with_predictions)
        
        # Display results
        self.format_and_display_results(df_normalized)
        
        # Save predictions
        self.save_predictions(df_normalized)
        
        self.logger.info(f"Race prediction complete for {self.target_date}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Predict race winners using prepared racecard')
    parser.add_argument('--date', 
                       type=str,
                       help='Target date for predictions (YYYY-MM-DD), defaults to today')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show predictions without saving to file')
    parser.add_argument('--model-version', 
                       type=str,
                       default='v1',
                       help='Model version to use (default: v1)')
    
    args = parser.parse_args()
    
    try:
        predictor = RacePredictor(
            date=args.date,
            dry_run=args.dry_run,
            model_version=args.model_version
        )
        
        success = predictor.run_prediction()
        
        if success:
            print(f"\n✓ Race prediction completed successfully for {predictor.target_date}")
        else:
            print(f"\n✗ Race prediction failed for {predictor.target_date}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
