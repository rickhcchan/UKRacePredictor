"""
Race prediction script using prepared racecard data.class RacePredictor:
    def __init__(self, date: str = None, dry_run: bool = False, model_name: str = "default", strategy_name: str = "default", no_save: bool = False):
        self.target_date = date or datetime.now().strftime('%Y-%m-%d')
        self.dry_run = dry_run
        self.model_name = model_name
        self.strategy_name = strategy_name
        self.no_save = no_save
        self.logger = setup_logging() script handles the complete race prediction workflow:
1. Load prepared racecard with engineered features
2. Load trained model and calibrator
3. Make predictions and apply calibration
4. Format and display results with race context
5. Save predictions to file

The script requires a prepared racecard from prepare_racecard.py.

Betting Logic:
- Displays races where at least one horse has calibrated probability > 20%
- Recommends bets only when exactly one horse > 20% AND second-highest < 80% of top
- Shows all horses with calibrated probability > 20% in qualifying races

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python predict_races.py [--date YYYY-MM-DD] [--dry-run] [--model-name MODEL_NAME]
    
Examples:
    # Predict today's races using default model
    python predict_races.py
    
    # Predict races for specific date
    python predict_races.py --date 2025-07-08
    
    # Test run without saving predictions
    python predict_races.py --dry-run
    
    # Use specific model
    python predict_races.py --model-name v2
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import configparser
import joblib
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# Suppress specific pandas FutureWarnings about DataFrame concatenation
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation.*')

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging, convert_to_24h_time
from model_config import load_model_config
from strategy_factory import StrategyFactory


class RacePredictor:
    def __init__(self, date: str = None, dry_run: bool = False, model_names: List[str] = None, strategy_name: str = "default", threshold: float = 0.20):
        self.target_date = date or datetime.now().strftime('%Y-%m-%d')
        self.dry_run = dry_run
        self.model_names = model_names or ["default"]
        self.strategy_name = strategy_name
        self.threshold = threshold
        self.logger = setup_logging()
        
        # Support backward compatibility - if single model passed as string
        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]
        
        # Determine if this is single or multi-model mode
        self.is_multi_model = len(self.model_names) > 1
        
        # Load model configurations
        self.model_configs = {}
        for model_name in self.model_names:
            self.model_configs[model_name] = load_model_config(model_name)
            self.logger.info(f"Loaded model config for {model_name}: {self.model_configs[model_name].description}")
        
        # For backward compatibility, set model_name and model_config
        self.model_name = self.model_names[0] if len(self.model_names) == 1 else "multi"
        self.model_config = self.model_configs[self.model_names[0]] if len(self.model_names) == 1 else None
        
        # Load betting strategy
        self.strategy = StrategyFactory.create_strategy(self.strategy_name)
        self.logger.info(f"Loaded betting strategy: {self.strategy.description}")
        
        # Load system configuration
        self.config = self._load_config()
        
        # Live odds integration removed - sites change too frequently
        self.live_odds_manager = None
        
        # Odds display disabled (live odds integration removed)
        self.show_odds = False
        
        # Set paths from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.prediction_dir = Path(self._get_config_value('common', 'data_dir', 'data')) / 'prediction'
        self.models_dir = Path(self._get_config_value('common', 'models_dir', 'models'))
        
        self.logger.info(f"Predicting races for date: {self.target_date}")
        self.logger.info(f"Using model name: {self.model_name}")
        self.logger.info(f"Models directory: {self.models_dir}")
        
        # Model components for single/multi model support
        if self.is_multi_model:
            self.models = {}  # {model_name: {'base_model': ..., 'calibrator': ..., 'features': ...}}
        else:
            # Backward compatibility - single model
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
        """Load the trained model(s), calibrator(s), and feature list(s)."""
        if self.is_multi_model:
            return self._load_multiple_models()
        else:
            return self._load_single_model(self.model_names[0])

    def _load_multiple_models(self) -> bool:
        """Load multiple models for multi-model prediction."""
        for model_name in self.model_names:
            if not self._load_single_model_multi(model_name):
                return False
        return True

    def _load_single_model_multi(self, model_name: str) -> bool:
        """Load a single model's components for multi-model mode."""
        try:
            model_dir = self.models_dir / model_name
            
            if not model_dir.exists():
                model_dir = self.models_dir
                self.logger.warning(f"Model directory not found for {model_name}, using root models directory: {model_dir}")
            
            # Load base model
            model_file = model_dir / "lightgbm_model.pkl"
            if not model_file.exists():
                model_file = model_dir / f"lightgbm_model_{model_name}.pkl"
            if not model_file.exists():
                model_file = model_dir / "lightgbm_model_clean.pkl"
            
            if not model_file.exists():
                self.logger.error(f"Model file not found for {model_name}: {model_file}")
                return False
            
            base_model = joblib.load(model_file)
            self.logger.info(f"✓ Loaded base model for {model_name} from: {model_file}")
            
            # Load calibrator
            calibrator_file = model_dir / "probability_calibrator.pkl"
            if not calibrator_file.exists():
                calibrator_file = model_dir / f"probability_calibrator_{model_name}.pkl"
            
            if not calibrator_file.exists():
                self.logger.error(f"Calibrator file not found for {model_name}: {calibrator_file}")
                return False
            
            calibrator = joblib.load(calibrator_file)
            self.logger.info(f"✓ Loaded calibrator for {model_name} from: {calibrator_file}")
            
            # Load feature list
            feature_list_file = model_dir / "feature_list.txt"
            if not feature_list_file.exists():
                feature_list_file = model_dir / f"feature_list_{model_name}.txt"
            if not feature_list_file.exists():
                feature_list_file = model_dir / "feature_list_clean.txt"
            
            if feature_list_file.exists():
                with open(feature_list_file, 'r') as f:
                    feature_list = [line.strip() for line in f.readlines() if line.strip()]
                self.logger.info(f"✓ Loaded feature list for {model_name} with {len(feature_list)} features")
            else:
                feature_list = self.model_configs[model_name].get_all_features()
                self.logger.info(f"✓ Using config feature list for {model_name} with {len(feature_list)} features")
            
            # Store model components
            self.models[model_name] = {
                'base_model': base_model,
                'calibrator': calibrator,
                'features': feature_list,
                'config': self.model_configs[model_name]
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return False

    def _load_single_model(self, model_name: str) -> bool:
        """Load a single model for backward compatibility."""
        try:
            model_dir = self.models_dir / self.model_name
            
            if not model_dir.exists():
                # Fallback to root models directory for backward compatibility
                model_dir = self.models_dir
                self.logger.warning(f"Model directory not found, using root models directory: {model_dir}")
            
            # Load base model
            model_file = model_dir / "lightgbm_model.pkl"
            if not model_file.exists():
                # Try old naming convention for backward compatibility
                model_file = model_dir / f"lightgbm_model_{self.model_name}.pkl"
            if not model_file.exists():
                # Try alternative naming
                model_file = model_dir / "lightgbm_model_clean.pkl"
            
            if not model_file.exists():
                self.logger.error(f"Model file not found: {model_file}")
                return False
            
            self.base_model = joblib.load(model_file)
            self.logger.info(f"✓ Loaded base model from: {model_file}")
            
            # Load calibrator
            calibrator_file = model_dir / "probability_calibrator.pkl"
            if not calibrator_file.exists():
                # Try old naming convention for backward compatibility
                calibrator_file = model_dir / f"probability_calibrator_{self.model_name}.pkl"
            
            if not calibrator_file.exists():
                self.logger.error(f"Calibrator file not found: {calibrator_file}")
                return False
            
            self.calibrator = joblib.load(calibrator_file)
            self.logger.info(f"✓ Loaded calibrator from: {calibrator_file}")
            
            # Load feature list - prefer actual model feature list over config
            feature_list_file = model_dir / "feature_list.txt"
            if not feature_list_file.exists():
                # Try old naming convention for backward compatibility
                feature_list_file = model_dir / f"feature_list_{self.model_name}.txt"
            if not feature_list_file.exists():
                # Try alternative naming
                feature_list_file = model_dir / "feature_list_clean.txt"
            
            if feature_list_file.exists():
                with open(feature_list_file, 'r') as f:
                    self.feature_list = [line.strip() for line in f.readlines() if line.strip()]
                self.logger.info(f"✓ Loaded feature list from file with {len(self.feature_list)} features")
            else:
                # Fallback to config feature list
                self.feature_list = self.model_config.all_features
                self.logger.info(f"✓ Using feature list from config with {len(self.feature_list)} features")
                self.logger.warning("Feature list file not found, using config - there may be mismatches")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
            if not calibrator_file.exists():
                # Try old naming convention for backward compatibility
                calibrator_file = model_dir / f"probability_calibrator_{self.model_name}.pkl"
            
            if not calibrator_file.exists():
                self.logger.error(f"Calibrator file not found: {calibrator_file}")
                return False
            
            self.calibrator = joblib.load(calibrator_file)
            self.logger.info(f"✓ Loaded calibrator from: {calibrator_file}")
            
            # Load feature list - prefer actual model feature list over config
            feature_list_file = model_dir / "feature_list.txt"
            if not feature_list_file.exists():
                # Try old naming convention for backward compatibility
                feature_list_file = model_dir / f"feature_list_{self.model_name}.txt"
            if not feature_list_file.exists():
                # Try alternative naming
                feature_list_file = model_dir / "feature_list_clean.txt"
            
            if feature_list_file.exists():
                with open(feature_list_file, 'r') as f:
                    self.feature_list = [line.strip() for line in f.readlines() if line.strip()]
                self.logger.info(f"✓ Loaded feature list from file with {len(self.feature_list)} features")
            else:
                # Fallback to config feature list
                self.feature_list = self.model_config.all_features
                self.logger.info(f"✓ Using feature list from config with {len(self.feature_list)} features")
                self.logger.warning("Feature list file not found, using config - there may be mismatches")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def load_prepared_racecard(self) -> Optional[pd.DataFrame]:
        """Load prepared racecard with engineered features."""
        racecard_file = self.prediction_dir / f"racecard_{self.target_date}_prepared.csv"
        
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
        
        if self.is_multi_model:
            return self._make_multi_model_predictions(df)
        else:
            return self._make_single_model_predictions(df)

    def _make_single_model_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using a single model (backward compatibility)."""
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

    def _make_multi_model_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using multiple models and combine with union logic."""
        df = df.copy()
        model_predictions = {}
        
        # Make predictions for each model
        for model_name in self.model_names:
            model = self.models[model_name]
            self.logger.info(f"Making predictions with model: {model_name}")
            
            # Check feature alignment for this model
            missing_features = [f for f in model['features'] if f not in df.columns]
            if missing_features:
                self.logger.warning(f"Missing features for model {model_name}: {missing_features}")
                # Fill missing features with defaults
                for feature in missing_features:
                    df[feature] = 0
            
            # Prepare features for this model
            X = df[model['features']]
            
            # Make base predictions
            base_predictions = model['base_model'].predict(X, num_iteration=model['base_model'].best_iteration)
            
            # Apply calibration
            calibrated_predictions = model['calibrator'].predict(base_predictions)
            
            # Store model predictions
            model_predictions[model_name] = {
                'base_probability': base_predictions,
                'win_probability': calibrated_predictions
            }
            
            # Add model-specific columns to dataframe
            df[f'base_probability_{model_name}'] = base_predictions
            df[f'win_probability_{model_name}'] = calibrated_predictions
            
            self.logger.info(f"Model {model_name} - Base range: {base_predictions.min():.3f} - {base_predictions.max():.3f}")
            self.logger.info(f"Model {model_name} - Calibrated range: {calibrated_predictions.min():.3f} - {calibrated_predictions.max():.3f}")
        
        # Calculate union predictions (maximum of all models)
        # Use max to show the highest confidence any model has in each horse
        all_base_probs = np.array([model_predictions[name]['base_probability'] for name in self.model_names])
        all_win_probs = np.array([model_predictions[name]['win_probability'] for name in self.model_names])
        
        df['base_probability'] = np.max(all_base_probs, axis=0)
        df['win_probability'] = np.max(all_win_probs, axis=0)
        
        # Add model agreement analysis
        df['model_agreement'] = self._calculate_model_agreement(model_predictions, df)
        df['models_agreeing'] = self._count_agreeing_models(model_predictions, df)
        
        self.logger.info(f"Union predictions - Base range: {df['base_probability'].min():.3f} - {df['base_probability'].max():.3f}")
        self.logger.info(f"Union predictions - Calibrated range: {df['win_probability'].min():.3f} - {df['win_probability'].max():.3f}")
        
        return df

    def _calculate_model_agreement(self, model_predictions: dict, df: pd.DataFrame) -> np.ndarray:
        """Calculate the agreement score between models (0-1, higher = more agreement)."""
        # For each horse, calculate the standard deviation of predictions
        # Lower std = higher agreement
        all_probs = np.array([model_predictions[name]['win_probability'] for name in self.model_names])
        std_dev = np.std(all_probs, axis=0)
        
        # Convert to agreement score (1 - normalized std)
        max_std = np.max(std_dev) if np.max(std_dev) > 0 else 1
        agreement = 1 - (std_dev / max_std)
        
        return agreement

    def _count_agreeing_models(self, model_predictions: dict, df: pd.DataFrame) -> np.ndarray:
        """Count how many models would select each horse using the strategy threshold."""
        # Apply strategy to each model's predictions to see which horses would be selected
        counts = np.zeros(len(df))
        
        for i, race_group in df.groupby('race_id'):
            race_indices = race_group.index
            
            for model_name in self.model_names:
                model_probs = model_predictions[model_name]['win_probability'][race_indices]
                
                # Apply threshold logic (simplified - using basic threshold)
                threshold = self.threshold if hasattr(self, 'threshold') else 0.15
                selected = model_probs >= threshold
                
                counts[race_indices] += selected.astype(int)
        
        return counts

    def normalize_race_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize probabilities within each race to sum to 100%."""
        self.logger.info("Normalizing probabilities within races...")
        
        def normalize_race_group(race_group):
            total_prob = race_group['win_probability'].sum()
            if total_prob > 0:
                race_group = race_group.copy()
                race_group['win_probability_normalized'] = (race_group['win_probability'] / total_prob) * 100
            else:
                race_group = race_group.copy()
                race_group['win_probability_normalized'] = 100 / len(race_group)  # Equal probability
            return race_group
        
        # Apply normalization to avoid deprecated groupby behavior
        normalized_groups = []
        for race_id, race_group in df.groupby('race_id'):
            normalized_group = normalize_race_group(race_group)
            normalized_groups.append(normalized_group)
        
        df = pd.concat(normalized_groups, ignore_index=True)
        
        return df

    def format_and_display_results(self, df: pd.DataFrame):
        """Format and display prediction results using betting strategy."""
        self.logger.info("Formatting prediction results...")
        
        bet_races = []
        bet_horses = []
        
        # Process each race using the betting strategy
        for race_id, race_group in df.groupby('race_id'):
            # Convert race group to list of horse dictionaries
            horses = []
            for _, horse_row in race_group.iterrows():
                horse_dict = {
                    'horse_id': str(race_id) + "_" + str(horse_row.name),  # Create unique ID
                    'horse_name': horse_row.get('horse_name', 'Unknown'),  # Fixed: use horse_name column
                    'calibrated_probability': horse_row.get('win_probability', 0.0),
                    'jockey': horse_row.get('jockey', 'Unknown'),
                    'trainer': horse_row.get('trainer', 'Unknown'),
                    'weight': horse_row.get('lbs', 0),
                    'draw': horse_row.get('draw', 0),
                    'age': horse_row.get('age', 0),
                    'course_name': horse_row.get('course', 'Unknown'),  # Add course for strategy use
                    'race_time': horse_row.get('time', 'Unknown'),      # Add time for strategy use
                    # Add multi-model specific data
                    'model_agreement': horse_row.get('model_agreement', 1.0) if self.is_multi_model else 1.0,
                    'models_agreeing': horse_row.get('models_agreeing', 1) if self.is_multi_model else 1,
                    # Add all other columns
                    **{col: horse_row.get(col) for col in horse_row.index}
                }
                horses.append(horse_dict)
            
            # Create race data dictionary
            race_data = {
                'race_id': race_id,
                'course_name': race_group.iloc[0].get('course', 'Unknown'),
                'race_time': race_group.iloc[0].get('time', 'Unknown'),
                'field_size': len(race_group),
                'distance': race_group.iloc[0].get('dist_f', 0),
                # Add other race-level data
                **{col: race_group.iloc[0].get(col) for col in ['class_number', 'going_id', 'pattern_id'] if col in race_group.columns}
            }
            
            # Use strategy to select horses
            selected_horses = self.strategy.select_horses(horses, race_data)
            
            if selected_horses:
                bet_races.append(race_id)
                bet_horses.extend(selected_horses)
        
        # Prepare header content
        model_info = f" ({', '.join(self.model_names)})" if self.is_multi_model else f" ({self.model_name})"
        header_line = f"🏇 UK HORSE RACING PREDICTIONS{model_info} - {self.strategy.name.upper()}"
        recommendations_line = "🎯 BETTING RECOMMENDATIONS:"
        strategy_line = f"💡 {self.strategy.description}"
        
        # Calculate the width needed (longest line) + buffer for emoji/encoding issues
        header_width = max(len(header_line), len(recommendations_line), len(strategy_line)) + 2
        
        print(f"\n{header_line}")
        print("=" * header_width)
        
        if len(bet_horses) == 0:
            print(f"❌ No races found with betting recommendations")
            print(f"   Strategy: {self.strategy.description}")
            print("💡 No bets recommended for today.")
            return
        
        # Convert selected horses back to DataFrame format for display
        bet_horses_df = pd.DataFrame()
        for horse in bet_horses:
            # Create a row that matches the original DataFrame structure
            horse_row = {
                'course': horse.get('course_name', 'Unknown'),
                'time': horse.get('race_time', 'Unknown'),
                'race_id': horse.get('race_id', 'Unknown'),
                'horse_name': horse.get('horse_name', 'Unknown'),  # Fixed: use horse_name consistently
                'win_probability': horse.get('calibrated_probability', 0.0),
                'jockey': horse.get('jockey', 'Unknown'),
                'trainer': horse.get('trainer', 'Unknown'),
                'lbs': horse.get('weight', 0),
                'draw': horse.get('draw', 0),
                'age': horse.get('age', 0),
                'model_agreement': horse.get('model_agreement', 1.0),
                'models_agreeing': horse.get('models_agreeing', 1)
            }
            
            # Add model-specific probabilities if in multi-model mode
            if self.is_multi_model:
                for model_name in self.model_names:
                    prob_col = f'win_probability_{model_name}'
                    if prob_col in df.columns:
                        horse_row[prob_col] = horse.get(prob_col, 0.0)
            
            bet_horses_df = pd.concat([bet_horses_df, pd.DataFrame([horse_row])], ignore_index=True)
        
        # Sort by time (if available), then course, then calibrated probability
        if 'time' in bet_horses_df.columns:
            try:
                bet_horses_df['time_24h'] = bet_horses_df['time'].apply(convert_to_24h_time)
                bet_horses_df = bet_horses_df.sort_values(['time_24h', 'course', 'win_probability'], ascending=[True, True, False])
            except:
                bet_horses_df = bet_horses_df.sort_values(['course', 'win_probability'], ascending=[True, False])
        else:
            bet_horses_df = bet_horses_df.sort_values(['course', 'win_probability'], ascending=[True, False])
        
        print(recommendations_line)
        print(strategy_line)
        print("=" * header_width)
        
        current_race = None
        for _, horse in bet_horses_df.iterrows():
            race_key = f"{horse.get('course', 'Unknown')} {horse.get('time', 'Unknown')}"
            
            if race_key != current_race:
                if current_race is not None:
                    print()  # Add space between races
                
                # Count total horses in this race
                total_horses_in_race = len(df[df['race_id'] == horse['race_id']])
                
                print(f"\n📍 {horse.get('course', 'Unknown')} - {horse.get('time', 'Unknown')} ({total_horses_in_race} horses total)")
                print("-" * 100)
                
                if self.is_multi_model:
                    # Multi-model header
                    header = f"{'Horse':18}"
                    for model_name in self.model_names:
                        header += f" | {model_name[:10]:>12}"
                    print(header)
                else:
                    # Single model header
                    print(f"{'Horse':18} | {'Probability':>11}")
                
                print("-" * 100)
                current_race = race_key
            
            # Format probability display
            calib_prob = horse['win_probability'] * 100
            
            if self.is_multi_model:
                # Multi-model display with consistent column widths
                line = f"{horse.get('horse_name', 'Unknown'):18}"
                
                # Add individual model probabilities with tick/cross indicators
                for model_name in self.model_names:
                    model_prob_col = f'win_probability_{model_name}'
                    model_prob = horse.get(model_prob_col, 0.0) * 100
                    
                    # Use simple threshold check for individual model indicators
                    # This shows if the individual model probability meets the basic threshold
                    threshold = 0.20  # 20% threshold for t20win strategy
                    indicator = "✓" if horse.get(model_prob_col, 0.0) >= threshold else "✗"
                    
                    # Format with consistent width: 12 chars total to match header
                    line += f" | {model_prob:7.1f}% {indicator:>2}"
                
                print(line)
            else:
                # Single model display
                print(f"{horse.get('horse_name', 'Unknown'):18} | {calib_prob:9.1f}%")
        
        # Calculate summary statistics
        total_races = df['race_id'].nunique()
        bet_race_count = len(bet_races)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total races analyzed: {total_races}")
        print(f"Races with bet recommendations: {bet_race_count}")
        print(f"Bet coverage: {bet_race_count/total_races*100:.1f}% of races")
        
        if self.is_multi_model:
            print(f"Models used: {', '.join(self.model_names)}")
            print(f"Union strategy: Maximum ensemble")
            if len(bet_horses_df) > 0:
                avg_agreement = bet_horses_df['model_agreement'].mean() * 100
                print(f"Average model agreement: {avg_agreement:.1f}%")
        else:
            print(f"Model used: {self.model_name}")
        
        print(f"Strategy used: {self.strategy.name} - {self.strategy.description}")
        if len(bet_horses_df) > 0:
            print(f"Average recommended horse probability: {bet_horses_df['win_probability'].mean()*100:.1f}%")
            print(f"Highest recommended probability: {bet_horses_df['win_probability'].max()*100:.1f}%")

    def save_predictions(self, df: pd.DataFrame):
        """Save predictions to file."""
        # Create filename based on single or multi-model mode
        if self.is_multi_model:
            model_suffix = "_".join(self.model_names)
            output_file = self.prediction_dir / f"predictions_{self.target_date}_multi_{model_suffix}.csv"
        else:
            output_file = self.prediction_dir / f"predictions_{self.target_date}_{self.model_name}.csv"
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would save {len(df)} predictions to: {output_file}")
            return
        
        # Select key columns for output
        output_columns = [
            'race_id', 'horse_id', 'horse_name', 'course', 'time',
            'win_probability', 'win_probability_normalized', 'base_probability'
        ]
        
        # Add multi-model specific columns
        if self.is_multi_model:
            output_columns.extend(['model_agreement', 'models_agreeing'])
            # Add individual model predictions
            for model_name in self.model_names:
                base_col = f'base_probability_{model_name}'
                win_col = f'win_probability_{model_name}'
                if base_col in df.columns:
                    output_columns.append(base_col)
                if win_col in df.columns:
                    output_columns.append(win_col)
        
        # Add additional columns if they exist
        for col in ['age', 'draw', 'lbs', 'jockey_id', 'trainer_id']:
            if col in df.columns:
                output_columns.append(col)
        
        # Filter columns that actually exist in the dataframe
        existing_columns = [col for col in output_columns if col in df.columns]
        output_df = df[existing_columns].copy()
        
        output_df.to_csv(output_file, index=False)
        self.logger.info(f"✓ Saved {len(output_df)} predictions to: {output_file}")
        
        if self.is_multi_model:
            self.logger.info(f"Multi-model predictions saved with {len(self.model_names)} models: {', '.join(self.model_names)}")

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
    parser.add_argument('--model', '-m',
                       type=str,
                       default='default',
                       help='Model name(s) to use. For single model: --model win. For multiple models: --model win,top3 (comma-separated, no spaces)')
    parser.add_argument('--strategy', '-s',
                       type=str,
                       default='default',
                       help='Betting strategy to use (default: default)')
    
    args = parser.parse_args()
    
    try:
        # Parse comma-separated model names
        model_names = [name.strip() for name in args.model.split(',') if name.strip()]
        
        # Log the mode being used
        if len(model_names) > 1:
            print(f"🔗 Multi-model mode: Using {len(model_names)} models: {', '.join(model_names)}")
        else:
            print(f"📊 Single-model mode: Using model: {model_names[0]}")
        
        predictor = RacePredictor(
            date=args.date,
            dry_run=args.dry_run,
            model_names=model_names,  # Pass list of model names
            strategy_name=args.strategy
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
