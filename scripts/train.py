"""
Model training script for horse racing prediction.

This script handles the complete model training workflow:
1. Load encoded features from database
2. Split data by race to prevent leakage
3. Train LightGBM model with proper categorical handling
4. Apply isotonic calibration for better probability estimates
5. Evaluate model performance with P&L analysis
6. Save model artifacts with versioning

Configuration is loaded from:
1. config/user_settings.conf (if exists) - personal settings, not in git
2. config/default_settings.conf (fallback) - default settings, in git

Usage:
    python train.py [--model-version VERSION] [--experiment-name NAME] [--dry-run]
    
Examples:
    # Train with default version
    python train.py
    
    # Train with specific version
    python train.py --model-version v2 --experiment-name "leak_free_model"
    
    # Test run without saving
    python train.py --dry-run
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import argparse
import configparser
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# Add the scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from common import setup_logging

class ModelTrainer:
    def __init__(self, model_version: str = None, experiment_name: str = None, dry_run: bool = False):
        self.model_version = model_version or "v1"
        self.experiment_name = experiment_name
        self.dry_run = dry_run
        self.logger = setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Set paths from config
        self.db_path = self._get_config_value('common', 'db_path', 'data/race_data.db')
        self.models_dir = self._get_config_value('common', 'models_dir', 'models')
        
        self.logger.info(f"Using database: {self.db_path}")
        self.logger.info(f"Models directory: {self.models_dir}")
        self.logger.info(f"Model version: {self.model_version}")
        if self.experiment_name:
            self.logger.info(f"Experiment: {self.experiment_name}")
        
        # Define feature categories
        self._define_features()

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

    def _load_config(self) -> configparser.ConfigParser:
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

    def _define_features(self):
        """Define feature categories for training."""
        # Identifier features (excluded from training)
        self.identifier_features = [
            'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id', 
            'horse_name', 'course', 'race_name', 'date', 'created_at'
        ]

        # Categorical features (nominal categories)
        self.categorical_features = [
            'course_id', 'type_id', 'sex_id', 'sire_id', 'dam_id', 'damsire_id', 'hg',
            'sex_rest'  # Keep sex_rest as it's useful for modeling
        ]

        # Ordinal features (meaningful order/continuous)
        self.ordinal_features = [
            'pattern_id', 'going_id', 'age', 'lbs', 'dist_f', 'ran', 'draw',
            # Engineered features from categorical transformations
            'rating_upper', 'age_min', 'age_max', 'class_number',
            # Horse historical features
            'horse_total_runs', 'horse_total_wins', 'horse_win_pct',
            'horse_course_runs', 'horse_course_wins', 'horse_course_win_pct',
            'horse_distance_runs', 'horse_distance_wins', 'horse_distance_win_pct',
            'horse_going_runs', 'horse_going_wins', 'horse_going_win_pct',
            'horse_days_since_last_run',  # ✅ Our new feature!
            # Jockey features
            'jockey_total_runs', 'jockey_total_wins', 'jockey_win_pct',
            'jockey_course_runs', 'jockey_course_wins', 'jockey_course_win_pct',
            'jockey_distance_runs', 'jockey_distance_wins', 'jockey_distance_win_pct',
            'jockey_going_runs', 'jockey_going_wins', 'jockey_going_win_pct',
            'jockey_14d_runs', 'jockey_14d_wins', 'jockey_14d_win_pct',
            'jockey_14d_type_runs', 'jockey_14d_type_wins', 'jockey_14d_type_win_pct',
            # Trainer features
            'trainer_total_runs', 'trainer_total_wins', 'trainer_win_pct',
            'trainer_course_runs', 'trainer_course_wins', 'trainer_course_win_pct',
            'trainer_distance_runs', 'trainer_distance_wins', 'trainer_distance_win_pct',
            'trainer_going_runs', 'trainer_going_wins', 'trainer_going_win_pct',
            'trainer_14d_runs', 'trainer_14d_wins', 'trainer_14d_win_pct',
            'trainer_14d_type_runs', 'trainer_14d_type_wins', 'trainer_14d_type_win_pct',
            # Ratings (potentially leaky - will be filtered out later)
            'or_rating', 'rpr', 'ts'
        ]

        # Continuous features
        self.continuous_features = [
            'month_sin', 'month_cos'
        ]

        # All training features
        self.all_features = self.categorical_features + self.ordinal_features + self.continuous_features

    def load_data(self) -> pd.DataFrame:
        """Load encoded training data from database."""
        self.logger.info("Loading encoded training data from database")
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if encoded table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='encoded_race_data'
            """)
            
            if not cursor.fetchone():
                raise FileNotFoundError("encoded_race_data table not found. Run encode_incremental.py first.")
            
            # Load data ordered by date and race
            df = pd.read_sql_query("""
                SELECT * FROM encoded_race_data 
                ORDER BY date, race_id
            """, conn)
        
        self.logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        self.logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        self.logger.info("Preparing features for training")
        
        # Apply feature engineering transformations
        df = self._engineer_features(df.copy())
        
        # Check which features are available
        available_features = [f for f in self.all_features if f in df.columns]
        missing_features = [f for f in self.all_features if f not in df.columns]
        
        self.logger.info(f"Using {len(available_features)} features for training")
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        # Remove potentially leaky features (ratings that might be post-race)
        potentially_leaky = ['or_rating', 'rpr', 'ts']
        clean_features = [f for f in available_features if f not in potentially_leaky]
        
        if set(potentially_leaky) & set(available_features):
            removed = set(potentially_leaky) & set(available_features)
            self.logger.info(f"Removed potentially leaky features: {list(removed)}")
        
        # Update categorical features list to only include available ones
        self.available_categorical_features = [f for f in self.categorical_features if f in clean_features]
        
        # Prepare target variable
        if 'win' not in df.columns:
            raise ValueError("Target variable 'win' not found in data")
        
        X = df[clean_features]
        y = df['win']
        
        target_distribution = y.value_counts().to_dict()
        self.logger.info(f"Target distribution: {target_distribution}")
        
        return X, y, clean_features

    def split_data(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Split data by race to prevent leakage."""
        self.logger.info("Splitting data by race to prevent leakage")
        
        # Race-based splitting
        unique_races = df['race_id'].unique()
        train_races, test_races = train_test_split(
            unique_races, test_size=0.2, random_state=42
        )
        
        train_mask = df['race_id'].isin(train_races)
        test_mask = df['race_id'].isin(test_races)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        self.logger.info(f"Training races: {len(train_races):,}, Training samples: {len(X_train):,}")
        self.logger.info(f"Test races: {len(test_races):,}, Test samples: {len(X_test):,}")
        
        # Further split training data for calibration
        X_train_fit, X_train_cal, y_train_fit, y_train_cal = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.logger.info(f"Model training: {len(X_train_fit):,} samples")
        self.logger.info(f"Calibration: {len(X_train_cal):,} samples")
        
        return X_train_fit, X_train_cal, X_test, y_train_fit, y_train_cal, y_test, test_mask

    def train_model(self, X_train_fit: pd.DataFrame, X_train_cal: pd.DataFrame, 
                   y_train_fit: pd.Series, y_train_cal: pd.Series) -> tuple:
        """Train LightGBM model with calibration."""
        self.logger.info("Training LightGBM model")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train_fit, 
            label=y_train_fit, 
            categorical_feature=self.available_categorical_features
        )
        valid_data = lgb.Dataset(
            X_train_cal, 
            label=y_train_cal, 
            reference=train_data,
            categorical_feature=self.available_categorical_features
        )
        
        # Model parameters - conservative to prevent overfitting
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True
        }
        
        self.logger.info(f"Categorical features: {self.available_categorical_features}")
        
        # Train the model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'cal'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Apply calibration
        self.logger.info("Applying isotonic calibration")
        y_cal_pred = model.predict(X_train_cal, num_iteration=model.best_iteration)
        
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_cal_pred, y_train_cal)
        
        return model, calibrator

    def evaluate_model(self, model, calibrator, X_test: pd.DataFrame, y_test: pd.Series, 
                      df_test: pd.DataFrame, feature_names: list):
        """Evaluate model performance with P&L analysis."""
        self.logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred_base = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_calibrated = calibrator.predict(y_pred_base)
        
        # Basic metrics
        auc_base = roc_auc_score(y_test, y_pred_base)
        auc_calibrated = roc_auc_score(y_test, y_pred_calibrated)
        
        self.logger.info(f"Base model AUC: {auc_base:.4f}")
        self.logger.info(f"Calibrated model AUC: {auc_calibrated:.4f}")
        
        # Calibration analysis
        base_calib_error = self._check_calibration(y_test, y_pred_base, "Base Model")
        calibrated_calib_error = self._check_calibration(y_test, y_pred_calibrated, "Calibrated Model")
        
        improvement = base_calib_error - calibrated_calib_error
        self.logger.info(f"Calibration improvement: {improvement:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Top 15 most important features:")
        for _, row in feature_importance.head(15).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.0f}")
        
        # P&L analysis if odds available
        if 'dec' in df_test.columns:
            self._analyze_betting_performance(y_test, y_pred_calibrated, df_test)
        
        return {
            'base_model': model,
            'calibrator': calibrator,
            'feature_importance': feature_importance,
            'metrics': {
                'auc_base': auc_base,
                'auc_calibrated': auc_calibrated,
                'calibration_error_base': base_calib_error,
                'calibration_error_calibrated': calibrated_calib_error
            }
        }

    def _check_calibration(self, y_true: pd.Series, y_pred_proba: np.ndarray, model_name: str) -> float:
        """Check model calibration."""
        self.logger.info(f"Checking calibration for {model_name}")
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        calibration_errors = []
        for i, (predicted, actual) in enumerate(zip(mean_predicted_value, fraction_of_positives)):
            error = abs(predicted - actual)
            calibration_errors.append(error)
            self.logger.debug(f"Bin {i+1}: Predicted={predicted:.3f}, Actual={actual:.3f}, Error={error:.3f}")
        
        mean_error = np.mean(calibration_errors)
        
        if mean_error < 0.05:
            status = "✅ Well calibrated"
        elif mean_error < 0.1:
            status = "⚠️ Moderately calibrated"
        else:
            status = "❌ Poorly calibrated"
        
        self.logger.info(f"{model_name} calibration error: {mean_error:.3f} - {status}")
        
        return mean_error

    def _analyze_betting_performance(self, y_true: pd.Series, y_pred_proba: np.ndarray, df_test: pd.DataFrame):
        """Analyze betting performance with P&L calculations."""
        self.logger.info("Analyzing betting performance")
        
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        
        self.logger.info("Threshold analysis (£1 stakes):")
        best_pnl = -float('inf')
        best_threshold = 0.15
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            if y_pred_thresh.sum() == 0:
                continue
                
            # Calculate P&L
            bet_mask = y_pred_thresh == 1
            bet_outcomes = y_true[bet_mask]
            bet_odds = df_test.loc[bet_mask, 'dec']
            
            pnl = 0
            valid_bets = 0
            
            for outcome, odds in zip(bet_outcomes, bet_odds):
                if pd.isna(odds) or odds <= 0:
                    continue
                valid_bets += 1
                if outcome == 1:
                    pnl += (odds - 1)  # Profit
                else:
                    pnl += -1  # Loss
            
            if valid_bets > 0:
                roi = (pnl / valid_bets) * 100
                precision = sum(bet_outcomes) / len(bet_outcomes)
                
                status = "✅ PROFITABLE" if pnl > 0 else "❌ Loss"
                self.logger.info(f"  {threshold:.0%}: P&L = £{pnl:+7.2f} | ROI = {roi:+6.1f}% | Precision = {precision:.3f} | {status}")
                
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_threshold = threshold
        
        self.logger.info(f"Best threshold: {best_threshold:.0%} with P&L of £{best_pnl:+.2f}")

    def save_model_artifacts(self, results: dict, feature_names: list) -> dict:
        """Save model artifacts with versioning."""
        if self.dry_run:
            self.logger.info("[DRY RUN] Would save model artifacts")
            return {}
        
        self.logger.info(f"Saving model artifacts for version: {self.model_version}")
        
        # Create model directory
        model_dir = Path(self.models_dir) / self.model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save base model
        base_model_file = model_dir / f"lightgbm_model_{self.model_version}.pkl"
        joblib.dump(results['base_model'], base_model_file)
        saved_files['base_model'] = str(base_model_file)
        self.logger.info(f"Base model saved to: {base_model_file}")
        
        # Save calibrator
        calibrator_file = model_dir / f"probability_calibrator_{self.model_version}.pkl"
        joblib.dump(results['calibrator'], calibrator_file)
        saved_files['calibrator'] = str(calibrator_file)
        self.logger.info(f"Calibrator saved to: {calibrator_file}")
        
        # Save feature list
        feature_list_file = model_dir / f"feature_list_{self.model_version}.txt"
        with open(feature_list_file, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        saved_files['feature_list'] = str(feature_list_file)
        self.logger.info(f"Feature list saved to: {feature_list_file}")
        
        # Save categorical features
        categorical_file = model_dir / f"categorical_features_{self.model_version}.txt"
        with open(categorical_file, 'w') as f:
            for feature in self.available_categorical_features:
                f.write(f"{feature}\n")
        saved_files['categorical_features'] = str(categorical_file)
        self.logger.info(f"Categorical features saved to: {categorical_file}")
        
        # Save feature importance
        importance_file = model_dir / f"feature_importance_{self.model_version}.csv"
        results['feature_importance'].to_csv(importance_file, index=False)
        saved_files['feature_importance'] = str(importance_file)
        self.logger.info(f"Feature importance saved to: {importance_file}")
        
        return saved_files

    def run_training(self):
        """Main method to run the complete training workflow."""
        self.logger.info("Starting model training workflow")
        
        try:
            # Load data
            df = self.load_data()
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            
            # Split data
            X_train_fit, X_train_cal, X_test, y_train_fit, y_train_cal, y_test, test_mask = self.split_data(df, X, y)
            
            # Train model
            model, calibrator = self.train_model(X_train_fit, X_train_cal, y_train_fit, y_train_cal)
            
            # Evaluate model
            df_test = df[test_mask].reset_index(drop=True)
            results = self.evaluate_model(model, calibrator, X_test, y_test, df_test, feature_names)
            
            # Save artifacts
            saved_files = self.save_model_artifacts(results, feature_names)
            
            self.logger.info("Training completed successfully!")
            
            # Summary
            metrics = results['metrics']
            self.logger.info(f"Final summary:")
            self.logger.info(f"  Model version: {self.model_version}")
            self.logger.info(f"  AUC improvement: {metrics['auc_base']:.4f} -> {metrics['auc_calibrated']:.4f}")
            self.logger.info(f"  Calibration improvement: {metrics['calibration_error_base']:.3f} -> {metrics['calibration_error_calibrated']:.3f}")
            self.logger.info(f"  Training samples: {len(X_train_fit):,}")
            self.logger.info(f"  Test samples: {len(X_test):,}")
            self.logger.info(f"  Features used: {len(feature_names)}")
            
            return results, saved_files
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features by transforming categorical features into numeric ones.
        
        Transformations:
        - rating_band: Extract upper limit (e.g., "0-100" -> 100)
        - age_band: Extract min/max age (e.g., "4-6yo" -> 4, 6; "4yo+" -> 4, 99)
        - class: Extract class number (e.g., "Class 5" -> 5)
        - sex_rest: Keep as categorical (useful for modeling)
        """
        self.logger.info("Engineering features...")
        
        # Rating band: extract upper limit
        if 'rating_band' in df.columns:
            self.logger.info("Engineering rating_band feature")
            def extract_rating_upper(rating_band):
                if pd.isna(rating_band):
                    return np.nan
                try:
                    # Extract number after dash (e.g., "0-100" -> 100)
                    if '-' in str(rating_band):
                        return int(str(rating_band).split('-')[1])
                    else:
                        return np.nan
                except (ValueError, IndexError):
                    return np.nan
            
            df['rating_upper'] = df['rating_band'].apply(extract_rating_upper)
            self.logger.info(f"Created rating_upper: min={df['rating_upper'].min()}, max={df['rating_upper'].max()}")
        
        # Age band: extract min and max age
        if 'age_band' in df.columns:
            self.logger.info("Engineering age_band feature")
            def extract_age_range(age_band):
                if pd.isna(age_band):
                    return np.nan, np.nan
                
                age_str = str(age_band).lower().replace('yo', '')
                
                try:
                    if '+' in age_str:
                        # Cases like "4+", "5+" -> min age, max = 99
                        min_age = int(age_str.replace('+', ''))
                        return min_age, 99
                    elif '-' in age_str:
                        # Cases like "4-6", "3-5" -> min, max
                        parts = age_str.split('-')
                        min_age = int(parts[0])
                        max_age = int(parts[1])
                        return min_age, max_age
                    else:
                        # Cases like "2", "3" -> same min and max
                        age = int(age_str)
                        return age, age
                except (ValueError, IndexError):
                    return np.nan, np.nan
            
            age_ranges = df['age_band'].apply(extract_age_range)
            df['age_min'] = [x[0] for x in age_ranges]
            df['age_max'] = [x[1] for x in age_ranges]
            self.logger.info(f"Created age_min: min={df['age_min'].min()}, max={df['age_min'].max()}")
            self.logger.info(f"Created age_max: min={df['age_max'].min()}, max={df['age_max'].max()}")
        
        # Class: extract class number
        if 'class' in df.columns:
            self.logger.info("Engineering class feature")
            def extract_class_number(class_val):
                if pd.isna(class_val):
                    return np.nan
                try:
                    # Extract number from "Class X" (e.g., "Class 5" -> 5)
                    class_str = str(class_val).lower()
                    if 'class' in class_str:
                        return int(class_str.replace('class', '').strip())
                    else:
                        return np.nan
                except (ValueError, AttributeError):
                    return np.nan
            
            df['class_number'] = df['class'].apply(extract_class_number)
            self.logger.info(f"Created class_number: min={df['class_number'].min()}, max={df['class_number'].max()}")
        
        self.logger.info("Feature engineering completed")
        return df

def main():
    parser = argparse.ArgumentParser(description='Train horse racing prediction model')
    parser.add_argument('--model-version', '-v', 
                       default='v1',
                       help='Model version (e.g., v1, v2, v3)')
    parser.add_argument('--experiment-name', '-e',
                       help='Optional experiment name for this training run')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Test run without saving model artifacts')
    
    args = parser.parse_args()
    
    try:
        trainer = ModelTrainer(
            model_version=args.model_version,
            experiment_name=args.experiment_name,
            dry_run=args.dry_run
        )
        
        # Ensure models directory exists
        models_dir = Path(trainer.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.run_training()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
