# config/pipeline_config.py
"""
Configuration for the daily horse racing prediction pipeline.
"""

from pathlib import Path

# Pipeline Configuration
PIPELINE_CONFIG = {
    # Timing
    "daily_run_time": "09:00",  # When to run the pipeline
    "rpscrape_update_time": "09:00",  # When to update historical data
    
    # Data Sources
    "rpscrape_path": None,  # Will be auto-detected or set manually
    "historical_years": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "race_types": ["flat"],  # Only flat racing for now
    
    # Prediction Settings
    "min_probability_threshold": 0.15,  # 15% minimum win probability
    "max_predictions_per_race": 1,  # Only predict top horse per race
    
    # Google Sheets
    "spreadsheet_name": "UK Horse Racing Predictions",
    "active_sheet_name": "Active",
    "historical_sheet_name": "Historical",
    "results_template_name": "Results Template",
    
    # Notifications (future implementation)
    "enable_notifications": False,
    "notification_channels": {
        "whatsapp": False,
        "email": False,
        "telegram": False,
        "sms": False
    },
    
    # Model Settings
    "retrain_frequency": "weekly",  # How often to retrain the model
    "auto_retrain": False,  # Whether to automatically retrain
    "default_model_version": "v1",  # Default model version to use
    
    # Backup and Logging
    "backup_predictions": True,
    "log_level": "INFO",
    "max_log_files": 30
}

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LOG_DIR = PROJECT_DIR / 'logs'
BACKUP_DIR = PROJECT_DIR / 'backups'

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# Feature Configuration
FEATURE_CONFIG = {
    "use_14d_stats": True,
    "use_course_history": True,
    "use_distance_history": True,
    "use_going_history": True,
    "use_seasonal_features": True,
    
    # 14-day feature types
    "14d_feature_types": [
        "jockey_14d_runs", "jockey_14d_wins", "jockey_14d_win_pct",
        "trainer_14d_runs", "trainer_14d_wins", "trainer_14d_win_pct",
        "jockey_14d_type_runs", "jockey_14d_type_wins", "jockey_14d_type_win_pct",
        "trainer_14d_type_runs", "trainer_14d_type_wins", "trainer_14d_type_win_pct"
    ]
}

# File Patterns
FILE_PATTERNS = {
    "racecard_raw": "data/prediction/raw/{date}.json",
    "racecard_cleansed": "data/racecard_{date}_cleansed.csv",
    "predictions_output": "data/predictions_{date}_calibrated.csv",
    "encoded_training": "data/training/processed/encoded.csv",
    "model_file": "models/v1/lightgbm_model_v1.pkl",
    "calibrator_file": "models/v1/probability_calibrator_v1.pkl",
    "feature_list": "models/v1/feature_list_v1.txt"
}

def get_file_path(pattern_name, date=None, **kwargs):
    """Get file path from pattern"""
    from datetime import datetime
    
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    pattern = FILE_PATTERNS.get(pattern_name)
    if not pattern:
        raise ValueError(f"Unknown file pattern: {pattern_name}")
    
    # Replace placeholders
    file_path = pattern.format(date=date, **kwargs)
    return PROJECT_DIR / file_path

def get_model_file_path(file_type, model_version=None):
    """Get model-specific file path"""
    if model_version is None:
        model_version = PIPELINE_CONFIG["default_model_version"]
    
    model_patterns = {
        "model_file": f"models/{model_version}/lightgbm_model_{model_version}.pkl",
        "calibrator_file": f"models/{model_version}/probability_calibrator_{model_version}.pkl",
        "feature_list": f"models/{model_version}/feature_list_{model_version}.txt",
        "feature_importance": f"models/{model_version}/feature_importance_{model_version}.csv",
        "categorical_features": f"models/{model_version}/categorical_features_{model_version}.txt"
    }
    
    pattern = model_patterns.get(file_type)
    if not pattern:
        raise ValueError(f"Unknown model file type: {file_type}")
    
    return PROJECT_DIR / pattern

def validate_config():
    """Validate pipeline configuration"""
    issues = []
    
    # Check critical paths
    if not DATA_DIR.exists():
        issues.append(f"Data directory not found: {DATA_DIR}")
    
    # Check model files
    model_file = PROJECT_DIR / FILE_PATTERNS["model_file"]
    if not model_file.exists():
        issues.append(f"Model file not found: {model_file}")
    
    calibrator_file = PROJECT_DIR / FILE_PATTERNS["calibrator_file"]
    if not calibrator_file.exists():
        issues.append(f"Calibrator file not found: {calibrator_file}")
    
    feature_file = PROJECT_DIR / FILE_PATTERNS["feature_list"]
    if not feature_file.exists():
        issues.append(f"Feature list not found: {feature_file}")
    
    return issues

if __name__ == "__main__":
    print("🔧 Pipeline Configuration")
    print("=" * 40)
    
    # Validate configuration
    issues = validate_config()
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ Configuration is valid")
    
    print(f"\n📊 Settings:")
    print(f"  • Probability threshold: {PIPELINE_CONFIG['min_probability_threshold']:.0%}")
    print(f"  • Max predictions per race: {PIPELINE_CONFIG['max_predictions_per_race']}")
    print(f"  • Spreadsheet: {PIPELINE_CONFIG['spreadsheet_name']}")
    print(f"  • Data directory: {DATA_DIR}")
    print(f"  • Log directory: {LOG_DIR}")
    
    # Show file paths for today
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📁 Today's Files ({today}):")
    print(f"  • Racecard: {get_file_path('racecard_cleansed', today)}")
    print(f"  • Predictions: {get_file_path('predictions_output', today)}")
    print(f"  • Training data: {get_file_path('encoded_training')}")
