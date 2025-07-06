# scripts/train_lightgbm_final.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import joblib
import argparse
from pathlib import Path

from common import DATA_DIR, PROJECT_DIR
import sys
sys.path.append(str(Path(__file__).parent.parent / 'config'))
from pipeline_config import get_model_file_path, PIPELINE_CONFIG

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train horse racing prediction model')
    parser.add_argument('--model-version', '-v', 
                       default=PIPELINE_CONFIG["default_model_version"],
                       help='Model version (e.g., v1, v2, v3)')
    parser.add_argument('--experiment-name', '-e',
                       help='Optional experiment name for this training run')
    return parser.parse_args()

# Define feature categories - REMOVED POTENTIALLY LEAKY FEATURES
identifier_features = [
    'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id'
]

categorical_features = [
    'type', 'sex', 'sire', 'dam', 'damsire', 'track_id', 'hg'
]

# REMOVED: 'or', 'rpr', 'ts' (potentially post-race data)
ordinal_features = [
    'class', 'pattern', 'going', 'age', 'lbs', 
    'age_min', 'age_max', 'dist_f', 'ran', 'draw', 'rating_max',
    'horse_course_win_pct', 'horse_distance_win_pct', 'horse_going_win_pct',
    'jockey_win_pct', 'trainer_win_pct',
    # 14-day performance stats (overall)
    'jockey_14d_runs', 'jockey_14d_wins', 'jockey_14d_win_pct',
    'trainer_14d_runs', 'trainer_14d_wins', 'trainer_14d_win_pct',
    # 14-day performance stats (by race type)
    'jockey_14d_type_runs', 'jockey_14d_type_wins', 'jockey_14d_type_win_pct',
    'trainer_14d_type_runs', 'trainer_14d_type_wins', 'trainer_14d_type_win_pct'
]

continuous_features = [
    'month_sin', 'month_cos'
]

print("🏇 TRAINING IMPROVED LIGHTGBM MODEL (LEAK-FREE + CALIBRATED)")
print("=" * 70)

# Load the encoded training data
data_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
print(f"Loading data from {data_file}")

df = pd.read_csv(data_file)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Prepare features (excluding potentially leaky ones)
all_features = categorical_features + ordinal_features + continuous_features

# Check which features actually exist in the data
available_features = [f for f in all_features if f in df.columns]
missing_features = [f for f in all_features if f not in df.columns]

print(f"Using {len(available_features)} features for training")
if missing_features:
    print(f"Missing features: {missing_features}")

excluded_features = identifier_features + ['win']
print(f"Excluded features: {excluded_features}")
print(f"Removed potentially leaky features: or, rpr, ts")

# Prepare the data
X = df[available_features]
y = df['win']

print(f"Target distribution: {y.value_counts().to_dict()}")

# Race-based splitting to prevent leakage
print("Splitting data by race to prevent leakage...")
unique_races = df['race_id'].unique()
train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)

train_mask = df['race_id'].isin(train_races)
test_mask = df['race_id'].isin(test_races)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"Training races: {len(train_races)}, Training samples: {len(X_train)}")
print(f"Test races: {len(test_races)}, Test samples: {len(X_test)}")

# Split training data for calibration
X_train_fit, X_train_cal, y_train_fit, y_train_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Model training: {len(X_train_fit)} samples")
print(f"Calibration: {len(X_train_cal)} samples")

# Check categorical features availability
available_categorical_features = [f for f in categorical_features if f in available_features]

# Create LightGBM datasets
train_data = lgb.Dataset(
    X_train_fit, 
    label=y_train_fit, 
    categorical_feature=available_categorical_features
)
valid_data = lgb.Dataset(
    X_train_cal, 
    label=y_train_cal, 
    reference=train_data,
    categorical_feature=available_categorical_features
)

# LightGBM parameters - more conservative to prevent overfitting
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # Reduced to prevent overfitting
    'learning_rate': 0.05,  # Lower learning rate
    'feature_fraction': 0.7,  # More aggressive feature sampling
    'bagging_fraction': 0.7,  # More aggressive data sampling
    'bagging_freq': 5,
    'min_data_in_leaf': 50,  # Increased to prevent overfitting
    'lambda_l1': 0.2,  # Increased regularization
    'lambda_l2': 0.2,  # Increased regularization
    'verbose': -1,
    'random_state': 42,
    'is_unbalance': True
}

print("Training base LightGBM model...")
print(f"Categorical features: {available_categorical_features}")

# Train the model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'cal'],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# Get predictions for calibration
print("Applying probability calibration...")
y_cal_pred = model.predict(X_train_cal, num_iteration=model.best_iteration)

# Fit isotonic regression for calibration (better than Platt scaling for this case)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_cal_pred, y_train_cal)

# Make predictions with both models
y_pred_base = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_calibrated = calibrator.predict(y_pred_base)

# Evaluate both models
def evaluate_model(y_true, y_pred_proba, model_name):
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"AUC: {auc:.4f}")
    
    # Threshold analysis
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        tp = ((y_pred_thresh == 1) & (y_true == 1)).sum()
        fp = ((y_pred_thresh == 1) & (y_true == 0)).sum()
        fn = ((y_pred_thresh == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        predicted_winners = y_pred_thresh.sum()
        
        print(f"Threshold {threshold:.1f}: Precision={precision:.3f} | Recall={recall:.3f} | "
              f"Predicted Winners={predicted_winners}")

# Check calibration improvement
def check_calibration(y_true, y_pred_proba, model_name):
    from sklearn.calibration import calibration_curve
    
    print(f"\n{model_name} Calibration Check:")
    print("-" * 40)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    
    calibration_errors = []
    for i, (predicted, actual) in enumerate(zip(mean_predicted_value, fraction_of_positives)):
        error = abs(predicted - actual)
        calibration_errors.append(error)
        print(f"Bin {i+1}: Predicted={predicted:.3f}, Actual={actual:.3f}, Error={error:.3f}")
    
    mean_error = np.mean(calibration_errors)
    print(f"Mean Calibration Error: {mean_error:.3f}")
    
    if mean_error < 0.05:
        print("✅ Well calibrated!")
    elif mean_error < 0.1:
        print("⚠️  Moderately calibrated")  
    else:
        print("❌ Poorly calibrated")
    
    return mean_error

# Evaluate both models
evaluate_model(y_test, y_pred_base, "BASE MODEL (Uncalibrated)")
evaluate_model(y_test, y_pred_calibrated, "CALIBRATED MODEL")

base_calib_error = check_calibration(y_test, y_pred_base, "BASE MODEL")
calibrated_calib_error = check_calibration(y_test, y_pred_calibrated, "CALIBRATED MODEL")

print(f"\n📊 CALIBRATION IMPROVEMENT:")
print(f"Base model calibration error: {base_calib_error:.3f}")
print(f"Calibrated model error: {calibrated_calib_error:.3f}")
print(f"Improvement: {(base_calib_error - calibrated_calib_error):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importance(importance_type='gain'),
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features (After Removing Leaky Features):")
print(feature_importance.head(15))

# Parse command line arguments for model version
args = parse_arguments()
model_version = args.model_version
experiment_name = args.experiment_name or ""

print(f"\n🎯 Saving model as version: {model_version}")
if experiment_name:
    print(f"📝 Experiment: {experiment_name}")

# Create model version directory
model_dir = PROJECT_DIR / "models" / model_version
model_dir.mkdir(parents=True, exist_ok=True)

# Save base model with version
base_model_file = model_dir / f"lightgbm_model_{model_version}.pkl"
joblib.dump(model, base_model_file)
print(f"\nBase model saved to {base_model_file}")

# Save calibrator with version
calibrator_file = model_dir / f"probability_calibrator_{model_version}.pkl"
joblib.dump(calibrator, calibrator_file)
print(f"Calibrator saved to {calibrator_file}")

# Save feature list with version
feature_list_file = model_dir / f"feature_list_{model_version}.txt"
with open(feature_list_file, 'w') as f:
    for feature in available_features:
        f.write(f"{feature}\n")
print(f"Feature list saved to {feature_list_file}")

# Save categorical features list with version
categorical_file = model_dir / f"categorical_features_{model_version}.txt"
with open(categorical_file, 'w') as f:
    for feature in available_categorical_features:
        f.write(f"{feature}\n")
print(f"Categorical features saved to {categorical_file}")

# Save feature importance with version
importance_file = model_dir / f"feature_importance_{model_version}.csv"
feature_importance.to_csv(importance_file, index=False)
print(f"Feature importance saved to {importance_file}")

print(f"\n🎯 SUMMARY:")
print(f"✅ Model version: {model_version}")
if experiment_name:
    print(f"✅ Experiment: {experiment_name}")
print(f"✅ Removed potentially leaky features: or, rpr, ts")
print(f"✅ Kept rating_max: race rating band limit (not leaky)")
print(f"✅ Applied isotonic calibration")
print(f"✅ Used conservative parameters to prevent overfitting")
print(f"✅ Race-based splitting to prevent data leakage")
print(f"✅ Calibration error improved from {base_calib_error:.3f} to {calibrated_calib_error:.3f}")
print(f"\n💡 To use for predictions:")
print(f"   1. Load the base model: lightgbm_model_{model_version}.pkl")
print(f"   2. Load the calibrator: probability_calibrator_{model_version}.pkl")
print(f"   3. Use: calibrator.predict(model.predict(X))")
print(f"\nTraining completed successfully!")
