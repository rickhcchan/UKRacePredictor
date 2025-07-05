# scripts/train_lightgbm.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path

from common import DATA_DIR

# Define feature categories for proper LightGBM handling

# Identifier features - exclude from training (no predictive value for new races)
identifier_features = [
    'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id'
]

# Categorical features (no inherent ordering, but have predictive value)
categorical_features = [
    'type', 'sex', 'sire', 'dam', 'damsire', 'track_id', 'hg'
]

# Ordinal features (meaningful ordering - bigger/smaller matters)
ordinal_features = [
    'class', 'pattern', 'going', 'age', 'lbs', 'or', 'rpr', 'ts', 
    'rating_max', 'age_min', 'age_max', 'dist_f', 'ran', 'draw'
]

# Continuous/numerical features (cyclical or truly continuous)
continuous_features = [
    'month_sin', 'month_cos'
]

# Note on feature choices:
# - track_id: Keeps course+distance combination (e.g., "Haydock_12f") as unique track characteristics
# - dist_f: Also kept as ordinal to capture distance effects across all tracks
# - draw: Ordinal because position order matters (inside vs outside draws)
# - ran: Ordinal because field size affects competitiveness

# Load the encoded training data
data_file = DATA_DIR / 'training' / 'processed' / 'encoded.csv'
print(f"Loading data from {data_file}")

df = pd.read_csv(data_file)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Remove identifier features from training
target_col = 'win'
all_features = categorical_features + ordinal_features + continuous_features

# Only use features that exist in the dataset
available_features = [col for col in all_features if col in df.columns]
missing_features = [col for col in all_features if col not in df.columns]

if missing_features:
    print(f"Warning: Missing expected features: {missing_features}")

print(f"Using {len(available_features)} features for training")
print(f"Excluded identifier features: {[col for col in identifier_features if col in df.columns]}")
X = df[available_features]
y = df[target_col]

print(f"Target distribution: {y.value_counts().to_dict()}")

# CRITICAL: Split by race_id to prevent data leakage
# Horses in the same race should all be in training OR test, never split across both
print("Splitting data by race to prevent leakage...")

race_ids = df['race_id'].unique()
print(f"Total unique races: {len(race_ids)}")

# Split races into train/test (not individual records)
train_races, test_races = train_test_split(race_ids, test_size=0.2, random_state=42)

# Create masks for train/test based on race_id
train_mask = df['race_id'].isin(train_races)
test_mask = df['race_id'].isin(test_races)

# Apply masks to get final datasets
X_train = df[train_mask][available_features]
y_train = df[train_mask][target_col]
X_test = df[test_mask][available_features]
y_test = df[test_mask][target_col]

print(f"Training races: {len(train_races)}, Training samples: {len(X_train)}")
print(f"Test races: {len(test_races)}, Test samples: {len(X_test)}")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Identify categorical features that are available
available_categorical_features = [col for col in categorical_features if col in available_features]

# Create LightGBM datasets with categorical feature specification
train_data = lgb.Dataset(
    X_train, 
    label=y_train, 
    categorical_feature=available_categorical_features
)
valid_data = lgb.Dataset(
    X_test, 
    label=y_test, 
    reference=train_data,
    categorical_feature=available_categorical_features
)

# LightGBM parameters optimized for horse racing
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,  # Increased for more complex patterns
    'learning_rate': 0.03,  # Lower for better convergence
    'feature_fraction': 0.8,  # Feature sampling
    'bagging_fraction': 0.8,  # Data sampling
    'bagging_freq': 5,
    'min_data_in_leaf': 20,  # Prevent overfitting
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1,  # L2 regularization
    'verbose': 0,
    'random_state': 42,
    'is_unbalance': True  # Handle class imbalance
}

print("Training LightGBM model...")
print(f"Categorical features: {available_categorical_features}")

# Train the model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'eval'],
    num_boost_round=2000,  # Increased for better training
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
)

# Make predictions
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model at default 0.5 threshold
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Performance at 0.5 threshold:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Best iteration: {model.best_iteration}")
print(f"\nClassification Report (0.5 threshold):")
print(classification_report(y_test, y_pred))

# Analyze different thresholds - NO RETRAINING NEEDED!
print(f"\n" + "="*50)
print("THRESHOLD ANALYSIS - Different Win Probability Cutoffs")
print("="*50)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics for this threshold
    accuracy_thresh = accuracy_score(y_test, y_pred_thresh)
    
    # Calculate precision and recall manually
    true_positives = ((y_pred_thresh == 1) & (y_test == 1)).sum()
    false_positives = ((y_pred_thresh == 1) & (y_test == 0)).sum()
    false_negatives = ((y_pred_thresh == 0) & (y_test == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    predicted_winners = y_pred_thresh.sum()
    actual_winners = y_test.sum()
    
    print(f"Threshold {threshold:.1f}: Precision={precision:.3f} | Recall={recall:.3f} | "
          f"Accuracy={accuracy_thresh:.3f} | Predicted Winners={predicted_winners} | "
          f"Actual Winners={actual_winners}")

print(f"\nInterpretation:")
print(f"- Higher threshold = Higher precision (more accurate picks) but fewer picks")
print(f"- Lower threshold = More picks but lower precision")
print(f"- AUC stays the same ({auc:.4f}) - it measures ranking ability regardless of threshold")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importance(importance_type='gain'),
    'split_importance': model.feature_importance(importance_type='split')
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Save the model and metadata
model_dir = Path("../models")
model_dir.mkdir(exist_ok=True)

model_file = model_dir / "lightgbm_model.pkl"
joblib.dump(model, model_file)
print(f"\nModel saved to {model_file}")

# Save feature list for inference
feature_list_file = model_dir / "feature_list.txt"
with open(feature_list_file, 'w') as f:
    for feature in available_features:
        f.write(f"{feature}\n")
print(f"Feature list saved to {feature_list_file}")

# Save categorical features list
categorical_file = model_dir / "categorical_features.txt"
with open(categorical_file, 'w') as f:
    for feature in available_categorical_features:
        f.write(f"{feature}\n")
print(f"Categorical features saved to {categorical_file}")

# Save feature importance
importance_file = model_dir / "feature_importance.csv"
feature_importance.to_csv(importance_file, index=False)
print(f"Feature importance saved to {importance_file}")

print(f"\nTraining completed successfully!")
