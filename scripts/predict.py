# scripts/predict_calibrated.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from common import DATA_DIR, PROJECT_DIR, convert_to_24h_time
# Import Google Sheets functionality
try:
    from google_sheets_writer import write_formatted_predictions_to_sheet
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("💡 Google Sheets integration not available. Install 'gspread google-auth' to enable.")

def load_models():
    """Load the v1 calibrated model"""
    model_dir = PROJECT_DIR / "models" / "v1"
    
    # Load base model
    base_model = joblib.load(model_dir / "lightgbm_model_v1.pkl")
    
    # Load calibrator
    calibrator = joblib.load(model_dir / "probability_calibrator_v1.pkl")
    
    # Load feature list
    with open(model_dir / "feature_list_v1.txt", 'r') as f:
        feature_list = [line.strip() for line in f]
    
    return base_model, calibrator, feature_list

def predict_race_winners():
    """Make predictions using the cleaned, calibrated model"""
    print("🏇 UK HORSE RACING PREDICTIONS (CALIBRATED MODEL)")
    print("=" * 60)
    
    # Load models
    base_model, calibrator, feature_list = load_models()
    print(f"✅ Loaded cleaned model with {len(feature_list)} features")
    
    # Load today's cleansed racecard
    racecard_file = DATA_DIR / 'prediction' / 'processed' / f"racecard_{datetime.now().strftime('%Y-%m-%d')}_cleansed.csv"
    
    if not racecard_file.exists():
        print(f"❌ No racecard file found: {racecard_file}")
        print("💡 Run cleanse_racecard.py first to download and process today's races")
        return
    
    df = pd.read_csv(racecard_file)
    print(f"📊 Loaded {len(df)} horses from {df['race_id'].nunique()} races")
    
    # Load original racecard for race details (course, time)
    import json
    racecard_json_file = DATA_DIR / 'prediction' / 'raw' / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    race_details = {}
    
    if racecard_json_file.exists():
        with open(racecard_json_file, 'r', encoding='utf-8') as f:
            racecard_data = json.load(f)
            
        # Extract race details
        courses = racecard_data.get('GB', {})
        for course_name, races in courses.items():
            for race_time, race_info in races.items():
                race_id = int(race_info.get('race_id', 0))
                race_details[race_id] = {
                    'course': course_name,
                    'time': race_time
                }
    
    # Add race details to dataframe
    df['course'] = df['race_id'].map(lambda x: race_details.get(x, {}).get('course', f'Race_{x}'))
    df['time'] = df['race_id'].map(lambda x: race_details.get(x, {}).get('time', 'Unknown'))
    
    # Convert time to 24-hour format for proper chronological sorting
    df['time_24h'] = df['time'].apply(convert_to_24h_time)
    
    # Check feature alignment
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        print(f"⚠️  Warning: Missing features in racecard: {missing_features}")
        # Fill missing features with defaults
        for feature in missing_features:
            df[feature] = 0

    # Prepare features for prediction
    X = df[feature_list]
    
    # Make base predictions
    base_predictions = base_model.predict(X, num_iteration=base_model.best_iteration)
    
    # Apply calibration
    calibrated_predictions = calibrator.predict(base_predictions)
    
    # Add predictions to dataframe
    df['base_probability'] = base_predictions
    df['win_probability'] = calibrated_predictions
    
    print(f"\n📈 PREDICTION SUMMARY:")
    print(f"Base model probability range: {base_predictions.min():.3f} - {base_predictions.max():.3f}")
    print(f"Calibrated probability range: {calibrated_predictions.min():.3f} - {calibrated_predictions.max():.3f}")
    
    # Group by race and normalize probabilities within each race
    def normalize_race_probabilities(race_group):
        total_prob = race_group['win_probability'].sum()
        if total_prob > 0:
            race_group['win_probability_normalized'] = (race_group['win_probability'] / total_prob) * 100
        else:
            race_group['win_probability_normalized'] = 100 / len(race_group)  # Equal probability
        return race_group
    
    df = df.groupby('race_id').apply(normalize_race_probabilities).reset_index(drop=True)
    
    # Strategy: Take only the TOP horse per race if above threshold
    # This prevents field size bias and improves precision
    MIN_PROBABILITY_THRESHOLD = 0.15  # 15% calibrated probability minimum
    MAX_PREDICTIONS_PER_RACE = 1  # Only predict the favorite
    
    significant_horses = []
    race_groups = df.groupby('race_id')
    
    for race_id, race_group in race_groups:
        # Sort by calibrated probability (highest first)
        race_sorted = race_group.sort_values('win_probability', ascending=False)
        
        # Take top horses above threshold, limited per race
        top_horses = race_sorted.head(MAX_PREDICTIONS_PER_RACE)
        qualifying_horses = top_horses[top_horses['win_probability'] >= MIN_PROBABILITY_THRESHOLD]
        
        if len(qualifying_horses) > 0:
            significant_horses.append(qualifying_horses)
    
    if significant_horses:
        significant_horses = pd.concat(significant_horses, ignore_index=True)
    else:
        significant_horses = pd.DataFrame()
    
    if len(significant_horses) == 0:
        print(f"\n❌ No horses found with >{MIN_PROBABILITY_THRESHOLD:.0%} win probability")
        print("💡 Lowering threshold to show top horses...")
        # Show top horse per race regardless of threshold
        top_per_race = []
        for race_id, race_group in df.groupby('race_id'):
            top_horse = race_group.nlargest(1, 'win_probability')
            top_per_race.append(top_horse)
        significant_horses = pd.concat(top_per_race, ignore_index=True)
        print(f"📊 Top horse per race (forced selection):")
    
    # Sort by 24-hour time first (chronological order), then course, then probability
    significant_horses = significant_horses.sort_values(['time_24h', 'course', 'win_probability'], ascending=[True, True, False])
    
    print(f"\n🎯 TOP HORSES PER RACE (>{MIN_PROBABILITY_THRESHOLD:.0%} threshold, max {MAX_PREDICTIONS_PER_RACE} per race):")
    print("=" * 80)
    
    current_race = None
    for _, horse in significant_horses.iterrows():
        race_key = f"{horse['course']} {horse['time']}"
        
        if race_key != current_race:
            if current_race is not None:
                print()  # Add space between races
            print(f"\n📍 {horse['course']} - {horse['time']}")
            print("-" * 50)
            current_race = race_key
        
        # Format probability display
        base_prob = horse['base_probability'] * 100
        calib_prob = horse['win_probability'] * 100
        
        print(f"{horse['horse_name']:25} | {calib_prob:5.1f}% | (Base: {base_prob:4.1f}%)")
    
    # Calculate proper precision metrics
    total_predictions = len(significant_horses)
    races_with_predictions = significant_horses['race_id'].nunique()
    avg_predictions_per_race = total_predictions / races_with_predictions if races_with_predictions > 0 else 0
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total races analyzed: {df['race_id'].nunique()}")
    print(f"Races with predictions: {races_with_predictions}")
    print(f"Total predictions: {total_predictions}")
    print(f"Avg predictions per race: {avg_predictions_per_race:.1f}")
    print(f"Coverage: {races_with_predictions/df['race_id'].nunique()*100:.1f}% of races")
    print(f"Min probability threshold: {MIN_PROBABILITY_THRESHOLD:.0%}")
    print(f"Max predictions per race: {MAX_PREDICTIONS_PER_RACE}")
    if total_predictions > 0:
        print(f"Average probability: {significant_horses['win_probability'].mean()*100:.1f}%")
        print(f"Max probability: {significant_horses['win_probability'].max()*100:.1f}%")
    
    # Feature importance reminder
    print(f"\n🔍 MODEL FEATURES USED (Top 5):")
    importance_df = pd.read_csv(PROJECT_DIR / "models" / "v1" / "feature_importance_v1.csv")
    for _, row in importance_df.head(5).iterrows():
        print(f"• {row['feature']}: {row['importance']:,.0f}")
    
    print(f"\n💡 IMPORTANT NOTES:")
    print(f"• This model uses ONLY pre-race data (no leakage)")
    print(f"• Probabilities are properly calibrated")
    print(f"• Performance is conservative but realistic")
    print(f"• Consider this as ONE input to your betting strategy")
    
    # Save predictions for later analysis
    output_file = DATA_DIR / f"predictions_{datetime.now().strftime('%Y-%m-%d')}_calibrated.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Full predictions saved to: {output_file}")
    
    # Optional: Write to Google Sheets
    if GOOGLE_SHEETS_AVAILABLE and len(significant_horses) > 0:
        print(f"\n📊 GOOGLE SHEETS INTEGRATION:")
        try_sheets = input("Write predictions to Google Sheets? (y/n): ").lower().strip()
        if try_sheets in ['y', 'yes']:
            success = write_formatted_predictions_to_sheet(significant_horses)
            if not success:
                print("⚠️  Google Sheets write failed, but CSV file is still available")
    elif len(significant_horses) > 0:
        print(f"\n💡 To enable Google Sheets integration:")
        print(f"   pip install gspread google-auth")
        print(f"   Then set up Google Cloud credentials")

if __name__ == "__main__":
    predict_race_winners()
