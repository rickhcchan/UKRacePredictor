# scripts/predict.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from common import DATA_DIR, PROJECT_DIR, convert_to_24h_time
# Google Sheets integration removed for now

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
    
    # Enhanced Strategy: Cherry-pick races with strong favorites and show multiple horses
    MIN_CALIB_THRESHOLD = 0.20  # 20% calibrated probability minimum (X)
    
    # Calculate dynamic normalized threshold (Y) based on number of runners
    def calculate_norm_threshold(num_runners):
        # Optimized formula: Y = 1.5 × (100/num_runners)
        # This ensures we're always well above random chance
        # 4 horses: 1.5 × 25% = 37.5% (vs random 25%)
        # 8 horses: 1.5 × 12.5% = 18.75% (vs random 12.5%)
        # 12 horses: 1.5 × 8.33% = 12.5% (vs random 8.33%)
        factor = 1.5  # 50% above random chance (optimal from backtest)
        threshold = factor * (100 / num_runners) / 100  # Convert to decimal
        return threshold
    
    MIN_HORSES_PER_RACE = 3     # Show at least 3 horses when race qualifies
    
    all_race_predictions = []
    qualifying_races = []
    
    # Process each race
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
        
        # Always collect all horses for full output file
        all_race_predictions.append(race_sorted)
    
    # Combine all predictions for CSV output
    all_predictions = pd.concat(all_race_predictions, ignore_index=True) if all_race_predictions else pd.DataFrame()
    
    # Combine qualifying races for display
    if qualifying_races:
        significant_horses = pd.concat(qualifying_races, ignore_index=True)
    else:
        significant_horses = pd.DataFrame()
    
    if len(significant_horses) == 0:
        print(f"\n❌ No races found with top horse meeting EITHER criteria:")
        print(f"   • Calibrated probability >{MIN_CALIB_THRESHOLD:.0%} OR")
        print(f"   • Normalized probability >dynamic threshold (1.5 × random chance)")
        print("💡 Showing top 3 horses from each race...")
        # Show top 3 horses per race as fallback
        fallback_races = []
        for race_id, race_group in df.groupby('race_id'):
            top_3 = race_group.nlargest(3, 'win_probability_normalized')
            fallback_races.append(top_3)
        significant_horses = pd.concat(fallback_races, ignore_index=True) if fallback_races else df.head(0)
        print(f"📊 Fallback: Top 3 horses per race")
    
    # Sort by 24-hour time first (chronological order), then course, then normalized probability
    significant_horses = significant_horses.sort_values(['time_24h', 'course', 'win_probability_normalized'], ascending=[True, True, False])
    
    print(f"\n🎯 CHERRY-PICKED RACES (At least 1 horse: Calibrated >{MIN_CALIB_THRESHOLD:.0%} OR normalized >dynamic threshold):")
    print("💡 BET LEGEND: ✅ BET = Qualifies (Calibrated >20% OR normalized >dynamic%), 📋 REF = Reference only")
    print("=" * 90)
    
    current_race = None
    for _, horse in significant_horses.iterrows():
        race_key = f"{horse['course']} {horse['time']}"
        
        if race_key != current_race:
            if current_race is not None:
                print()  # Add space between races
            
            # Count total horses in this race
            race_horses = significant_horses[
                (significant_horses['course'] == horse['course']) & 
                (significant_horses['time'] == horse['time'])
            ]
            total_horses_in_race = len(df[df['race_id'] == horse['race_id']])
            
            # Calculate and display the dynamic threshold for this race
            dynamic_threshold = calculate_norm_threshold(total_horses_in_race) * 100  # Convert to percentage
            
            print(f"\n📍 {horse['course']} - {horse['time']} ({total_horses_in_race} horses total)")
            print(f"   Qualifies: Calibrated >20% OR normalized >{dynamic_threshold:.1f}%")
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
        
        print(f"{horse['horse_name']:25} | {base_prob:5.1f}% | {calib_prob:6.1f}% | {norm_prob:5.1f}% | {bet_indicator:>6}")
    
    # Calculate proper precision metrics
    total_predictions = len(significant_horses)
    races_with_predictions = significant_horses['race_id'].nunique() if len(significant_horses) > 0 else 0
    total_races = df['race_id'].nunique()
    
    # Summary statistics
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
    
    # Save predictions for later analysis (use all predictions, not just significant ones)
    final_df = all_predictions if len(all_predictions) > 0 else df
    output_file = DATA_DIR / f"predictions_{datetime.now().strftime('%Y-%m-%d')}_calibrated.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n💾 Full predictions saved to: {output_file}")

if __name__ == "__main__":
    predict_race_winners()
