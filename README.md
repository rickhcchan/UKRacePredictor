# UK Race Predictor

A machine learning system for predicting UK horse race outcomes using LightGBM. The system processes historical race data, engineers features, trains models, and generates predictions for upcoming races.

## 🐎 What This Project Does

The UK Race Predictor is a complete end-to-end machine learning pipeline that:

- **Collects** historical UK horse racing data using [rpscrape](https://github.com/4A47/rpscrape)
- **Processes** and encodes race data into features for machine learning
- **Engineers** sophisticated features including:
  - Horse historical performance (win rates, course/distance/going preferences)
  - Jockey and trainer statistics (recent form, course specialties)
  - 14-day recent form indicators
  - Time-based features (seasonal patterns)
- **Trains** LightGBM models with probability calibration
- **Predicts** race outcomes with calibrated win probabilities
- **Supports** multiple model versions and betting strategies

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **rpscrape** - Clone and set up [rpscrape](https://github.com/4A47/rpscrape) for data collection

### Installation

1. Clone this repository:
```bash
git clone https://github.com/rickhcchan/UKRacePredictor.git
cd UKRacePredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration (see Configuration section below)

## ⚙️ Configuration

### Setting Up Custom Configuration

1. **Copy the template:**
```bash
cp config/user_settings.conf.template config/user_settings.conf
```

2. **Edit your personal settings:**
```ini
# config/user_settings.conf
[common]
# Path to your rpscrape installation
rpscrape_dir = /path/to/your/rpscrape

# Adjust timeout if needed (seconds)
timeout = 60

# Optional: Change data locations
db_path = data/race_data.db
data_dir = data
models_dir = models
```

### Configuration Files

- **`config/default_settings.conf`** - Default settings (version controlled)
- **`config/user_settings.conf`** - Your personal settings (not in git)
- **`config/user_settings.conf.template`** - Template for personal settings

The system loads `user_settings.conf` first, falling back to `default_settings.conf` for missing values.

## 📊 Pipeline Overview

The system consists of 5 main scripts that form a complete ML pipeline:

```mermaid
graph LR
    A[update_race_data.py] --> B[encode_incremental.py]
    B --> C[train.py]
    B --> D[prepare_racecard.py]
    D --> E[predict_races.py]
    C -.-> E
```

**Pipeline Flow:**
- `update_race_data.py` → `encode_incremental.py` → Data preparation
- `encode_incremental.py` → `train.py` → Model training (optional, when retraining)
- `encode_incremental.py` → `prepare_racecard.py` → `predict_races.py` → Daily predictions

All scripts support `--dry-run` for testing and use minimal, essential parameters for clean operation.

## 🔄 Usage Guide

### 1. Initial Setup (First Time Only)

```bash
# Update historical data (may take time for initial download)
python scripts/update_race_data.py

# Encode all historical data into features
python scripts/encode_incremental.py --force-rebuild

# Train the default model
python scripts/train.py
```

### 2. Daily Prediction Workflow

```bash
# Step 1: Update historical data (adds yesterday's results)
python scripts/update_race_data.py

# Step 2: Encode new data into features  
python scripts/encode_incremental.py

# Step 3: Prepare today's racecard (or tomorrow's)
python scripts/prepare_racecard.py                    # Today
python scripts/prepare_racecard.py --date 2025-07-12  # Tomorrow

# Step 4: Generate predictions (uses default model and strategy)
python scripts/predict_races.py                       # Today
python scripts/predict_races.py --date 2025-07-12     # Tomorrow
```

### 3. Use Custom Models

```bash
# Retrain default model with new data
python scripts/train.py

# Train custom model versions
python scripts/train.py --model v1
python scripts/train.py --model experimental

# Use custom models
python scripts/train.py --model my_model
python scripts/predict_races.py --model my_model
```

## 📋 Script Reference

### `update_race_data.py`
Downloads and stores historical race results using rpscrape.

```bash
# Update to current date (default behavior)
python scripts/update_race_data.py

# Test run without changes
python scripts/update_race_data.py --dry-run
```

### `encode_incremental.py`
Processes raw race data into machine learning features.

```bash
# Encode new data since last run
python scripts/encode_incremental.py

# Force complete rebuild (after schema changes)
python scripts/encode_incremental.py --force-rebuild

# Test run without changes
python scripts/encode_incremental.py --dry-run
```

### `train.py`
Trains LightGBM models with probability calibration using JSON configurations.

```bash
# Train default model
python scripts/train.py

# Train specific model configuration
python scripts/train.py --model custom

# Test run without training
python scripts/train.py --dry-run
```

### `prepare_racecard.py`
Downloads and prepares racecard for prediction with feature engineering.

**Note:** Can only download racecards for today or tomorrow (rpscrape limitation).

```bash
# Prepare today's racecard (default)
python scripts/prepare_racecard.py

# Prepare tomorrow's racecard
python scripts/prepare_racecard.py --date 2025-07-12

# Test run without changes
python scripts/prepare_racecard.py --dry-run

# Dry run for specific date
python scripts/prepare_racecard.py --date 2025-07-12 --dry-run
```

### `predict_races.py`
Generates win probability predictions for prepared racecards using JSON model configurations.

```bash
# Predict with default model and default strategy
python scripts/predict_races.py

# Predict with specific model
python scripts/predict_races.py --model custom

# Test run without saving files
python scripts/predict_races.py --dry-run

# Predict with specific strategy
python scripts/predict_races.py --strategy default

# Predict with both custom model and strategy
python scripts/predict_races.py --model custom --strategy default

# Predict specific date
python scripts/predict_races.py --date 2025-07-08

# Combined options
python scripts/predict_races.py --date 2025-07-08 --model custom --strategy default --dry-run
```

## 📁 Project Structure

```
UKRacePredictor/
├── README.md
├── requirements.txt
├── config/
│   ├── default_settings.conf     # Default system settings
│   ├── user_settings.conf.template
│   └── models/
│       └── default.json          # Default model configuration
├── scripts/
│   ├── update_race_data.py       # Data collection
│   ├── encode_incremental.py     # Feature engineering
│   ├── train.py                  # Model training
│   ├── prepare_racecard.py       # Racecard preparation
│   ├── predict_races.py          # Prediction generation
│   ├── betting_strategy.py       # Strategy base class
│   ├── strategy_factory.py       # Strategy factory
│   └── strategies/
│       └── default.py            # Default betting strategy
├── data/                         # Data files (not in git)
├── models/                       # Trained models (not in git)
└── docs/
```

## 🎯 Model Configuration

### Using Different Model Configurations

```bash
# Train different models
python scripts/train.py                           # Uses default.json
python scripts/train.py --model custom            # Uses custom.json  
python scripts/train.py --model experimental      # Uses experimental.json

# Predict with specific models
python scripts/predict_races.py                   # Uses default model
python scripts/predict_races.py --model custom    # Uses custom model
python scripts/predict_races.py --model experimental # Uses experimental model

# Quick start for simple workflow - just edit default.json
python scripts/train.py      # Train with your custom default.json
python scripts/predict_races.py  # Predict with your custom default.json
```

### JSON Configuration Format

Model configurations are stored in `config/models/*.json`:

```json
{
  "model_name": "default",
  "description": "Default model configuration",
  "features": {
    "categorical": ["course_id", "type_id", "sex_id"],
    "ordinal": ["age", "lbs", "dist_f", "ran", "draw"]
  },
  "training_params": {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05
  },
  "validation": {
    "test_size": 0.2,
    "calibration_size": 0.2,
    "random_state": 42
  }
}
```

### Creating Custom Model Configurations

1. **Create a new model config file:**
```bash
# Create your model config in config/models/
cp config/models/default.json config/models/my_model.json
```

2. **Edit the configuration:**
```json
{
  "model_name": "my_model",
  "description": "Custom model with specific features",
  "features": {
    "categorical": [
      "course_id", "type_id", "sex_id", "going_id", "jockey_id", "trainer_id"
    ],
    "ordinal": [
      "age", "lbs", "dist_f", "ran", "draw", "or_rating", "ts_rating"
    ],
    "historical": [
      "horse_win_rate_14d", "jockey_win_rate_14d", "trainer_win_rate_14d"
    ]
  },
  "training_params": {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
  },
  "validation": {
    "test_size": 0.2,
    "calibration_size": 0.2,
    "random_state": 42
  }
}
```

3. **Use your custom model:**
```bash
# Train with custom config
python scripts/train.py --model my_model

# Predict with custom model
python scripts/predict_races.py --model my_model
```

### Model Configuration Guidelines

**Feature Categories:**
- **`categorical`** - Text/ID features (course, jockey, trainer, etc.)
- **`ordinal`** - Numeric features with meaningful order (age, weight, rating)
- **`historical`** - Time-based performance features (win rates, form)

**Key Training Parameters:**
- **`num_leaves`** - Model complexity (31-127, higher = more complex)
- **`learning_rate`** - Training speed (0.01-0.1, lower = more stable)
- **`feature_fraction`** - Random feature sampling (0.6-1.0)
- **`min_child_samples`** - Minimum samples per leaf (10-50)

**Performance Tips:**
- Start with `default.json` and modify incrementally
- Test with `--dry-run` before full training
- Monitor validation scores during training
- Use fewer features for faster predictions

### Multiple Model Workflow

```bash
# Train different model versions
python scripts/train.py --model custom     # Uses config/models/custom.json
python scripts/train.py --model experimental # Uses config/models/experimental.json

# Predict with specific version
python scripts/predict_races.py --model custom  # Uses custom config and models
python scripts/predict_races.py --model experimental  # Uses experimental config and models
```

## 🎯 Betting Strategies

The system supports pluggable betting strategies that determine which horses to bet on based on model predictions.

### Available Strategies

- **`default`** - Conservative strategy: min 20% probability, returns highest probability horse only if unique

### Using Strategies

```bash
# Use default strategy
python scripts/predict_races.py --strategy default

# Combine with specific model
python scripts/predict_races.py --model custom --strategy default
```

### Creating Custom Strategies

1. **Create a new strategy file:**
```bash
# Create your strategy in scripts/strategies/
touch scripts/strategies/mystrategy.py
```

2. **Implement the strategy class:**
```python
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from betting_strategy import BettingStrategy

class MystrategyStrategy(BettingStrategy):  # Class name must be: CapitalCase + "Strategy"
    def __init__(self):
        super().__init__(
            name="mystrategy",  # File name without .py
            description="Custom betting strategy: min 25% probability, max 2 horses per race"
        )
    
    def select_horses(self, horses: List[Dict[str, Any]], race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement your betting logic here.
        
        Args:
            horses: List of horses with predictions and racecard data
            race_data: Race-level information (course, time, field size, etc.)
            
        Returns:
            List of horses to bet on (can be empty, single, or multiple)
        """
        if not horses:
            return []
        
        # Filter horses above probability threshold
        threshold = 0.25
        candidates = [
            horse for horse in horses 
            if horse.get('calibrated_probability', 0) >= threshold
        ]
        
        if not candidates:
            return []
        
        # Sort by probability (highest first)
        candidates.sort(key=lambda h: h.get('calibrated_probability', 0), reverse=True)
        
        # Return top 2 horses maximum
        return candidates[:2]
```

3. **Advanced strategy example:**
```python
class AdvancedStrategy(BettingStrategy):
    def __init__(self):
        super().__init__(
            name="advanced",
            description="Advanced strategy: probability + race context filters"
        )
    
    def select_horses(self, horses: List[Dict[str, Any]], race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not horses:
            return []
        
        # Get race context
        field_size = len(horses)
        race_type = race_data.get('type', '').lower()
        
        # Adjust strategy based on race type
        if 'handicap' in race_type:
            min_prob = 0.15  # Lower threshold for handicaps
            max_horses = 3
        else:
            min_prob = 0.20  # Higher threshold for non-handicaps  
            max_horses = 2
        
        # Filter by field size
        if field_size > 16:
            min_prob += 0.05  # Higher threshold for large fields
        
        # Select horses
        candidates = [
            horse for horse in horses
            if horse.get('calibrated_probability', 0) >= min_prob
        ]
        
        # Sort by probability and limit
        candidates.sort(key=lambda h: h.get('calibrated_probability', 0), reverse=True)
        return candidates[:max_horses]
```

4. **Use your strategy:**
```bash
python scripts/predict_races.py --strategy mystrategy
python scripts/predict_races.py --strategy advanced
```

### Strategy Naming Convention

**Important**: The strategy factory expects specific naming:
- **File name**: `mystrategy.py` (lowercase, underscores allowed)
- **Class name**: `MystrategyStrategy` (capitalize each word, remove underscores, add "Strategy")
- **Strategy name**: `mystrategy` (same as file name without .py)

**Examples:**
- `mystrategy.py` → `MystrategyStrategy` → `--strategy mystrategy`
- `custom_bet.py` → `CustomBetStrategy` → `--strategy custom_bet`

### Strategy Guidelines

**Available Horse Data:**
- **`calibrated_probability`** - Main prediction (0.0-1.0)
- **`horse_name`** - Horse name
- **`jockey_name`** - Jockey name  
- **`trainer_name`** - Trainer name
- **`age`** - Horse age
- **`lbs`** - Weight carried
- **`or_rating`** - Official rating
- **`ts_rating`** - Timeform rating
- **`draw`** - Starting position

**Available Race Data:**
- **`course`** - Race course name
- **`time`** - Race time
- **`type`** - Race type (handicap, maiden, etc.)
- **`distance`** - Race distance
- **`going`** - Ground conditions

**Best Practices:**
- **Keep it simple** - Complex logic can be hard to debug
- **Use calibrated_probability** - This is the main prediction value
- **Consider race context** - Use race_data for filtering (field size, race type)
- **Return empty list** if no bets recommended
- **Test thoroughly** with different scenarios
- **Start conservative** - Begin with higher probability thresholds
- **Log your logic** - Add print statements for debugging

**Common Strategy Patterns:**
```python
# Probability threshold
candidates = [h for h in horses if h.get('calibrated_probability', 0) >= 0.20]

# Top N horses
horses_sorted = sorted(horses, key=lambda h: h.get('calibrated_probability', 0), reverse=True)
return horses_sorted[:2]

# Conditional logic
if len(horses) > 12:  # Large field
    threshold = 0.25
else:  # Small field  
    threshold = 0.15

# Race type filtering
if 'handicap' in race_data.get('type', '').lower():
    # Different logic for handicaps
    pass
```

## 🔧 Development

### Adding New Features

1. **Historical features** - Modify `encode_incremental.py`
2. **Model parameters** - Edit JSON configs in `config/models/`
3. **Betting logic** - Create new strategy in `scripts/strategies/`

### Testing

```bash
# Test all scripts with dry-run
python scripts/update_race_data.py --dry-run
python scripts/encode_incremental.py --dry-run
python scripts/train.py --dry-run
python scripts/prepare_racecard.py --dry-run
python scripts/predict_races.py --dry-run

# Test strategy system
python scripts/test_strategy_system.py
```

## 📝 Important Notes

### Data Leakage Prevention
- All historical features use `race_date < target_date` filtering
- No same-day data is used for feature calculation
- 14-day windows strictly exclude the prediction date

### File Management
- **CSV outputs are optional** - Use `--dry-run` for testing without file changes
- **Models are not in git** - Train locally for your data
- **Strategies are private** - Only `default.py` is version controlled

### Performance Tips
- **Incremental encoding** is much faster than full rebuilds
- **Feature lists** in model configs control prediction speed
- **Probability calibration** significantly improves accuracy

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Test with `--dry-run` options
4. Ensure all scripts work independently
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [rpscrape](https://github.com/4A47/rpscrape) for data collection capabilities
- Racing Post for providing the underlying data source
- LightGBM team for the excellent gradient boosting framework
