"""
Model configuration loader for JSON-based model definitions.

This module provides utilities to load and validate model configurations
from JSON files, allowing for flexible model versioning and experimentation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class ModelConfig:
    """Model configuration loaded from JSON file."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}")
    
    def _validate_config(self):
        """Validate required configuration fields."""
        required_fields = ['model_name', 'features', 'training_params']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in {self.config_path}")
        
        # Validate features structure
        features = self.config['features']
        required_feature_types = ['categorical', 'ordinal', 'continuous']
        for feature_type in required_feature_types:
            if feature_type not in features:
                raise ValueError(f"Missing feature type '{feature_type}' in {self.config_path}")
            if not isinstance(features[feature_type], list):
                raise ValueError(f"Feature type '{feature_type}' must be a list in {self.config_path}")
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config['model_name']
    
    @property
    def description(self) -> str:
        """Get model description."""
        return self.config.get('description', '')
    
    @property
    def target_column(self) -> str:
        """Get target column name."""
        return self.config.get('target_column', 'target_win')
    
    @property
    def categorical_features(self) -> List[str]:
        """Get categorical feature names."""
        return self.config['features']['categorical']
    
    @property
    def ordinal_features(self) -> List[str]:
        """Get ordinal feature names."""
        return self.config['features']['ordinal']
    
    @property
    def continuous_features(self) -> List[str]:
        """Get continuous feature names."""
        return self.config['features']['continuous']
    
    @property
    def excluded_features(self) -> List[str]:
        """Get excluded feature names."""
        return self.config['features'].get('excluded', [])
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names (excluding excluded ones)."""
        return self.categorical_features + self.ordinal_features + self.continuous_features
    
    @property
    def training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.config['training_params']
    
    @property
    def validation_params(self) -> Dict[str, Any]:
        """Get validation parameters."""
        return self.config.get('validation', {
            'test_size': 0.2,
            'calibration_size': 0.2,
            'random_state': 42
        })
    
    def filter_available_features(self, available_features: List[str]) -> tuple:
        """
        Filter configured features to only include those available in data.
        
        Args:
            available_features: List of features available in the dataset
            
        Returns:
            tuple: (filtered_features, available_categorical_features, missing_features)
        """
        # Check which configured features are available
        configured_features = set(self.all_features)
        available_set = set(available_features)
        
        # Features that are configured and available
        filtered_features = list(configured_features & available_set)
        
        # Categorical features that are available
        available_categorical = [f for f in self.categorical_features if f in available_set]
        
        # Features that are configured but missing
        missing_features = list(configured_features - available_set)
        
        return filtered_features, available_categorical, missing_features


class ModelConfigLoader:
    """Utility class for loading model configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize model config loader.
        
        Args:
            config_dir: Directory containing model config files. 
                       Defaults to config/models/ relative to project root.
        """
        if config_dir is None:
            # Default to config/models/ relative to project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / 'config' / 'models'
        
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_model_config(self, model_name: str) -> ModelConfig:
        """
        Load model configuration for specified name.
        
        Args:
            model_name: Model name (e.g., 'v1', 'v2', 'default')
            
        Returns:
            ModelConfig: Loaded and validated model configuration
        """
        config_file = self.config_dir / f"{model_name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config for '{model_name}' not found: {config_file}"
            )
        
        self.logger.info(f"Loading model config from: {config_file}")
        return ModelConfig(str(config_file))
    
    def list_available_models(self) -> List[str]:
        """
        List all available model names.
        
        Returns:
            List[str]: List of available model names
        """
        if not self.config_dir.exists():
            return []
        
        model_files = self.config_dir.glob("*.json")
        return [f.stem for f in model_files]
    
    def get_default_model_name(self) -> str:
        """
        Get default model name.
        
        Returns:
            str: Default model name ('default', 'v1', or first available)
        """
        available_models = self.list_available_models()
        
        if not available_models:
            raise ValueError("No model configurations found")
        
        # Prefer 'default' if available, then 'v1', otherwise use first available
        if 'default' in available_models:
            return 'default'
        elif 'v1' in available_models:
            return 'v1'
        else:
            return sorted(available_models)[0]


def load_model_config(model_name: str, config_dir: Optional[str] = None) -> ModelConfig:
    """
    Convenience function to load model configuration.
    
    Args:
        model_name: Model name to load
        config_dir: Directory containing model configs (optional)
        
    Returns:
        ModelConfig: Loaded model configuration
    """
    loader = ModelConfigLoader(config_dir)
    return loader.load_model_config(model_name)
