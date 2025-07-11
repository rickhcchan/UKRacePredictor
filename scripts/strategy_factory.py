"""
Strategy factory for creating and managing betting strategies.

This module provides a centralized way to create and list available
betting strategies from the strategies/ directory.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Type

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from betting_strategy import BettingStrategy


def _load_strategy_class(strategy_name: str) -> Type[BettingStrategy]:
    """Dynamically load a strategy class from strategies/ directory."""
    strategies_dir = Path(__file__).parent / 'strategies'
    strategy_file = strategies_dir / f'{strategy_name}.py'
    
    if not strategy_file.exists():
        raise ValueError(f"Strategy file not found: {strategy_file}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(strategy_name, strategy_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the strategy class (assumes class name is CapitalCase version of file name)
    class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
    
    if not hasattr(module, class_name):
        raise ValueError(f"Strategy class {class_name} not found in {strategy_file}")
    
    return getattr(module, class_name)


class StrategyFactory:
    """
    Factory class for creating betting strategies.
    
    Automatically discovers and manages strategies from the strategies/ directory.
    """
    
    @classmethod
    def _get_available_strategy_files(cls) -> List[str]:
        """Get list of available strategy files."""
        strategies_dir = Path(__file__).parent / 'strategies'
        if not strategies_dir.exists():
            return []
        
        strategy_files = []
        for file in strategies_dir.glob('*.py'):
            if file.name != '__init__.py':
                strategy_files.append(file.stem)
        return strategy_files
    
    @classmethod
    def create_strategy(cls, strategy_name: str) -> BettingStrategy:
        """
        Create a betting strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy to create (e.g., 'default', 'place_only')
            
        Returns:
            Instance of the requested betting strategy
            
        Raises:
            ValueError: If strategy_name is not recognized
        """
        available = cls._get_available_strategy_files()
        if strategy_name not in available:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")
        
        strategy_class = _load_strategy_class(strategy_name)
        return strategy_class()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get list of all available strategy names.
        
        Returns:
            List of strategy names that can be created
        """
        return cls._get_available_strategy_files()
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict:
        """
        Get information about a strategy without creating an instance.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information
            
        Raises:
            ValueError: If strategy_name is not recognized
        """
        strategy = cls.create_strategy(strategy_name)
        return strategy.get_strategy_info()
    
    @classmethod
    def list_all_strategies(cls) -> Dict[str, Dict]:
        """
        Get information about all available strategies.
        
        Returns:
            Dictionary mapping strategy names to their information
        """
        strategies_info = {}
        for strategy_name in cls.get_available_strategies():
            strategies_info[strategy_name] = cls.get_strategy_info(strategy_name)
        return strategies_info
