"""
Abstract base class for betting strategies.

This module defines the interface for betting strategies that select horses
to bet on based on model predictions and race data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BettingStrategy(ABC):
    """
    Abstract base class for betting strategies.
    
    A betting strategy takes model predictions and race data as input
    and returns a list of horses to bet on for a single race.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the betting strategy.
        
        Args:
            name: Strategy name identifier
            description: Human-readable description of the strategy
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def select_horses(self, horses: List[Dict[str, Any]], race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select horses to bet on for a single race.
        
        This method is called once per race with all horses in that race.
        
        Args:
            horses: List of horse dictionaries containing:
                - horse_id: Unique identifier
                - horse_name: Horse name
                - calibrated_probability: Model prediction (0.0 to 1.0)
                - All racecard fields (jockey, trainer, weight, etc.)
            
            race_data: Race-level information containing:
                - race_id: Unique race identifier
                - course_name: Race course
                - race_time: Race start time
                - distance: Race distance
                - field_size: Number of runners
                - Other race metadata
        
        Returns:
            List of horse dictionaries to bet on. Can be:
            - Empty list: No bets for this race
            - Single horse: One bet
            - Multiple horses: Multiple bets in same race
        """
        pass
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get basic information about this strategy.
        
        Returns:
            Dictionary with name and description
        """
        return {
            'name': self.name,
            'description': self.description
        }
