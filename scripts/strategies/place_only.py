"""
Place Only betting strategy implementation.

This strategy implements conservative betting logic:
- Minimum 20% probability for top horse
- Second place ratio maximum 80% of top horse
- Returns maximum 1 horse per race
- No field size limits
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from betting_strategy import BettingStrategy


class PlaceOnlyStrategy(BettingStrategy):
    """
    Conservative betting strategy for place bets only.
    
    Uses proven logic:
    - Top horse must have >= 20% probability
    - Second horse must be <= 80% of top horse probability
    - Returns at most 1 horse per race
    """
    
    def __init__(self):
        super().__init__(
            name="place_only",
            description="Place betting strategy: min 20% probability, max 80% second place ratio, 1 horse per race"
        )
    
    def select_horses(self, horses: List[Dict[str, Any]], race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select horses using place betting criteria.
        
        Logic:
        1. Sort horses by calibrated probability (descending)
        2. Check if top horse has >= 20% probability
        3. Check if second horse is <= 80% of top horse probability
        4. Return top horse if both conditions met, otherwise empty list
        
        Args:
            horses: All horses in the race with predictions
            race_data: Race information (not used in current logic)
            
        Returns:
            List containing top horse if criteria met, otherwise empty list
        """
        # Need at least 1 horse to consider
        if not horses:
            return []
        
        # Sort horses by calibrated probability (highest first)
        sorted_horses = sorted(
            horses, 
            key=lambda h: h.get('calibrated_probability', 0.0), 
            reverse=True
        )
        
        top_horse = sorted_horses[0]
        top_probability = top_horse.get('calibrated_probability', 0.0)
        
        # Check minimum probability threshold (20%)
        if top_probability < 0.20:
            return []
        
        # If only one horse, return it (no second place to compare)
        if len(sorted_horses) == 1:
            return [top_horse]
        
        # Check second place ratio
        second_horse = sorted_horses[1]
        second_probability = second_horse.get('calibrated_probability', 0.0)
        
        # Avoid division by zero
        if top_probability == 0.0:
            return []
        
        second_place_ratio = second_probability / top_probability
        
        # Return top horse only if second place ratio <= 80%
        if second_place_ratio <= 0.80:
            return [top_horse]
        
        return []
