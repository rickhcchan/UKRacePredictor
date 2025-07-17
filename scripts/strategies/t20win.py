from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from betting_strategy import BettingStrategy


class T20winStrategy(BettingStrategy):
    def __init__(self):
        super().__init__(
            name="t20win",
            description="T20 Win strategy: min 20% probability, max 80% second place ratio, 1 horse per race"
        )
    
    def select_horses(self, horses: List[Dict[str, Any]], race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Need at least 1 horse to consider
        if not horses:
            return []
        
        # Find the highest probability
        max_probability = max(horse.get('calibrated_probability', 0) for horse in horses)
        
        # Only proceed if the highest probability meets our threshold
        if max_probability < 0.20:
            return []
        
        top_horses = [horse for horse in horses if horse.get('calibrated_probability', 0.0) == max_probability]

        if len(top_horses) < len(horses) / 2:
            return top_horses
        
        return []
