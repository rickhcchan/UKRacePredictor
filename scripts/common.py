# scripts/common.py
# Shared utilities for the UK Race Predictor scripts

import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def convert_to_24h_time(time_str):
    """
    Convert race time to 24-hour format for proper chronological sorting.
    
    Logic:
    - Morning times (11:00, 12:00) stay as-is
    - Afternoon times (1:00-10:00) get converted to 13:00-22:00
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        24-hour format time string (HH:MM)
    """
    try:
        time_part = datetime.strptime(time_str, '%H:%M').time()
        hour = time_part.hour
        if hour in [11, 12]:
            final_hour = hour
        else:
            final_hour = hour + 12 if hour <= 10 else hour
        return f"{final_hour:02d}:{time_part.minute:02d}"
    except:
        return time_str  # Return original if conversion fails
