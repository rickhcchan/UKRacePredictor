# scripts/common.py

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR  / 'data'
RPSCRAPE_DIR = PROJECT_DIR.parent.parent / 'joenano' / 'rpscrape'

top_hg = ['p', 't', 'b', 'h', 'v']
hg_map = {k: i+1 for i, k in enumerate(top_hg)}
hg_map[np.nan] = 0

def map_hg(val):
    if pd.isna(val):
        return 0
    elif val in hg_map:
        return hg_map[val]
    else:
        return len(top_hg) + 1

def parse_age_band(age_band):
    if pd.isna(age_band):
        return np.nan, np.nan
    age_band = age_band.replace('yo', '')
    if '+' in age_band:
        min_age = int(age_band.replace('+', ''))
        max_age = np.nan
    elif '-' in age_band:
        parts = age_band.split('-')
        min_age = int(parts[0])
        max_age = int(parts[1])
    else:
        min_age = max_age = int(age_band)
    return min_age, max_age
