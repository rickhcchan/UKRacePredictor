# scripts/common.py

import numpy as np
import pandas as pd

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