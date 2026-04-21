"""
Adds virality metrics:
    - z-score of TI (total interactions)
    - z-score of S (spread)
Adds virality labels:
    - viral_S_TI (both z-scores >= 3)
    - viral_speed_S (reposts after x hours above threshold)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

eps = 1 # Laplace smoothing

def add_virality_metrics(dataset: pd.DataFrame):
    dataset["TI_z"] = 