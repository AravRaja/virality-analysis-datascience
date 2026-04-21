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

BASE_PATH = Path.cwd().parent

def get_values_for_thresholds(path = BASE_PATH/"datasets/bluesky_cascade/values_for_virality_thresholds.parquet"):
    """
    Returns: "Time_Period", "Threshold_Reposts_h", "TI_mean", "TI_std", "S_mean", "S_std"
    """
    df = pd.read_csv(path)
    return tuple(df.iloc[0])

def add_virality_metrics(dataset: pd.DataFrame, chosen_time_period, threshold_reposts_h, TI_mean, TI_std, S_mean, S_std, eps=1):
    dataset["TI_z"] = (np.log(dataset["like_count"] + dataset["reply_count"] + dataset["repost_count"] + eps) - TI_mean) / TI_std
    dataset["S_z"] = (np.log(((dataset["repost_count"] + eps) / (dataset["author_followers"] + eps)) + eps) - S_mean) / S_std
    dataset["viral_TI_S"] = (dataset["TI_z"] >= 3) & (dataset["S_z"] >= 3)
    dataset["viral_speed_S"] = dataset[f"repost_{chosen_time_period}h"] >= threshold_reposts_h
    return dataset

if __name__ == "main":
    file_path_in = None # FILL IN
    file_path_out = None # FILL IN
    training_set = pd.read_parquet(file_path_in)
    chosen_time_period, threshold_reposts_h, TI_mean, TI_std, S_mean, S_std = get_values_for_thresholds()
    processed_training_set = add_virality_metrics(training_set , chosen_time_period, threshold_reposts_h, TI_mean, TI_std, S_mean, S_std)
    print("Added virality metrics")
    processed_training_set.to_parquet(file_path_out)
    print("saved to file")