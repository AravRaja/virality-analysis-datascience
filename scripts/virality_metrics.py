"""
Implements virality definitions
"""

import pandas as pd
import numpy as np

eps_counts = 0
eps_ratio = 1e-6

# Total interaction and spread z scores
def get_S_TI_z_scores(post, follower_counts, S_mean, S_std, TI_mean, TI_std):
    TI_z_score = (np.log(post.like_count + post.reply_count + post.repost_count + eps_counts) - TI_mean) / TI_std
    S_z_score = ((np.log(post.repost_count / (follower_counts.follower_count + eps_ratio)) + eps_counts) - S_mean) / S_std
    return TI_z_score, S_z_score

# Virality definition, based on total interaction and spread
def is_Viral_TI_S(post, follower_counts, S_mean, S_std, TI_mean, TI_std):
    TI_z_score, S_z_score = get_S_TI_z_scores(post, follower_counts, S_mean, S_std, TI_mean, TI_std)
