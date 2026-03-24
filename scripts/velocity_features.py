"""
step_02_early_like_features.py

Computes post-level early activity features from timestamped likes:

    Layer 1 - Like Velocity & Acceleration:
    likes_5m, likes_15m, likes_30m
    likes_15_to_30m, like_acceleration, like_velocity_ratio

    Layer 2 - Network Quality of Early Likers:
    max_liker_followers, mean_liker_followers, median_liker_followers
    sum_liker_reach, whale_count, reach_concentration

Inputs:  features/base_posts.parquet
         features/base_likes.parquet
         features/follower_counts.parquet  (from step_01)

Outputs: features/early_like_features.parquet
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = os.path.join("../data", "processed", "features")

# Time windows
WINDOW_5M  = pd.Timedelta(minutes=5)
WINDOW_15M = pd.Timedelta(minutes=15)
WINDOW_30M = pd.Timedelta(minutes=30)

# A "whale" is an account with more followers than this
# (75th percentile of ALL users is 14, so this is well above typical)
WHALE_THRESHOLD = 1000

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 70)
print("STEP 1: Loading foundation data")
print("=" * 70)

posts = pd.read_parquet(os.path.join(FEATURES_DIR, "base_posts.parquet"))
likes = pd.read_parquet(os.path.join(FEATURES_DIR, "base_likes.parquet"))
follower_counts = pd.read_parquet(os.path.join(FEATURES_DIR, "follower_counts.parquet"))

print(f"  Posts loaded:           {len(posts):>12,}")
print(f"  Likes loaded:           {len(likes):>12,}")
print(f"  Follower counts loaded: {len(follower_counts):>12,}")

# Ensure datetime types
posts['created_at'] = pd.to_datetime(posts['created_at'])
likes['liked_at'] = pd.to_datetime(likes['liked_at'])

# =============================================================================
# STEP 2: JOIN LIKES TO POST CREATION TIMES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Joining likes to post creation times")
print("=" * 70)

# Merge only the columns we need
likes_with_time = likes.merge(
    posts[['post_id', 'created_at']],
    on='post_id',
    how='inner'
)

print(f"  Likes matched to posts: {len(likes_with_time):,}")

# Compute time delta once - reused for all window filters
likes_with_time['minutes_after'] = (
    (likes_with_time['liked_at'] - likes_with_time['created_at'])
    .dt.total_seconds() / 60.0
)

# =============================================================================
# STEP 3: FILTER TO 30-MINUTE WINDOW
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Filtering to early window (0-30 minutes)")
print("=" * 70)

# Apply two-sided filter:
#   >= 0  (exclude likes before post creation - clock skew / data artifacts)
#   <= 30 (our maximum early window)
early_mask = (
    (likes_with_time['minutes_after'] >= 0) &
    (likes_with_time['minutes_after'] <= 30)
)

early_likes = likes_with_time[early_mask].copy()

# Stats on what we kept vs filtered
total_likes = len(likes_with_time)
early_count = len(early_likes)
before_post = (likes_with_time['minutes_after'] < 0).sum()
after_30m = (likes_with_time['minutes_after'] > 30).sum()

print(f"  Total likes with timestamps:   {total_likes:>12,}")
print(f"  Likes before post creation:    {before_post:>12,} (excluded)")
print(f"  Likes within 0-30m:            {early_count:>12,} <- kept")
print(f"  Likes after 30m:               {after_30m:>12,} (excluded)")
print(f"  Early like rate:               {early_count/total_likes*100:.1f}%")

# Posts that received at least one early like
posts_with_early_likes = early_likes['post_id'].nunique()
print(f"\n  Posts with >=1 early like: {posts_with_early_likes:,} / {len(posts):,} "
      f"({posts_with_early_likes/len(posts)*100:.1f}%)")

# Free the full likes table from memory
del likes_with_time, likes

# =============================================================================
# STEP 4: COMPUTE LIKE VELOCITY FEATURES (Layer 1)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Computing like velocity & acceleration features")
print("=" * 70)

# ---- Count likes in each window ----
# 30m window (all early likes, already filtered)
likes_30m = (
    early_likes
    .groupby('post_id')
    .size()
    .reset_index(name='likes_30m')
)

# 15m window (subset of 30m - no re-merge needed)
likes_15m = (
    early_likes[early_likes['minutes_after'] <= 15]
    .groupby('post_id')
    .size()
    .reset_index(name='likes_15m')
)

# 5m window (even tighter - captures the immediate burst)
likes_5m = (
    early_likes[early_likes['minutes_after'] <= 5]
    .groupby('post_id')
    .size()
    .reset_index(name='likes_5m')
)

# ---- Combine into one dataframe ----
velocity_features = likes_30m.copy()
velocity_features = velocity_features.merge(likes_15m, on='post_id', how='left')
velocity_features = velocity_features.merge(likes_5m, on='post_id', how='left')

# Fill NaN (posts with 30m likes but no 15m or 5m likes)
velocity_features['likes_15m'] = velocity_features['likes_15m'].fillna(0).astype(int)
velocity_features['likes_5m'] = velocity_features['likes_5m'].fillna(0).astype(int)

# ---- Derived features ----

# Second-half volume: likes between 15m and 30m
velocity_features['likes_15_to_30m'] = (
    velocity_features['likes_30m'] - velocity_features['likes_15m']
)

# First-half volume: likes between 5m and 15m (for finer acceleration)
velocity_features['likes_5_to_15m'] = (
    velocity_features['likes_15m'] - velocity_features['likes_5m']
)

# Acceleration: is engagement speeding up or slowing down?
# Positive = accelerating, Negative = decelerating
velocity_features['like_acceleration'] = (
    velocity_features['likes_15_to_30m'] - velocity_features['likes_15m']
)

# Velocity ratio: second half / first half
# > 1 means accelerating, < 1 means decelerating
velocity_features['like_velocity_ratio'] = (
    velocity_features['likes_15_to_30m'] / (velocity_features['likes_15m'] + 1)
)

# Early burst ratio: what fraction of 30m likes came in the first 5 minutes?
velocity_features['early_burst_ratio'] = (
    velocity_features['likes_5m'] / (velocity_features['likes_30m'] + 1)
)

# Print stats
print("\n  Velocity feature statistics:")
for col in ['likes_5m', 'likes_15m', 'likes_30m', 'like_acceleration', 
            'like_velocity_ratio', 'early_burst_ratio']:
    stats = velocity_features[col]
    print(f"    {col:25s}  mean={stats.mean():>8.2f}  "
          f"median={stats.median():>6.1f}  "
          f"max={stats.max():>8.0f}")

# =============================================================================
# STEP 5: COMPUTE NETWORK QUALITY FEATURES (Layer 2)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Computing network quality of early likers")
print("=" * 70)

# Join early likes with follower counts of the LIKERS
early_likes_with_network = early_likes.merge(
    follower_counts[['user_id', 'follower_count']],
    on='user_id',  # user_id in likes = the liker
    how='left'
)

# Check coverage
n_matched = early_likes_with_network['follower_count'].notna().sum()
n_total = len(early_likes_with_network)
print(f"  Early likers with follower data: {n_matched:,} / {n_total:,} "
      f"({n_matched/n_total*100:.1f}%)")

# Fill missing follower counts with 0 (unknown users assumed small)
early_likes_with_network['follower_count'] = (
    early_likes_with_network['follower_count'].fillna(0)
)

# ---- Aggregate per post ----
network_features = (
    early_likes_with_network
    .groupby('post_id')['follower_count']
    .agg(
        max_liker_followers='max',
        mean_liker_followers='mean',
        median_liker_followers='median',
        sum_liker_reach='sum',
    )
    .reset_index()
)

# ---- Whale count: large accounts that liked early ----
whale_likes = early_likes_with_network[
    early_likes_with_network['follower_count'] >= WHALE_THRESHOLD
]

whale_counts = (
    whale_likes
    .groupby('post_id')
    .size()
    .reset_index(name='whale_count')
)

# ---- Unique liker count (useful for normalization) ----
unique_likers = (
    early_likes_with_network
    .groupby('post_id')['user_id']
    .nunique()
    .reset_index(name='unique_early_likers')
)

# ---- Combine network features ----
network_features = network_features.merge(whale_counts, on='post_id', how='left')
network_features = network_features.merge(unique_likers, on='post_id', how='left')
network_features['whale_count'] = network_features['whale_count'].fillna(0).astype(int)

# Reach concentration: is early reach dominated by one account?
# High = one whale drove it, Low = distributed organic interest
network_features['reach_concentration'] = (
    network_features['max_liker_followers'] / 
    (network_features['sum_liker_reach'] + 1)
)

# Whale ratio: what fraction of early likers are large accounts?
network_features['whale_ratio'] = (
    network_features['whale_count'] / 
    (network_features['unique_early_likers'] + 1)
)

# Print stats
print("\n  Network feature statistics:")
for col in ['max_liker_followers', 'mean_liker_followers', 'sum_liker_reach',
            'whale_count', 'reach_concentration', 'whale_ratio']:
    stats = network_features[col]
    print(f"    {col:25s}  mean={stats.mean():>10.2f}  "
          f"median={stats.median():>8.1f}  "
          f"max={stats.max():>12.0f}")

# Free memory
del early_likes_with_network, whale_likes, early_likes


# =============================================================================
# STEP 6: MERGE ALL EARLY LIKE FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Assembling early like feature set")
print("=" * 70)

# Start with ALL post_ids (so posts with no early likes get zeros)
all_post_ids = posts[['post_id']].copy()

# Merge velocity features
early_features = all_post_ids.merge(velocity_features, on='post_id', how='left')

# Merge network features
early_features = early_features.merge(network_features, on='post_id', how='left')

# ---- Fill zeros for posts with no early likes ----
fill_zero_cols = [
    # Velocity
    'likes_5m', 'likes_15m', 'likes_30m',
    'likes_15_to_30m', 'likes_5_to_15m',
    'like_acceleration',
    # Network
    'max_liker_followers', 'mean_liker_followers', 'median_liker_followers',
    'sum_liker_reach', 'whale_count', 'unique_early_likers',
]
for col in fill_zero_cols:
    if col in early_features.columns:
        early_features[col] = early_features[col].fillna(0)

# Ratios default to 0 for posts with no early likes
fill_zero_ratios = [
    'like_velocity_ratio', 'early_burst_ratio',
    'reach_concentration', 'whale_ratio',
]
for col in fill_zero_ratios:
    if col in early_features.columns:
        early_features[col] = early_features[col].fillna(0)

# Convert count columns to int
int_cols = ['likes_5m', 'likes_15m', 'likes_30m', 'likes_15_to_30m',
            'likes_5_to_15m', 'like_acceleration', 'whale_count', 
            'unique_early_likers']
for col in int_cols:
    if col in early_features.columns:
        early_features[col] = early_features[col].astype(int)

# ---- Final column list ----
feature_cols = [
    'post_id',
    # Layer 1: Velocity & Acceleration
    'likes_5m', 'likes_15m', 'likes_30m',
    'likes_15_to_30m', 'likes_5_to_15m',
    'like_acceleration', 'like_velocity_ratio', 'early_burst_ratio',
    # Layer 2: Network Quality
    'unique_early_likers',
    'max_liker_followers', 'mean_liker_followers', 'median_liker_followers',
    'sum_liker_reach', 'whale_count', 'whale_ratio', 'reach_concentration',
]

early_features = early_features[feature_cols]

# =============================================================================
# STEP 7: VALIDATE AND SAVE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Validation & output")
print("=" * 70)

# Sanity checks
assert len(early_features) == len(posts), \
    f"Row count mismatch: {len(early_features)} features vs {len(posts)} posts"

assert early_features['post_id'].nunique() == len(early_features), \
    "Duplicate post_ids in output!"

assert early_features.isnull().sum().sum() == 0, \
    f"Unexpected nulls:\n{early_features.isnull().sum()[early_features.isnull().sum() > 0]}"

print("  All validation checks passed")

# Quick correlation with target (preview of predictive power)
with_target = early_features.merge(
    posts[['post_id', 'is_viral']], on='post_id'
)

print("\n  Correlation with is_viral (top features):")
correlations = (
    with_target
    .drop(columns=['post_id'])
    .corr()['is_viral']
    .drop('is_viral')
    .abs()
    .sort_values(ascending=False)
)
for feat, corr in correlations.head(10).items():
    print(f"    {feat:30s}  r = {corr:.4f}")

# Distribution of posts by early like activity
print(f"\n  Posts with zero early likes:   "
      f"{(early_features['likes_30m'] == 0).sum():,} "
      f"({(early_features['likes_30m'] == 0).sum()/len(early_features)*100:.1f}%)")
print(f"  Posts with >=1 early like:      "
      f"{(early_features['likes_30m'] > 0).sum():,} "
      f"({(early_features['likes_30m'] > 0).sum()/len(early_features)*100:.1f}%)")
print(f"  Posts with >=10 early likes:    "
      f"{(early_features['likes_30m'] >= 10).sum():,} "
      f"({(early_features['likes_30m'] >= 10).sum()/len(early_features)*100:.1f}%)")

# Compare viral vs non-viral means
print("\n  Mean feature values: Viral vs Non-Viral")
print(f"  {'Feature':30s} {'Viral':>10s} {'Non-Viral':>10s} {'Ratio':>8s}")
print(f"  {'-'*62}")
for col in ['likes_30m', 'like_acceleration', 'like_velocity_ratio',
            'sum_liker_reach', 'whale_count', 'unique_early_likers']:
    viral_mean = with_target[with_target['is_viral'] == 1][col].mean()
    nonviral_mean = with_target[with_target['is_viral'] == 0][col].mean()
    ratio = viral_mean / (nonviral_mean + 0.001)
    print(f"  {col:30s} {viral_mean:>10.2f} {nonviral_mean:>10.2f} {ratio:>7.1f}x")

# Save
output_path = os.path.join(FEATURES_DIR, "early_like_features.parquet")
early_features.to_parquet(output_path, index=False)

print(f"\n  Saved: {output_path}")
print(f"     {len(early_features):,} posts x {len(feature_cols) - 1} features")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("EARLY LIKE FEATURES COMPLETE - SUMMARY")
print("=" * 70)
print(f"""
  Features computed: {len(feature_cols) - 1}
  
    Layer 1 - Velocity & Acceleration:
    likes_5m, likes_15m, likes_30m
    likes_15_to_30m, likes_5_to_15m
    like_acceleration, like_velocity_ratio, early_burst_ratio
  
    Layer 2 - Network Quality:
    unique_early_likers
    max_liker_followers, mean_liker_followers, median_liker_followers
    sum_liker_reach, whale_count, whale_ratio, reach_concentration
  
  Output: {output_path}
  
  Next step: Run step_03_repost_reply_proxy.py
""")