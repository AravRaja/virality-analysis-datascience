"""
step_03_repost_reply_proxy

Computes author-level proxy features for early repost and reply activity.

Since reposts.parquet and replies.parquet lack post_id, we approximate by
attributing all reposts/replies directed at an author within 30 minutes of
their post's creation to that post. An ambiguity flag marks posts where
the author published again within the window.

  Layer 5 — Repost Proxy Features:
    author_reposts_5m, author_reposts_15m, author_reposts_30m
    repost_acceleration, repost_velocity_ratio

  Layer 5 — Reply Proxy Features:
    author_replies_5m, author_replies_15m, author_replies_30m
    reply_acceleration, reply_velocity_ratio

  Ambiguity & Ratios:
    proxy_ambiguous, minutes_to_next_post
    early_amplification_ratio, early_discussion_ratio
    repost_to_like_ratio

Inputs:  features/base_posts.parquet        (from step_01)
         data/processed/reposts.parquet     (raw)
         data/processed/replies.parquet     (raw)

Outputs: features/repost_reply_proxy_features.parquet
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "../data/processed"
FEATURES_DIR = "../data/processed/features"

WINDOW_5M  = 5   # minutes
WINDOW_15M = 15
WINDOW_30M = 30

# =============================================================================
# HELPER: Parse YYYYMMDDHHmm integer dates to datetime
# =============================================================================

def parse_int_date(series, name="date"):
    """Converts int64 dates in YYYYMMDDHHmm format to datetime."""
    str_series = series.astype(str)
    mask_12 = str_series.str.len() == 12
    mask_8 = str_series.str.len() == 8

    result = pd.Series(pd.NaT, index=series.index)

    if mask_12.any():
        result[mask_12] = pd.to_datetime(
            str_series[mask_12], format="%Y%m%d%H%M", errors='coerce'
        )
    if mask_8.any():
        result[mask_8] = pd.to_datetime(
            str_series[mask_8], format="%Y%m%d", errors='coerce'
        )

    n_failed = result.isna().sum() - series.isna().sum()
    if n_failed > 0:
        print(f"   {n_failed:,} values in '{name}' could not be parsed")

    return result


# =============================================================================
# HELPER: Compute proxy features for a given event type
# =============================================================================

def compute_proxy_features(events_df, posts_df, author_col, event_name):
    """
    For each post, finds events directed at the post's author within 30 minutes
    of post creation, and computes velocity/acceleration features.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: [author_col, 'event_time']
    posts_df : pd.DataFrame
        Must have columns: ['post_id', 'user_id', 'created_at']
    author_col : str
        Column name in events_df that contains the target author ID
    event_name : str
        Label for the event type ('repost' or 'reply')

    Returns
    -------
    pd.DataFrame with post_id and proxy features
    """
    print(f"\n  Processing {event_name}s...")

    # Rename author column for consistent merging
    events = events_df.rename(columns={author_col: 'target_author'}).copy()

    # Get unique (post_id, user_id, created_at) for matching
    post_info = posts_df[['post_id', 'user_id', 'created_at']].copy()

    # STRATEGY: For each author, find their posts and the events directed at them.
    # Then compute time differences.
    #
    # This is more memory-efficient than a full cross join:
    # 1. Merge events to posts via author ID
    # 2. Compute minutes_after
    # 3. Filter to [0, 30] minute window

    merged = events.merge(
        post_info,
        left_on='target_author',
        right_on='user_id',
        how='inner'
    )

    print(f"    Events matched to author's posts: {len(merged):,}")

    # Compute time difference
    merged['minutes_after'] = (
        (merged['event_time'] - merged['created_at'])
        .dt.total_seconds() / 60.0
    )

    # Filter to early window (0 to 30 minutes)
    early_mask = (
        (merged['minutes_after'] >= 0) &
        (merged['minutes_after'] <= WINDOW_30M)
    )
    early = merged[early_mask].copy()

    print(f"    Events within 0–30m window: {len(early):,}")

    # Free memory
    del merged

    # ---- Aggregate by post_id across time windows ----
    counts_30m = (
        early
        .groupby('post_id')
        .size()
        .reset_index(name=f'author_{event_name}s_30m')
    )

    counts_15m = (
        early[early['minutes_after'] <= WINDOW_15M]
        .groupby('post_id')
        .size()
        .reset_index(name=f'author_{event_name}s_15m')
    )

    counts_5m = (
        early[early['minutes_after'] <= WINDOW_5M]
        .groupby('post_id')
        .size()
        .reset_index(name=f'author_{event_name}s_5m')
    )

    # Unique engagers (distinct users who reposted/replied)
    unique_engagers = (
        early
        .groupby('post_id')['user_id_x']  # user_id_x = the engager (from events)
        .nunique()
        .reset_index(name=f'unique_early_{event_name}_users')
    )

    # ---- Combine ----
    features = counts_30m.copy()
    features = features.merge(counts_15m, on='post_id', how='left')
    features = features.merge(counts_5m, on='post_id', how='left')
    features = features.merge(unique_engagers, on='post_id', how='left')

    # Fill NaN
    for col in features.columns:
        if col != 'post_id':
            features[col] = features[col].fillna(0).astype(int)

    # ---- Derived features ----
    col_30m = f'author_{event_name}s_30m'
    col_15m = f'author_{event_name}s_15m'
    col_5m = f'author_{event_name}s_5m'

    # Second-half volume
    features[f'{event_name}_15_to_30m'] = features[col_30m] - features[col_15m]

    # Acceleration
    features[f'{event_name}_acceleration'] = (
        features[f'{event_name}_15_to_30m'] - features[col_15m]
    )

    # Velocity ratio
    features[f'{event_name}_velocity_ratio'] = (
        features[f'{event_name}_15_to_30m'] / (features[col_15m] + 1)
    )

    posts_with_events = len(features)
    print(f"    Posts with ≥1 early {event_name}: {posts_with_events:,}")

    return features


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

posts = pd.read_parquet(os.path.join(FEATURES_DIR, "base_posts.parquet"))
posts['created_at'] = pd.to_datetime(posts['created_at'])

reposts = pd.read_parquet(os.path.join(DATA_DIR, "reposts.parquet"))
replies = pd.read_parquet(os.path.join(DATA_DIR, "replies.parquet"))

print(f"  Posts loaded:   {len(posts):>12,}")
print(f"  Reposts loaded: {len(reposts):>12,}")
print(f"  Replies loaded: {len(replies):>12,}")

# Parse event timestamps
print("\n  Parsing timestamps...")
reposts['event_time'] = parse_int_date(reposts['date'], name='repost_date')
replies['event_time'] = parse_int_date(replies['date'], name='reply_date')

# Drop rows with unparseable dates
reposts = reposts.dropna(subset=['event_time'])
replies = replies.dropna(subset=['event_time'])

print(f"  Reposts with valid dates: {len(reposts):,}")
print(f"  Replies with valid dates: {len(replies):,}")


# =============================================================================
# STEP 2: COMPUTE POSTING AMBIGUITY FLAG
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Computing posting ambiguity flags")
print("=" * 70)

# For each post, find the time until the SAME author's next post.
# If the author posted again within 30 minutes, we flag this post as
# ambiguous because we can't reliably attribute reposts/replies to it.

posts_sorted = posts[['post_id', 'user_id', 'created_at']].sort_values(
    ['user_id', 'created_at']
).copy()

# Time to next post by the same author
posts_sorted['next_post_time'] = (
    posts_sorted
    .groupby('user_id')['created_at']
    .shift(-1)
)

posts_sorted['minutes_to_next_post'] = (
    (posts_sorted['next_post_time'] - posts_sorted['created_at'])
    .dt.total_seconds() / 60.0
)

# Flag: is this post's 30-minute window "contaminated" by another post?
posts_sorted['proxy_ambiguous'] = (
    posts_sorted['minutes_to_next_post'] <= WINDOW_30M
).astype(int)

# For last posts by each author, minutes_to_next_post is NaN → not ambiguous
posts_sorted['proxy_ambiguous'] = posts_sorted['proxy_ambiguous'].fillna(0).astype(int)
posts_sorted['minutes_to_next_post'] = posts_sorted['minutes_to_next_post'].fillna(-1)

ambiguity_info = posts_sorted[['post_id', 'minutes_to_next_post', 'proxy_ambiguous']]

ambiguous_count = ambiguity_info['proxy_ambiguous'].sum()
total_count = len(ambiguity_info)
print(f"  Posts with ambiguous proxy window: {ambiguous_count:,} / {total_count:,} "
      f"({ambiguous_count/total_count*100:.1f}%)")
print(f"  Posts with clean proxy window:     {total_count - ambiguous_count:,} "
      f"({(total_count - ambiguous_count)/total_count*100:.1f}%)")

# Distribution of time to next post (for ambiguous ones)
ambiguous_posts = ambiguity_info[ambiguity_info['proxy_ambiguous'] == 1]
if len(ambiguous_posts) > 0:
    print(f"\n  Minutes to next post (ambiguous posts only):")
    print(f"    mean:   {ambiguous_posts['minutes_to_next_post'].mean():.1f}")
    print(f"    median: {ambiguous_posts['minutes_to_next_post'].median():.1f}")
    print(f"    min:    {ambiguous_posts['minutes_to_next_post'].min():.1f}")

del posts_sorted


# =============================================================================
# STEP 3: COMPUTE REPOST PROXY FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Computing repost proxy features")
print("=" * 70)

repost_features = compute_proxy_features(
    events_df=reposts,
    posts_df=posts,
    author_col='reposted_author',
    event_name='repost'
)

del reposts  # free memory


# =============================================================================
# STEP 4: COMPUTE REPLY PROXY FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Computing reply proxy features")
print("=" * 70)

reply_features = compute_proxy_features(
    events_df=replies,
    posts_df=posts,
    author_col='replied_author',
    event_name='reply'
)

del replies  # free memory


# =============================================================================
# STEP 5: LOAD EARLY LIKE FEATURES FOR CROSS-SIGNAL RATIOS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Computing cross-signal ratios")
print("=" * 70)

early_likes = pd.read_parquet(
    os.path.join(FEATURES_DIR, "early_like_features.parquet")
)

# We only need the like counts for ratio computation
like_counts = early_likes[['post_id', 'likes_30m', 'likes_15m']].copy()


# =============================================================================
# STEP 6: ASSEMBLE ALL PROXY FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Assembling proxy feature set")
print("=" * 70)

# Start with all post IDs
all_post_ids = posts[['post_id']].copy()

# Merge repost features
proxy_features = all_post_ids.merge(repost_features, on='post_id', how='left')

# Merge reply features
proxy_features = proxy_features.merge(reply_features, on='post_id', how='left')

# Merge ambiguity flags
proxy_features = proxy_features.merge(ambiguity_info, on='post_id', how='left')

# Merge like counts for ratio computation
proxy_features = proxy_features.merge(like_counts, on='post_id', how='left')

# ---- Fill zeros for posts with no early reposts/replies ----
fill_zero_cols = [
    # Repost proxy
    'author_reposts_30m', 'author_reposts_15m', 'author_reposts_5m',
    'repost_15_to_30m', 'repost_acceleration',
    'unique_early_repost_users',
    # Reply proxy
    'author_replies_30m', 'author_replies_15m', 'author_replies_5m',
    'reply_15_to_30m', 'reply_acceleration',
    'unique_early_reply_users',
    # Likes (should already be filled but just in case)
    'likes_30m', 'likes_15m',
]
for col in fill_zero_cols:
    if col in proxy_features.columns:
        proxy_features[col] = proxy_features[col].fillna(0).astype(int)

fill_zero_ratios = ['repost_velocity_ratio', 'reply_velocity_ratio']
for col in fill_zero_ratios:
    if col in proxy_features.columns:
        proxy_features[col] = proxy_features[col].fillna(0.0)

# ---- Compute cross-signal ratios ----

# Amplification ratio: how much "spread" vs "attention"?
# High = content is being actively propagated, not just passively liked
proxy_features['early_amplification_ratio'] = (
    proxy_features['author_reposts_30m'] /
    (proxy_features['likes_30m'] + 1)
)

# Discussion ratio: how much conversation vs passive engagement?
proxy_features['early_discussion_ratio'] = (
    proxy_features['author_replies_30m'] /
    (proxy_features['likes_30m'] + 1)
)

# Total early engagement (combined signal)
proxy_features['total_early_engagement'] = (
    proxy_features['likes_30m'] +
    proxy_features['author_reposts_30m'] +
    proxy_features['author_replies_30m']
)

# Engagement composition: what fraction of total engagement is reposts?
proxy_features['repost_share_of_engagement'] = (
    proxy_features['author_reposts_30m'] /
    (proxy_features['total_early_engagement'] + 1)
)

# Reply share
proxy_features['reply_share_of_engagement'] = (
    proxy_features['author_replies_30m'] /
    (proxy_features['total_early_engagement'] + 1)
)

# Drop the like columns used only for ratios (they live in early_like_features)
proxy_features = proxy_features.drop(columns=['likes_30m', 'likes_15m'])

# ---- Final column list ----
feature_cols = [
    'post_id',
    # Ambiguity
    'proxy_ambiguous', 'minutes_to_next_post',
    # Repost proxy
    'author_reposts_5m', 'author_reposts_15m', 'author_reposts_30m',
    'repost_15_to_30m', 'repost_acceleration', 'repost_velocity_ratio',
    'unique_early_repost_users',
    # Reply proxy
    'author_replies_5m', 'author_replies_15m', 'author_replies_30m',
    'reply_15_to_30m', 'reply_acceleration', 'reply_velocity_ratio',
    'unique_early_reply_users',
    # Cross-signal ratios
    'early_amplification_ratio', 'early_discussion_ratio',
    'total_early_engagement',
    'repost_share_of_engagement', 'reply_share_of_engagement',
]

proxy_features = proxy_features[feature_cols]


# =============================================================================
# STEP 7: VALIDATE AND SAVE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Validation & output")
print("=" * 70)

# Sanity checks
assert len(proxy_features) == len(posts), \
    f"Row count mismatch: {len(proxy_features)} features vs {len(posts)} posts"

assert proxy_features['post_id'].nunique() == len(proxy_features), \
    "Duplicate post_ids in output!"

null_counts = proxy_features.isnull().sum()
if null_counts.sum() > 0:
    print(f"    Unexpected nulls:\n{null_counts[null_counts > 0]}")
else:
    print("   All validation checks passed")

# Correlation with target
with_target = proxy_features.merge(
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
    print(f"    {feat:35s}  r = {corr:.4f}")

# Compare viral vs non-viral means
print(f"\n  Mean feature values: Viral vs Non-Viral")
print(f"  {'Feature':35s} {'Viral':>10s} {'Non-Viral':>10s} {'Ratio':>8s}")
print(f"  {'-'*67}")
key_features = [
    'author_reposts_30m', 'repost_acceleration',
    'author_replies_30m', 'reply_acceleration',
    'early_amplification_ratio', 'early_discussion_ratio',
    'total_early_engagement', 'repost_share_of_engagement',
]
for col in key_features:
    viral_mean = with_target[with_target['is_viral'] == 1][col].mean()
    nonviral_mean = with_target[with_target['is_viral'] == 0][col].mean()
    ratio = viral_mean / (nonviral_mean + 0.001)
    print(f"  {col:35s} {viral_mean:>10.2f} {nonviral_mean:>10.2f} {ratio:>7.1f}x")

# Ambiguity analysis: does the proxy work differently for clean vs ambiguous?
print(f"\n  Proxy reliability analysis:")
clean = with_target[with_target['proxy_ambiguous'] == 0]
ambig = with_target[with_target['proxy_ambiguous'] == 1]
print(f"    {'Subset':20s} {'Count':>8s} {'Mean reposts_30m':>18s} {'Viral rate':>12s}")
print(f"    {'-'*62}")
print(f"    {'Clean window':20s} {len(clean):>8,} "
      f"{clean['author_reposts_30m'].mean():>18.2f} "
      f"{clean['is_viral'].mean()*100:>11.1f}%")
print(f"    {'Ambiguous window':20s} {len(ambig):>8,} "
      f"{ambig['author_reposts_30m'].mean():>18.2f} "
      f"{ambig['is_viral'].mean()*100:>11.1f}%")

# Save
output_path = os.path.join(FEATURES_DIR, "repost_reply_proxy_features.parquet")
proxy_features.to_parquet(output_path, index=False)

print(f"\n   Saved: {output_path}")
print(f"     {len(proxy_features):,} posts × {len(feature_cols) - 1} features")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("REPOST/REPLY PROXY FEATURES COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  Features computed: {len(feature_cols) - 1}

  Repost Proxy (7 features):
    author_reposts_5m, author_reposts_15m, author_reposts_30m
    repost_15_to_30m, repost_acceleration, repost_velocity_ratio
    unique_early_repost_users

  Reply Proxy (7 features):
    author_replies_5m, author_replies_15m, author_replies_30m
    reply_15_to_30m, reply_acceleration, reply_velocity_ratio
    unique_early_reply_users

  Cross-Signal Ratios (5 features):
    early_amplification_ratio, early_discussion_ratio
    total_early_engagement
    repost_share_of_engagement, reply_share_of_engagement

  Ambiguity Flags (2 features):
    proxy_ambiguous, minutes_to_next_post

  Output: {output_path}

  Next step: Run step_04_content_temporal_features.py
""")