"""
step_01_foundation.py

Builds the clean base tables needed by all subsequent feature modules:
  - base_posts.parquet      (167K deduplicated posts with target variable)
  - base_likes.parquet      (cleaned, date-filtered likes)
  - follower_counts.parquet (per-user follower/following counts)

Outputs are saved to DATA_DIR/features/
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "../data/processed"                    # where parquets live
OUTPUT_DIR = os.path.join(DATA_DIR, "features") # where to save outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Virality threshold: a post is "viral" if repost_count >= this
# Compute this dynamically (top N percentile) but also allow a fixed value
VIRALITY_PERCENTILE = 90  # top 10% = "viral"

# =============================================================================
# HELPER: Parse YYYYMMDDHHmm integer dates to datetime
# =============================================================================

def parse_int_date(series, name="date"):
    """
    Converts int64 dates in YYYYMMDDHHmm format to proper datetime.
    
    Handles two formats found in this dataset:
      - 12-digit: YYYYMMDDHHmm  (e.g., 202309192352)
            - 8-digit:  YYYYMMDD      (e.g., 20230826) - found in quotes.parquet
    
    Returns a datetime Series. Unparseable values become NaT.
    """
    str_series = series.astype(str)
    
    # 12-digit: YYYYMMDDHHmm
    mask_12 = str_series.str.len() == 12
    # 8-digit: YYYYMMDD
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
        print(f"  WARNING: {n_failed:,} values in '{name}' could not be parsed")
    
    return result


# =============================================================================
# STEP 1: BUILD CLEAN POSTS TABLE
# =============================================================================

print("=" * 70)
print("STEP 1: Building clean posts table")
print("=" * 70)

# 1a. Load all 11 feed files
feed_files = sorted(glob.glob(os.path.join(DATA_DIR, "feed_*.parquet")))
feed_files = [f for f in feed_files if "likes" not in f and "bookmarks" not in f]

all_posts = []
for f in feed_files:
    fname = os.path.basename(f)
    feed_name = fname.replace("feed_", "").replace(".parquet", "")
    df = pd.read_parquet(f)
    df['feed'] = feed_name
    all_posts.append(df)
    print(f"  Loaded {fname:45s} -> {len(df):>8,} posts")

posts = pd.concat(all_posts, ignore_index=True)
print(f"\n  Combined total: {len(posts):,} rows")

# 1b. Deduplicate
# Strategy: for posts appearing in multiple feeds, keep the first and store
# all feed names as a list (useful feature later)
feed_map = (
    posts.groupby('post_id')['feed']
    .agg(lambda x: list(sorted(set(x))))
    .reset_index()
    .rename(columns={'feed': 'feeds'})
)
feed_map['feed_count'] = feed_map['feeds'].apply(len)

# Keep first occurrence of each post
posts_clean = posts.drop_duplicates(subset='post_id', keep='first').copy()
posts_clean = posts_clean.drop(columns=['feed'])  # replace with the full feed list
posts_clean = posts_clean.merge(feed_map, on='post_id', how='left')

print(f"  After deduplication: {len(posts_clean):,} unique posts")
print(f"  Posts in multiple feeds: {(feed_map['feed_count'] > 1).sum():,}")

# 1c. Parse dates
posts_clean['created_at'] = parse_int_date(posts_clean['date'], name='post_date')
n_null_dates = posts_clean['created_at'].isna().sum()
if n_null_dates > 0:
    print(f"  WARNING: Dropping {n_null_dates:,} posts with unparseable dates")
    posts_clean = posts_clean.dropna(subset=['created_at'])

print(f"  Date range: {posts_clean['created_at'].min()} -> {posts_clean['created_at'].max()}")


# =============================================================================
# STEP 2: DEFINE TARGET VARIABLE (Repost-Based)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Defining target variable")
print("=" * 70)

# Analyze repost distribution
repost_stats = posts_clean['repost_count'].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
)
print("\n  Repost count distribution:")
for idx, val in repost_stats.items():
    print(f"    {idx:>6s}: {val:>10.1f}")

# Calculate the threshold
threshold = posts_clean['repost_count'].quantile(VIRALITY_PERCENTILE / 100)
# Round up to nearest integer (need at least this many reposts)
threshold = int(np.ceil(threshold))

print(f"\n  Virality threshold (top {100 - VIRALITY_PERCENTILE}%): >= {threshold} reposts")

# Create binary target
posts_clean['is_viral'] = (posts_clean['repost_count'] >= threshold).astype(int)

viral_count = posts_clean['is_viral'].sum()
total_count = len(posts_clean)
print(f"  Viral posts:     {viral_count:,} ({viral_count/total_count*100:.1f}%)")
print(f"  Non-viral posts: {total_count - viral_count:,} ({(total_count - viral_count)/total_count*100:.1f}%)")

# Also store the raw count for potential regression or alternative thresholds
posts_clean['target_repost_count'] = posts_clean['repost_count']


# =============================================================================
# STEP 3: BUILD CLEAN LIKES TABLE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Building clean likes table")
print("=" * 70)

like_files = sorted(glob.glob(os.path.join(DATA_DIR, "feed_likes_*.parquet")))

all_likes = []
for f in like_files:
    fname = os.path.basename(f)
    df = pd.read_parquet(f)
    all_likes.append(df)
    print(f"  Loaded {fname:45s} -> {len(df):>10,} likes")

likes = pd.concat(all_likes, ignore_index=True)
print(f"\n  Combined total: {len(likes):,} likes")

# 3a. Parse dates
likes['liked_at'] = parse_int_date(likes['date'], name='like_date')

# 3b. Remove likes with impossible dates
# Validation showed max like date is 203010180531 - clearly erroneous
# Dataset posts span 2023-02 to 2024-03, so cap likes at 2024-04-01
DATE_CUTOFF = pd.Timestamp("2024-04-01")
bad_dates = (likes['liked_at'] > DATE_CUTOFF) | (likes['liked_at'].isna())
n_bad = bad_dates.sum()
if n_bad > 0:
    print(f"  WARNING: Removing {n_bad:,} likes with dates after {DATE_CUTOFF.date()} or unparseable")
    likes = likes[~bad_dates].copy()

# 3c. Keep only likes for posts in clean posts table
known_posts = set(posts_clean['post_id'].unique())
likes = likes[likes['post_id'].isin(known_posts)].copy()
print(f"  Likes matching known posts: {len(likes):,}")

# 3d. Deduplicate: one like per user per post
before = len(likes)
likes = likes.drop_duplicates(subset=['post_id', 'like_id'], keep='first')
after = len(likes)
if before != after:
    print(f"  Dedup removed {before - after:,} duplicate likes")

print(f"  Final clean likes: {len(likes):,}")
print(f"  Date range: {likes['liked_at'].min()} -> {likes['liked_at'].max()}")


# =============================================================================
# STEP 4: BUILD FOLLOWER COUNT LOOKUP
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Building follower count lookup")
print("=" * 70)

followers = pd.read_parquet(os.path.join(DATA_DIR, "followers.parquet"))
print(f"  Follower edges loaded: {len(followers):,}")

# follower_id follows followed_id
# -> follower_count  = how many people follow you (inbound)
# -> following_count = how many people you follow (outbound)

follower_counts = (
    followers
    .groupby('followed_id')
    .size()
    .reset_index(name='follower_count')
    .rename(columns={'followed_id': 'user_id'})
)

following_counts = (
    followers
    .groupby('follower_id')
    .size()
    .reset_index(name='following_count')
    .rename(columns={'follower_id': 'user_id'})
)

user_counts = follower_counts.merge(following_counts, on='user_id', how='outer')
user_counts = user_counts.fillna(0)
user_counts['follower_count'] = user_counts['follower_count'].astype(int)
user_counts['following_count'] = user_counts['following_count'].astype(int)

print(f"  Users with follower data: {len(user_counts):,}")
print(f"  Follower count stats:")
for stat, val in user_counts['follower_count'].describe().items():
    print(f"    {stat:>6s}: {val:>12.1f}")

# Free raw edge list from memory
del followers


# =============================================================================
# STEP 5: ATTACH AUTHOR INFO TO POSTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Attaching author follower counts to posts")
print("=" * 70)

posts_clean = posts_clean.merge(
    user_counts.rename(columns={
        'user_id': 'user_id',
        'follower_count': 'author_follower_count',
        'following_count': 'author_following_count'
    }),
    on='user_id',
    how='left'
)

posts_clean['author_follower_count'] = posts_clean['author_follower_count'].fillna(0).astype(int)
posts_clean['author_following_count'] = posts_clean['author_following_count'].fillna(0).astype(int)

coverage = (posts_clean['author_follower_count'] > 0).sum()
print(f"  Posts with author follower data: {coverage:,} / {len(posts_clean):,} "
      f"({coverage/len(posts_clean)*100:.1f}%)")


# =============================================================================
# STEP 6: COMPUTE AUTHOR BASELINES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Computing author baseline statistics")
print("=" * 70)

author_baselines = (
    posts_clean
    .groupby('user_id')
    .agg(
        author_post_count=('post_id', 'count'),
        author_median_likes=('like_count', 'median'),
        author_median_reposts=('repost_count', 'median'),
        author_mean_likes=('like_count', 'mean'),
        author_mean_reposts=('repost_count', 'mean'),
    )
    .reset_index()
)

posts_clean = posts_clean.merge(author_baselines, on='user_id', how='left')

print(f"  Computed baselines for {len(author_baselines):,} authors")
print(f"  Median author_median_reposts: {author_baselines['author_median_reposts'].median():.1f}")
print(f"  Median author_median_likes:   {author_baselines['author_median_likes'].median():.1f}")


# =============================================================================
# STEP 7: SAVE OUTPUTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Saving outputs")
print("=" * 70)

# Select columns to save for base_posts
post_cols = [
    'post_id', 'user_id', 'created_at', 'date',
    'text', 'langs', 'instance',
    'like_count', 'reply_count', 'repost_count',
    'reply_to', 'replied_author', 'thread_root', 'thread_root_author',
    'quotes', 'quoted_author', 'labels',
    'feeds', 'feed_count',
    'is_viral', 'target_repost_count',
    'author_follower_count', 'author_following_count',
    'author_post_count', 'author_median_likes', 'author_median_reposts',
    'author_mean_likes', 'author_mean_reposts',
]
# Only keep columns that actually exist (some might vary)
post_cols = [c for c in post_cols if c in posts_clean.columns]

base_posts_path = os.path.join(OUTPUT_DIR, "base_posts.parquet")
posts_clean[post_cols].to_parquet(base_posts_path, index=False)
print(f"  Saved: {base_posts_path}")
print(f"     {len(posts_clean):,} posts x {len(post_cols)} columns")

# Save likes
like_cols = ['like_id', 'user_id', 'post_id', 'liked_at']
like_cols = [c for c in like_cols if c in likes.columns]
base_likes_path = os.path.join(OUTPUT_DIR, "base_likes.parquet")
likes[like_cols].to_parquet(base_likes_path, index=False)
print(f"  Saved: {base_likes_path}")
print(f"     {len(likes):,} likes x {len(like_cols)} columns")

# Save follower counts (needed for network features in later steps)
follower_path = os.path.join(OUTPUT_DIR, "follower_counts.parquet")
user_counts.to_parquet(follower_path, index=False)
print(f"  Saved: {follower_path}")
print(f"     {len(user_counts):,} users x {len(user_counts.columns)} columns")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("FOUNDATION COMPLETE - SUMMARY")
print("=" * 70)
print(f"""
  Posts:           {len(posts_clean):>10,} (deduplicated)
  Likes:           {len(likes):>10,} (cleaned, date-filtered)
  Users w/ counts: {len(user_counts):>10,}
  
  Target variable: is_viral (repost_count >= {threshold})
    Viral:         {posts_clean['is_viral'].sum():>10,} ({posts_clean['is_viral'].mean()*100:.1f}%)
    Non-viral:     {(~posts_clean['is_viral'].astype(bool)).sum():>10,} ({(1-posts_clean['is_viral'].mean())*100:.1f}%)
  
  Output files:
    {base_posts_path}
    {base_likes_path}
    {follower_path}
  
  Next step: Run step_02_early_like_features.py
""")