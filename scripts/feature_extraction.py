"""
feature_extraction.py

Uses CascadeStore to build a flat training-ready feature table.
All window aggregations are vectorised — no row-by-row loops.

The entire pipeline is driven by a single `window_min` parameter.
Windows are automatically split into N equal fractions:
    window_min=5,  n_fractions=5  →  [1, 2, 3, 4, 5]
    window_min=30, n_fractions=5  →  [6, 12, 18, 24, 30]
    window_min=60, n_fractions=5  →  [12, 24, 36, 48, 60]

Inputs  (via CascadeStore):
    datasets/bluesky_cascade/root_posts.parquet
    datasets/bluesky_cascade/reposts.parquet
    datasets/bluesky_cascade/likes.parquet
    datasets/bluesky_cascade/replies.parquet
    datasets/bluesky_cascade/quotes.parquet

Output:
    datasets/bluesky_cascade/training_features.parquet
    datasets/bluesky_cascade/training_features.csv     (optional)

Run:
    python scripts/feature_extraction.py             # default 30m
    python scripts/feature_extraction.py --window 5  # 5m window
    python scripts/feature_extraction.py --window 60 # 60m window
"""

# make it a function so when you put the time in it gives the features for up to that specific time (do this within fraction of the time such as quarter number of windows), feature reduction.

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # prevent OMP crash on Windows/Anaconda

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datasets.bluesky_cascade.cascade_store import CascadeStore
from scripts.emotion_features import load_model, predict_emotions, HF_MODEL


# CONFIGURATION

CASCADE_DIR = ROOT / "datasets" / "bluesky_cascade"
OUTPUT_DIR  = CASCADE_DIR
SAVE_CSV    = True
WHALE_THRESHOLD = 1_000
N_FRACTIONS = 5

# HELPERS

# Generate window fractions based on the total window_min
def make_windows(window_min: int, n_fractions: int = N_FRACTIONS) -> list[int]:
    """
    Split window_min into n_fractions equal steps.
    """
    step    = window_min / n_fractions
    windows = [round(step * i) for i in range(1, n_fractions + 1)]
    windows[-1] = window_min   # guarantee exact endpoint (rounding safety)
    return windows

# Counts how many events of one type happen for each post within each time window.
def window_counts(events: pd.DataFrame,
                  posts: pd.DataFrame,
                  prefix: str,
                  windows: list[int]) -> pd.DataFrame:
    """
    Vectorised per-window event counts for all posts at once.
    Posts with zero events in a window get 0 (not NaN).
    """
    result = posts[["uri"]].set_index("uri")
 
    if events.empty:
        for w in windows:
            result[f"{prefix}_{w}m"] = 0
        return result.reset_index()
 
    for w in windows:
        sec    = w * 60
        mask   = events["time_delta_sec"].between(0, sec)
        counts = (
            events[mask]
            .groupby("root_post_uri")
            .size()
            .rename(f"{prefix}_{w}m")
        )
        result = result.join(counts, how="left")
        result[f"{prefix}_{w}m"] = result[f"{prefix}_{w}m"].fillna(0).astype(int)
 
    return result.reset_index()

# Computes the time from post creation to the first event of a given type
def time_to_first(events: pd.DataFrame,
                  posts: pd.DataFrame,
                  col_name: str,
                  sentinel: float = 9_999.0) -> pd.DataFrame:
    
    if events.empty:
        return posts[["uri"]].assign(**{col_name: sentinel})

    first = (
        events[events["time_delta_sec"] >= 0]
        .groupby("root_post_uri")["time_delta_sec"]
        .min()
        .rename(col_name)
        .reset_index()
        .rename(columns={"root_post_uri": "uri"})
    )
    return posts[["uri"]].merge(first, on="uri", how="left").fillna({col_name: sentinel})


# Derives velocity and acceleration features from the raw window counts
def velocity_features(counts_df: pd.DataFrame,
                      prefix: str,
                      windows: list[int]) -> pd.DataFrame:
    """
    Derive velocity and acceleration features dynamically from windows.
 
    Uses three key positions:
        w_burst = windows[0]           first fraction  (early burst)
        w_half  = windows[mid index]   midpoint        (velocity split)
        w_full  = windows[-1]          full window     (total)
 
    Features produced:
        {p}_{w_half}_to_{w_full}m  second-half volume
        {p}_acceleration           second half minus first half
        {p}_velocity_ratio         second half / (first half + 1)
        {p}_burst_ratio            first fraction / (full + 1)
        {p}_{w_full}m_log          log(1 + full count)
    """
    df      = counts_df.copy()
    p       = prefix
    n       = len(windows)
    w_burst = windows[0]
    w_half  = windows[n // 2]
    w_full  = windows[-1]
 
    c_burst = f"{p}_{w_burst}m"
    c_half  = f"{p}_{w_half}m"
    c_full  = f"{p}_{w_full}m"
 
    second_half_col         = f"{p}_{w_half}_to_{w_full}m"
    df[second_half_col]     = df[c_full] - df[c_half]
    df[f"{p}_acceleration"] = df[second_half_col] - df[c_half]
    df[f"{p}_velocity_ratio"] = df[second_half_col] / (df[c_half] + 1)
    df[f"{p}_burst_ratio"]  = df[c_burst] / (df[c_full] + 1)
    df[f"{p}_{w_full}m_log"]= np.log1p(df[c_full])
 
    return df

# Describes the quality of the people engaging, not just how many there are
def engager_network_features(events: pd.DataFrame,
                             posts: pd.DataFrame,
                             prefix: str,
                             did_col: str,
                             followers_col: str,
                             window_min: int,
                             whale_threshold: int = WHALE_THRESHOLD) -> pd.DataFrame:
    """Network quality of early engagers (follower data in event table)."""
    result = posts[["uri"]].set_index("uri")
    p      = prefix
 
    default_cols = [
        f"{p}_engager_count",
        f"{p}_engager_followers_max",    f"{p}_engager_followers_mean",
        f"{p}_engager_followers_median", f"{p}_engager_followers_sum",
        f"{p}_engager_followers_std",    f"{p}_engager_reach_log",
        f"{p}_whale_count",              f"{p}_whale_ratio",
        f"{p}_reach_concentration",
    ]
 
    if events.empty or followers_col not in events.columns:
        for col in default_cols:
            result[col] = 0.0
        return result.reset_index()
 
    windowed = events[events["time_delta_sec"].between(0, window_min * 60)].copy()
    if windowed.empty:
        for col in default_cols:
            result[col] = 0.0
        return result.reset_index()
 
    windowed = windowed.drop_duplicates(subset=["root_post_uri", did_col])
    windowed["_followers"] = pd.to_numeric(
        windowed[followers_col], errors="coerce"
    ).fillna(0)
 
    agg = (windowed
           .groupby("root_post_uri")["_followers"]
           .agg(["count", "max", "mean", "median", "sum", "std"])
           .rename(columns={
               "count":  f"{p}_engager_count",
               "max":    f"{p}_engager_followers_max",
               "mean":   f"{p}_engager_followers_mean",
               "median": f"{p}_engager_followers_median",
               "sum":    f"{p}_engager_followers_sum",
               "std":    f"{p}_engager_followers_std",
           }))
    agg[f"{p}_engager_followers_std"] = agg[f"{p}_engager_followers_std"].fillna(0)
    agg[f"{p}_engager_reach_log"]     = np.log1p(agg[f"{p}_engager_followers_sum"])
 
    whale_counts = (windowed[windowed["_followers"] >= whale_threshold]
                    .groupby("root_post_uri").size()
                    .rename(f"{p}_whale_count"))
    agg = agg.join(whale_counts, how="left")
    agg[f"{p}_whale_count"]          = agg[f"{p}_whale_count"].fillna(0).astype(int)
    agg[f"{p}_whale_ratio"]          = agg[f"{p}_whale_count"] / (agg[f"{p}_engager_count"] + 1)
    agg[f"{p}_reach_concentration"]  = (
        agg[f"{p}_engager_followers_max"] / (agg[f"{p}_engager_followers_sum"] + 1)
    )
 
    result = result.join(agg, how="left")
    for col in default_cols:
        result[col] = result.get(col, 0.0).fillna(0)
    return result.reset_index()


def build_liker_follower_lookup(reposts, replies, quotes) -> pd.DataFrame:
    """Build DID -> follower_count lookup from tables that have follower data."""
    chunks = []
    for df, did_col, fol_col in [
        (reposts, "reposter_did", "reposter_followers"),
        (replies, "replier_did",  "replier_followers"),
        (quotes,  "quoter_did",   "quoter_followers"),
    ]:
        if not df.empty and did_col in df.columns and fol_col in df.columns:
            chunks.append(
                df[[did_col, fol_col]].rename(
                    columns={did_col: "did", fol_col: "followers"}
                )
            )
    if not chunks:
        return pd.DataFrame(columns=["did", "followers"])
    lookup = pd.concat(chunks, ignore_index=True)
    lookup["followers"] = pd.to_numeric(lookup["followers"], errors="coerce").fillna(0)
    return lookup.groupby("did")["followers"].max().reset_index()


def like_network_features(likes: pd.DataFrame,
                          posts: pd.DataFrame,
                          follower_lookup: pd.DataFrame,
                          window_min: int,
                          whale_threshold: int = WHALE_THRESHOLD) -> pd.DataFrame:
    """Network features for likes via cross-table follower lookup."""
    result = posts[["uri"]].set_index("uri")
    p      = "like"
 
    default_cols = [
        f"{p}_engager_count",
        f"{p}_engager_followers_max",    f"{p}_engager_followers_mean",
        f"{p}_engager_followers_median", f"{p}_engager_followers_sum",
        f"{p}_engager_followers_std",    f"{p}_engager_reach_log",
        f"{p}_whale_count",              f"{p}_whale_ratio",
        f"{p}_reach_concentration",
    ]
 
    if likes.empty or follower_lookup.empty:
        for col in default_cols:
            result[col] = 0.0
        return result.reset_index()
 
    windowed = likes[likes["time_delta_sec"].between(0, window_min * 60)].copy()
    if windowed.empty:
        for col in default_cols:
            result[col] = 0.0
        return result.reset_index()
 
    windowed = windowed.drop_duplicates(subset=["root_post_uri", "liker_did"])
    windowed = windowed.merge(
        follower_lookup.rename(columns={"did": "liker_did", "followers": "_followers"}),
        on="liker_did", how="left"
    )
    windowed["_followers"] = windowed["_followers"].fillna(0)
 
    matched = (windowed["_followers"] > 0).sum()
    print(f"    Like follower coverage: {matched:,} / {len(windowed):,} "
          f"({matched / max(len(windowed), 1) * 100:.1f}%)")
 
    agg = (windowed
           .groupby("root_post_uri")["_followers"]
           .agg(["count", "max", "mean", "median", "sum", "std"])
           .rename(columns={
               "count":  f"{p}_engager_count",
               "max":    f"{p}_engager_followers_max",
               "mean":   f"{p}_engager_followers_mean",
               "median": f"{p}_engager_followers_median",
               "sum":    f"{p}_engager_followers_sum",
               "std":    f"{p}_engager_followers_std",
           }))
    agg[f"{p}_engager_followers_std"] = agg[f"{p}_engager_followers_std"].fillna(0)
    agg[f"{p}_engager_reach_log"]     = np.log1p(agg[f"{p}_engager_followers_sum"])
 
    whale_counts = (windowed[windowed["_followers"] >= whale_threshold]
                    .groupby("root_post_uri").size()
                    .rename(f"{p}_whale_count"))
    agg = agg.join(whale_counts, how="left")
    agg[f"{p}_whale_count"]         = agg[f"{p}_whale_count"].fillna(0).astype(int)
    agg[f"{p}_whale_ratio"]         = agg[f"{p}_whale_count"] / (agg[f"{p}_engager_count"] + 1)
    agg[f"{p}_reach_concentration"] = (
        agg[f"{p}_engager_followers_max"] / (agg[f"{p}_engager_followers_sum"] + 1)
    )
 
    result = result.join(agg, how="left")
    for col in default_cols:
        result[col] = result.get(col, 0.0).fillna(0)
    return result.reset_index()


# HELPER (Author Baseline) 

def author_history_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out author history features."""
    df     = posts.copy()
    result = posts[["uri"]].copy()
 
    print(f"  Unique authors: {df['did'].nunique():,}")
    df["_ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values(["did", "_ts"]).reset_index(drop=True)
 
    result["author_prior_post_count"]     = df.groupby("did").cumcount().values
    result["author_prior_post_count_log"] = np.log1p(result["author_prior_post_count"])
 
    for metric, raw_col in [
        ("likes",   "total_likes"),
        ("reposts", "total_reposts"),
        ("replies", "total_replies"),
        ("quotes",  "total_quotes"),
    ]:
        vals      = pd.to_numeric(df[raw_col], errors="coerce").fillna(0)
        cum_sum   = vals.groupby(df["did"]).cumsum() - vals
        cum_count = df.groupby("did").cumcount()
 
        result[f"author_hist_mean_{metric}"] = np.where(
            cum_count > 0, cum_sum / cum_count, 0.0
        )
        shifted_max = (vals.groupby(df["did"])
                       .apply(lambda s: s.shift(1).expanding().max())
                       .reset_index(level=0, drop=True)
                       .fillna(0))
        result[f"author_hist_max_{metric}"] = shifted_max.values
 
    return result

# PRINT HELPERS

def print_window_stats(name, df, prefix, windows):
    print(f"\n  {name.upper()}")
    for w in windows:
        col = f"{prefix}_{w}m"
        if col not in df.columns:
            continue
        n_active = (df[col] > 0).sum()
        pct      = n_active / len(df) * 100
        print(f"    {w:>3}m  |  posts with >=1: {n_active:>6,} ({pct:>5.1f}%)  "
              f"|  mean: {df[col].mean():>6.2f}  "
              f"|  p95: {df[col].quantile(0.95):>6.0f}  "
              f"|  max: {df[col].max():>6.0f}")


def print_velocity_stats(name, df, prefix, windows):
    w_full = windows[-1]
    w_half = windows[len(windows) // 2]
    print(f"\n  {name.upper()} velocity features:")
    vel_cols = [
        f"{prefix}_{w_half}_to_{w_full}m",
        f"{prefix}_acceleration",
        f"{prefix}_velocity_ratio",
        f"{prefix}_burst_ratio",
        f"{prefix}_{w_full}m_log",
    ]
    for col in vel_cols:
        if col not in df.columns:
            continue
        s = df[col]
        print(f"    {col:<40s}  mean={s.mean():>8.3f}  "
              f"median={s.median():>7.3f}  "
              f"p95={s.quantile(0.95):>8.3f}  "
              f"max={s.max():>10.3f}")


def print_ttf_stats(name, df, col, sentinel=9_999.0):
    has_event = df[df[col] < sentinel]
    no_event  = df[df[col] >= sentinel]
    print(f"  {name:<10s}  posts with event: {len(has_event):>6,}  "
          f"| no event (sentinel): {len(no_event):>6,}")
    if not has_event.empty:
        s = has_event[col]
        print(f"            median: {s.median():>8.0f}s  "
              f"| mean: {s.mean():>8.0f}s  "
              f"| p95: {s.quantile(0.95):>8.0f}s  "
              f"| min: {s.min():>6.0f}s")

# Network stats printer 
def print_network_stats(name, df, prefix):
    p      = prefix
    active = (df[f"{p}_engager_count"] > 0).sum()
    whale  = df[f"{p}_whale_count"].sum()
    print(f"\n  {name.upper()} network quality:"
          f"\n    posts with engagers:     {active:>6,} / {len(df):,}"
          f"\n    total whale engagements: {whale:>6,.0f}")
    for col in [f"{p}_engager_followers_max", f"{p}_engager_followers_mean",
                f"{p}_engager_followers_sum", f"{p}_reach_concentration"]:
        if col in df.columns:
            s = df[col]
            print(f"    {col:<45s}  mean={s.mean():>10.2f}  median={s.median():>8.2f}")

# MAIN

def main(window_min: int = 30):
    """
    Run full feature extraction for all events up to `window_min` minutes.
 
    Parameters
    ----------
    window_min : int
        Observation window in minutes. Automatically split into N_FRACTIONS
        equal steps:
            window_min=5  -> [1, 2, 3, 4, 5]
            window_min=30 -> [6, 12, 18, 24, 30]
            window_min=60 -> [12, 24, 36, 48, 60]
    """
    
    # Compute dynamic windows
    windows = make_windows(window_min, N_FRACTIONS)
    w_full = windows[-1]
    w_half = windows[len(windows) // 2]
    w_burst = windows[0]
    
    print("=" * 70)
    print(f"FEATURE EXTRACTION — window={window_min}m")
    print(f"  Windows : {windows}")
    print(f"  Burst   : {w_burst}m  |  Half : {w_half}m  |  Full : {w_full}m")
    print("=" * 70)

    # Load 
    print(f"\n  Loading CascadeStore from: {CASCADE_DIR}")
    store   = CascadeStore(CASCADE_DIR)
    summary = store.summary()
    print(f"\n  Dataset summary:")
    for k, v in summary.items():
        print(f"    {k:<20s}: {v:>10,}")
 
    posts   = store.posts.copy()
    reposts = store._reposts.copy()
    likes   = store._likes.copy()
    replies = store._replies.copy()
    quotes  = store._quotes.copy()
 
    for df in [reposts, likes, replies, quotes]:
        if "root_post_uri" not in df.columns:
            raise ValueError("Expected 'root_post_uri' column in event tables")
 
    print(f"\n  Posts: {len(posts):,}  "
          f"|  viral: {posts['is_viral'].sum():,}  "
          f"|  non-viral: {(~posts['is_viral']).sum():,}  "
          f"|  viral rate: {posts['is_viral'].mean()*100:.2f}%")


    # BLOCK 1: WINDOW COUNTS

    print("\n" + "=" * 70)
    print(f"BLOCK 1: Window counts {windows}")
    print("=" * 70)

    # Builds the raw early count features for each interaction type    
    repost_counts = window_counts(reposts, posts, "repost", windows)
    like_counts   = window_counts(likes,   posts, "like",   windows)
    reply_counts  = window_counts(replies, posts, "reply",  windows)
    quote_counts  = window_counts(quotes,  posts, "quote",  windows)
 
    print_window_stats("reposts", repost_counts, "repost", windows)
    print_window_stats("likes",   like_counts,   "like",   windows)
    print_window_stats("replies", reply_counts,  "reply",  windows)
    print_window_stats("quotes",  quote_counts,  "quote",  windows)


    # BLOCK 2: VELOCITY & ACCELERATION 

    print("\n" + "=" * 70)
    print("BLOCK 2: Velocity & acceleration features")
    print("=" * 70)

    repost_counts = velocity_features(repost_counts, "repost", windows)
    like_counts   = velocity_features(like_counts,   "like",   windows)
    reply_counts  = velocity_features(reply_counts,  "reply",  windows)
    quote_counts  = velocity_features(quote_counts,  "quote",  windows)
 
    print_velocity_stats("reposts", repost_counts, "repost", windows)
    print_velocity_stats("likes",   like_counts,   "like",   windows)
    print_velocity_stats("replies", reply_counts,  "reply",  windows)
    print_velocity_stats("quotes",  quote_counts,  "quote",  windows)


    # BLOCK 3: TIME-TO-FIRST-EVENT

    print("\n" + "=" * 70)
    print("BLOCK 3: Time-to-first-event (seconds from post creation)")
    print("=" * 70)

    ttf_repost = time_to_first(reposts, posts, "ttf_repost_sec")
    ttf_like   = time_to_first(likes,   posts, "ttf_like_sec")
    ttf_reply  = time_to_first(replies, posts, "ttf_reply_sec")
    ttf_quote  = time_to_first(quotes,  posts, "ttf_quote_sec")
 
    print_ttf_stats("repost", ttf_repost, "ttf_repost_sec")
    print_ttf_stats("like",   ttf_like,   "ttf_like_sec")
    print_ttf_stats("reply",  ttf_reply,  "ttf_reply_sec")
    print_ttf_stats("quote",  ttf_quote,  "ttf_quote_sec")

   
    # BLOCK 4: CONTENT & AUTHOR FEATURES (Layer 3 — expanded)

    print("\n" + "=" * 70)
    print("BLOCK 4: Content & author features")
    print("=" * 70)
 
    content = posts[["uri"]].copy()
    content["text_len"]  = posts["text"].fillna("").str.len()
    content["has_embed"] = posts.get(
        "has_embed", pd.Series(0, index=posts.index)
    ).fillna(0).astype(int)
 
    for col_name, src_col in [
        ("author_followers",   "author_followers"),
        ("author_follows",     "author_follows"),
        ("author_posts_count", "author_posts_count"),
    ]:
        content[col_name] = pd.to_numeric(
            posts.get(src_col, pd.Series(0, index=posts.index)),
            errors="coerce"
        ).fillna(0).astype(int)
 
    content["author_followers_log"]   = np.log1p(content["author_followers"])
    content["author_follows_log"]     = np.log1p(content["author_follows"])
    content["author_posts_count_log"] = np.log1p(content["author_posts_count"])
    content["author_is_whale"]        = (content["author_followers"] >= WHALE_THRESHOLD).astype(int)
    content["author_ff_ratio"]        = content["author_followers"] / (content["author_follows"] + 1)
 
    print(f"  text_len          mean={content['text_len'].mean():.0f}  | median={content['text_len'].median():.0f}")
    print(f"  has_embed         rate={content['has_embed'].mean()*100:.1f}%")
    print(f"  author_followers  mean={content['author_followers'].mean():.0f}  | max={content['author_followers'].max()}")
    print(f"  author_ff_ratio   mean={content['author_ff_ratio'].mean():.2f}  | median={content['author_ff_ratio'].median():.2f}")
    print(f"  author_is_whale   rate={content['author_is_whale'].mean()*100:.1f}%")
 
    print("\n  Computing leave-one-out author history...")
    author_hist = author_history_features(posts)
    


    # BLOCK 4B: ENGAGER NETWORK QUALITY (NEW — Layer 2)

    print("\n" + "=" * 70)
    print(f"BLOCK 4B: Engager network quality ({w_full}m window)")
    print("=" * 70)
 
    net_reposts = engager_network_features(
        reposts, posts, "repost",
        did_col="reposter_did", followers_col="reposter_followers",
        window_min=w_full
    )
    net_replies = engager_network_features(
        replies, posts, "reply",
        did_col="replier_did", followers_col="replier_followers",
        window_min=w_full
    )
    net_quotes = engager_network_features(
        quotes, posts, "quote",
        did_col="quoter_did", followers_col="quoter_followers",
        window_min=w_full
    )
 
    print("\n  Building liker follower lookup from other tables...")
    liker_lookup = build_liker_follower_lookup(reposts, replies, quotes)
    print(f"    Lookup size: {len(liker_lookup):,} unique DIDs")
    net_likes = like_network_features(likes, posts, liker_lookup, window_min=w_full)
 
    print_network_stats("reposts", net_reposts, "repost")
    print_network_stats("likes",   net_likes,   "like")
    print_network_stats("replies", net_replies,  "reply")
    print_network_stats("quotes",  net_quotes,   "quote")

 
    # BLOCK 5: COMBINED ENGAGEMENT FEATURES - dynamically built for each window

    print("\n" + "=" * 70)
    print(f"BLOCK 5: Combined engagement features ({w_full}m window)")
    print("=" * 70)
 
    combined = posts[["uri"]].copy()
 
    # Pull the full-window count for each event type
    for prefix, counts_df in [
        ("repost", repost_counts), ("like", like_counts),
        ("reply",  reply_counts),  ("quote", quote_counts),
    ]:
        full_col = f"{prefix}_{w_full}m"
        combined = combined.merge(counts_df[["uri", full_col]], on="uri", how="left")
 
    combined = combined.fillna(0)
 
    combined[f"total_engagement_{w_full}m"] = (
        combined[f"repost_{w_full}m"] + combined[f"like_{w_full}m"] +
        combined[f"reply_{w_full}m"]  + combined[f"quote_{w_full}m"]
    )
    combined[f"like_repost_ratio_{w_full}m"]  = (
        combined[f"like_{w_full}m"]  / (combined[f"repost_{w_full}m"] + 1)
    )
    combined[f"reply_repost_ratio_{w_full}m"] = (
        combined[f"reply_{w_full}m"] / (combined[f"repost_{w_full}m"] + 1)
    )
    combined[f"quote_repost_ratio_{w_full}m"] = (
        combined[f"quote_{w_full}m"] / (combined[f"repost_{w_full}m"] + 1)
    )
 
    # Midpoint totals (only if different from full)
    if w_half != w_full:
        for prefix, counts_df in [
            ("repost", repost_counts), ("like", like_counts),
            ("reply",  reply_counts),  ("quote", quote_counts),
        ]:
            half_col = f"{prefix}_{w_half}m"
            if half_col not in combined.columns:
                combined = combined.merge(
                    counts_df[["uri", half_col]], on="uri", how="left"
                ).fillna(0)
 
        combined[f"total_engagement_{w_half}m"] = (
            combined[f"repost_{w_half}m"] + combined[f"like_{w_half}m"] +
            combined[f"reply_{w_half}m"]  + combined[f"quote_{w_half}m"]
        )
        combined[f"like_repost_ratio_{w_half}m"] = (
            combined[f"like_{w_half}m"] / (combined[f"repost_{w_half}m"] + 1)
        )
 
    print(f"  total_engagement_{w_full}m  "
          f"mean={combined[f'total_engagement_{w_full}m'].mean():.2f}  "
          f"| max={combined[f'total_engagement_{w_full}m'].max():.0f}")
    print(f"  posts with any engagement: "
          f"{(combined[f'total_engagement_{w_full}m'] > 0).sum():,} / {len(combined):,}")
    print(f"  like_repost_ratio_{w_full}m   "
          f"mean={combined[f'like_repost_ratio_{w_full}m'].mean():.2f}")
    print(f"  reply_repost_ratio_{w_full}m  "
          f"mean={combined[f'reply_repost_ratio_{w_full}m'].mean():.2f}")
 
    # Drop raw window counts from combined before merge
    drop_from_combined = [
        f"{p}_{w}m"
        for p in ["repost", "like", "reply", "quote"]
        for w in windows
        if f"{p}_{w}m" in combined.columns
    ]


    
    # BLOCK 6: EMOTION FEATURES

    print("\n" + "=" * 70)
    print("BLOCK 6: Emotion features (RoBERTa)")
    print("=" * 70)

    tokenizer, emotion_model, device = load_model(HF_MODEL)
    id2label = emotion_model.config.id2label
    labels   = [id2label[i] for i in range(len(id2label))]

    texts = posts["text"].tolist()
    print(f"  Running inference on {len(texts):,} posts ...")
    probs = predict_emotions(texts, tokenizer, emotion_model, device)

    emotion_feat = posts[["uri"]].copy()
    for i, label in enumerate(labels):
        emotion_feat[f"emotion_{label}"] = probs[:, i].astype("float32")
    emotion_feat["emotion_label"] = [labels[i] for i in probs.argmax(axis=1)]

    print("\n  Emotion distribution (argmax):")
    dist = emotion_feat["emotion_label"].value_counts()
    for lbl, cnt in dist.items():
        print(f"    {lbl:<20s}: {cnt:>6,}  ({cnt/len(emotion_feat)*100:.1f}%)")


    # BLOCK 7: ASSEMBLE FINAL FEATURE TABLE

    print("\n" + "=" * 70)
    print("BLOCK 7: Assembling final feature table")
    print("=" * 70)
 
    feat = posts[["uri", "is_viral"]].copy()
    feat["is_viral"] = feat["is_viral"].astype(int)
 
    for block_df in [
        repost_counts, like_counts, reply_counts, quote_counts,
        ttf_repost, ttf_like, ttf_reply, ttf_quote,
        content, author_hist,
        net_reposts, net_likes, net_replies, net_quotes,
        combined.drop(columns=drop_from_combined, errors="ignore"),
        emotion_feat,
    ]:
        feat = feat.merge(block_df, on="uri", how="left")
 
    print(f"\n  Final feature table: {len(feat):,} rows x {len(feat.columns)} columns")
    print(f"  Viral posts:     {feat['is_viral'].sum():,}  ({feat['is_viral'].mean()*100:.2f}%)")
    print(f"  Non-viral posts: {(feat['is_viral']==0).sum():,}")

    
    # BLOCK 8: VALIDATION
    
    print("\n" + "=" * 70)
    print("BLOCK 8: Validation")
    print("=" * 70)
 
    null_counts = feat.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print(f"  WARNING: Columns with nulls:")
        for col, n in null_counts.items():
            print(f"      {col:<45s}: {n:,} nulls")
    else:
        print("  No null values")
 
    assert feat["uri"].nunique() == len(feat), "Duplicate URIs in output!"
    print("  All URIs unique")
 
    for prefix in ["repost", "like", "reply", "quote"]:
        for w1, w2 in zip(windows[:-1], windows[1:]):
            c1, c2 = f"{prefix}_{w1}m", f"{prefix}_{w2}m"
            if c1 in feat.columns and c2 in feat.columns:
                violations = (feat[c2] < feat[c1]).sum()
                status = "OK" if violations == 0 else f"VIOLATIONS: {violations}"
                print(f"  {status}  {prefix}: {w1}m <= {w2}m")
 
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    inf_counts   = np.isinf(feat[numeric_cols]).sum()
    inf_cols     = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print("  WARNING: Inf values found - replacing with 0:")
        for col, n in inf_cols.items():
            print(f"      {col:<45s}: {n:,} infs")
        feat[numeric_cols] = feat[numeric_cols].replace([np.inf, -np.inf], 0)
    else:
        print("  No inf values")
 
    check_cols = [
        f"repost_{w_burst}m", f"repost_{w_full}m", f"repost_{w_full}m_log",
        "repost_velocity_ratio", "repost_burst_ratio",
        f"like_{w_burst}m", f"like_{w_full}m",
        "like_velocity_ratio", "like_burst_ratio",
        f"reply_{w_full}m", f"quote_{w_full}m",
        "ttf_repost_sec", "ttf_like_sec",
        f"total_engagement_{w_full}m",
        "repost_engager_followers_max", "repost_whale_count",
        "author_followers", "author_hist_mean_reposts",
    ]
    print(f"\n  {'Feature':<40s} {'Viral':>10s} {'Non-viral':>10s} {'Ratio':>8s}")
    print(f"  {'-'*72}")
    for col in check_cols:
        if col not in feat.columns:
            continue
        v_mean  = feat[feat["is_viral"] == 1][col].mean()
        nv_mean = feat[feat["is_viral"] == 0][col].mean()
        ratio   = v_mean / (nv_mean + 0.001)
        print(f"  {col:<40s} {v_mean:>10.3f} {nv_mean:>10.3f} {ratio:>7.1f}x")

    
    # BLOCK 9: SAVE
    
    print("\n" + "=" * 70)
    print("BLOCK 9: Saving outputs")
    print("=" * 70)
 
    stem         = f"training_features_{w_full}m"
    parquet_path = OUTPUT_DIR / f"{stem}.parquet"
    feat.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}")
    print(f"     {len(feat):,} rows x {len(feat.columns)} columns")
 
    if SAVE_CSV:
        csv_path = OUTPUT_DIR / f"{stem}.csv"
        feat.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
 
    feature_cols = [c for c in feat.columns if c not in ("uri", "is_viral")]
    print(f"\n  All feature columns ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")
 
    print("\n" + "=" * 70)
    print(f"DONE - {parquet_path.name} ready for modelling")
    print("=" * 70)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract virality features up to a given time window."
    )
    parser.add_argument(
        "--window", type=int, default=30,
        help=(
            "Observation window in minutes (default: 30). "
            "Split into 5 equal fractions automatically. "
            "Examples: --window 5, --window 30, --window 60"
        )
    )
    args = parser.parse_args()
    main(window_min=args.window)