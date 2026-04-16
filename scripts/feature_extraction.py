"""
feature_extraction.py

Uses CascadeStore to build a flat training-ready feature table.
All window aggregations are vectorised — no row-by-row loops.

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
    python scripts/feature_extraction.py
"""

import os
import sys
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

WINDOWS_MIN     = [5, 15, 30, 60]
WHALE_THRESHOLD = 1_000
NETWORK_WINDOW = 30


# HELPERS


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
        sec   = w * 60
        mask  = events["time_delta_sec"].between(0, sec)
        counts = (
            events[mask]
            .groupby("root_post_uri")
            .size()
            .rename(f"{prefix}_{w}m")
        )
        result = result.join(counts, how="left")
        result[f"{prefix}_{w}m"] = result[f"{prefix}_{w}m"].fillna(0).astype(int)

    return result.reset_index()


def time_to_first(events: pd.DataFrame,
                  posts: pd.DataFrame,
                  col_name: str,
                  sentinel: float = 9_999.0) -> pd.DataFrame:
    """
    Seconds from post creation to first event.
    Posts with no events get the sentinel value (9999).
    """
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


def velocity_features(counts_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Derive acceleration and velocity ratio columns from raw window counts.
    Requires columns: {prefix}_5m, {prefix}_15m, {prefix}_30m, {prefix}_60m
    """
    df = counts_df.copy()
    p  = prefix

    # Volume in second half of 30m window (15→30m)
    df[f"{p}_15_to_30m"] = df[f"{p}_30m"] - df[f"{p}_15m"]

    # Acceleration: positive = speeding up, negative = dying down
    df[f"{p}_acceleration"] = df[f"{p}_15_to_30m"] - df[f"{p}_15m"]

    # Velocity ratio: second half / first half  (>1 = still accelerating at 30m)
    df[f"{p}_velocity_ratio"] = (
        df[f"{p}_15_to_30m"] / (df[f"{p}_15m"] + 1)
    )

    # Early burst ratio: fraction of 30m events that hit in first 5m
    df[f"{p}_burst_ratio"] = (
        df[f"{p}_5m"] / (df[f"{p}_30m"] + 1)
    )

    # Log transformed 30min counts (handles long tail)
    df[f"{p}_30m_log"] = np.log1p(df[f"{p}_30m"])

    # Long-run growth: how much did it keep growing after the 30m window?
    if f"{p}_60m" in df.columns:
        df[f"{p}_growth_30_to_60m"] = (
            df[f"{p}_60m"] / (df[f"{p}_30m"] + 1)
        )

    return df

# ── HELPERS (Engager Network Quality) ────────────────

def engager_network_features(events: pd.DataFrame,
                             posts: pd.DataFrame,
                             prefix: str,
                             did_col: str,
                             followers_col: str,
                             window_min: int = NETWORK_WINDOW,
                             whale_threshold: int = WHALE_THRESHOLD) -> pd.DataFrame:
    """
    Network quality of early engagers using follower data
    already embedded in the event table.

    Used for reposts (reposter_followers), replies (replier_followers),
    and quotes (quoter_followers).
    """
    result = posts[["uri"]].set_index("uri")
    p = prefix

    default_cols = [
        f"{p}_engager_count",
        f"{p}_engager_followers_max",
        f"{p}_engager_followers_mean",
        f"{p}_engager_followers_median",
        f"{p}_engager_followers_sum",
        f"{p}_engager_followers_std",
        f"{p}_engager_reach_log",
        f"{p}_whale_count",
        f"{p}_whale_ratio",
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

    # Deduplicate: one engagement per user per post
    windowed = windowed.drop_duplicates(subset=["root_post_uri", did_col])

    # Clean follower values
    windowed["_followers"] = pd.to_numeric(
        windowed[followers_col], errors="coerce"
    ).fillna(0)

    # Aggregate per post
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

    agg[f"{p}_engager_reach_log"] = np.log1p(agg[f"{p}_engager_followers_sum"])

    # Whale analysis
    whale_counts = (windowed[windowed["_followers"] >= whale_threshold]
                    .groupby("root_post_uri")
                    .size()
                    .rename(f"{p}_whale_count"))
    agg = agg.join(whale_counts, how="left")
    agg[f"{p}_whale_count"] = agg[f"{p}_whale_count"].fillna(0).astype(int)
    agg[f"{p}_whale_ratio"] = (
        agg[f"{p}_whale_count"] / (agg[f"{p}_engager_count"] + 1)
    )

    agg[f"{p}_reach_concentration"] = (
        agg[f"{p}_engager_followers_max"] / (agg[f"{p}_engager_followers_sum"] + 1)
    )

    result = result.join(agg, how="left")
    for col in default_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
        else:
            result[col] = 0.0

    return result.reset_index()


def build_liker_follower_lookup(reposts, replies, quotes) -> pd.DataFrame:
    """
    Likes don't have follower data, but reposts/replies/quotes do.
    Build a DID → follower_count lookup from the other three tables.
    """
    chunks = []

    if not reposts.empty and "reposter_did" in reposts.columns:
        chunks.append(
            reposts[["reposter_did", "reposter_followers"]]
            .rename(columns={"reposter_did": "did",
                              "reposter_followers": "followers"})
        )
    if not replies.empty and "replier_did" in replies.columns:
        chunks.append(
            replies[["replier_did", "replier_followers"]]
            .rename(columns={"replier_did": "did",
                              "replier_followers": "followers"})
        )
    if not quotes.empty and "quoter_did" in quotes.columns:
        chunks.append(
            quotes[["quoter_did", "quoter_followers"]]
            .rename(columns={"quoter_did": "did",
                              "quoter_followers": "followers"})
        )

    if not chunks:
        return pd.DataFrame(columns=["did", "followers"])

    lookup = pd.concat(chunks, ignore_index=True)
    lookup["followers"] = pd.to_numeric(
        lookup["followers"], errors="coerce"
    ).fillna(0)

    # Take max per user (most recent / highest observed)
    lookup = lookup.groupby("did")["followers"].max().reset_index()
    return lookup


def like_network_features(likes: pd.DataFrame,
                          posts: pd.DataFrame,
                          follower_lookup: pd.DataFrame,
                          window_min: int = NETWORK_WINDOW,
                          whale_threshold: int = WHALE_THRESHOLD) -> pd.DataFrame:
    """
    Network features for likes using cross-table follower lookup.
    Coverage is partial — only likers who also appear in other tables.
    """
    result = posts[["uri"]].set_index("uri")
    p = "like"

    default_cols = [
        f"{p}_engager_count",
        f"{p}_engager_followers_max",
        f"{p}_engager_followers_mean",
        f"{p}_engager_followers_median",
        f"{p}_engager_followers_sum",
        f"{p}_engager_followers_std",
        f"{p}_engager_reach_log",
        f"{p}_whale_count",
        f"{p}_whale_ratio",
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

    # Deduplicate
    windowed = windowed.drop_duplicates(subset=["root_post_uri", "liker_did"])

    # Join follower lookup
    windowed = windowed.merge(
        follower_lookup.rename(columns={"did": "liker_did",
                                         "followers": "_followers"}),
        on="liker_did", how="left"
    )
    windowed["_followers"] = windowed["_followers"].fillna(0)

    matched = (windowed["_followers"] > 0).sum()
    total   = len(windowed)
    print(f"    Like follower coverage: {matched:,} / {total:,} "
          f"({matched / max(total, 1) * 100:.1f}%)")

    # Aggregate
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
    agg[f"{p}_engager_reach_log"] = np.log1p(agg[f"{p}_engager_followers_sum"])

    whale_counts = (windowed[windowed["_followers"] >= whale_threshold]
                    .groupby("root_post_uri").size()
                    .rename(f"{p}_whale_count"))
    agg = agg.join(whale_counts, how="left")
    agg[f"{p}_whale_count"] = agg[f"{p}_whale_count"].fillna(0).astype(int)
    agg[f"{p}_whale_ratio"] = (
        agg[f"{p}_whale_count"] / (agg[f"{p}_engager_count"] + 1)
    )
    agg[f"{p}_reach_concentration"] = (
        agg[f"{p}_engager_followers_max"] / (agg[f"{p}_engager_followers_sum"] + 1)
    )

    result = result.join(agg, how="left")
    for col in default_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
        else:
            result[col] = 0.0

    return result.reset_index()


# ── HELPER (Author Baseline) ─────────────────────────

def author_history_features(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-out author history using only the posts DataFrame.
    Uses exact columns: did, created_at, total_likes, total_reposts,
    total_replies, total_quotes.
    """
    df = posts.copy()
    result = posts[["uri"]].copy()

    print(f"  Unique authors: {df['did'].nunique():,}")

    # Sort by author + creation time
    df["_ts"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values(["did", "_ts"]).reset_index(drop=True)

    # Prior post count (0-indexed = number of posts before this one)
    result["author_prior_post_count"] = df.groupby("did").cumcount().values
    result["author_prior_post_count_log"] = np.log1p(
        result["author_prior_post_count"]
    )

    # Leave-one-out mean and max for each engagement metric
    for metric, raw_col in [
        ("likes",   "total_likes"),
        ("reposts", "total_reposts"),
        ("replies", "total_replies"),
        ("quotes",  "total_quotes"),
    ]:
        vals = pd.to_numeric(df[raw_col], errors="coerce").fillna(0)

        cum_sum   = vals.groupby(df["did"]).cumsum() - vals
        cum_count = df.groupby("did").cumcount()

        # Mean of all prior posts
        result[f"author_hist_mean_{metric}"] = np.where(
            cum_count > 0, cum_sum / cum_count, 0.0
        )

        # Max of all prior posts
        shifted_max = (vals.groupby(df["did"])
                       .apply(lambda s: s.shift(1).expanding().max())
                       .reset_index(level=0, drop=True)
                       .fillna(0))
        result[f"author_hist_max_{metric}"] = shifted_max.values

    return result


def print_window_stats(name: str, df: pd.DataFrame, prefix: str, windows: list[int]):
    """Print per-window count statistics for one event type."""
    print(f"\n  {name.upper()}")
    for w in windows:
        col = f"{prefix}_{w}m"
        if col not in df.columns:
            continue
        n_active = (df[col] > 0).sum()
        pct      = n_active / len(df) * 100
        mean_val = df[col].mean()
        p95_val  = df[col].quantile(0.95)
        max_val  = df[col].max()
        print(
            f"    {w:>3}m  |  posts with ≥1: {n_active:>6,} ({pct:>5.1f}%)  "
            f"|  mean: {mean_val:>6.2f}  |  p95: {p95_val:>6.0f}  |  max: {max_val:>6.0f}"
        )


def print_velocity_stats(name: str, df: pd.DataFrame, prefix: str):
    """Print derived velocity feature statistics."""
    print(f"\n  {name.upper()} velocity features:")
    vel_cols = [
        f"{prefix}_15_to_30m",
        f"{prefix}_acceleration",
        f"{prefix}_velocity_ratio",
        f"{prefix}_burst_ratio",
        f"{prefix}_30m_log",
        f"{prefix}_growth_30_to_60m",
    ]
    for col in vel_cols:
        if col not in df.columns:
            continue
        s = df[col]
        print(
            f"    {col:<35s}  mean={s.mean():>8.3f}  "
            f"median={s.median():>7.3f}  "
            f"p95={s.quantile(0.95):>8.3f}  "
            f"max={s.max():>10.3f}"
        )


def print_ttf_stats(name: str, df: pd.DataFrame, col: str, sentinel: float = 9_999.0):
    """Print time-to-first-event stats, excluding sentinel values."""
    has_event = df[df[col] < sentinel]
    no_event  = df[df[col] >= sentinel]
    print(
        f"  {name:<10s}  posts with event: {len(has_event):>6,}  "
        f"| no event (sentinel): {len(no_event):>6,}"
    )
    if not has_event.empty:
        s = has_event[col]
        print(
            f"            median: {s.median():>8.0f}s  "
            f"| mean: {s.mean():>8.0f}s  "
            f"| p95: {s.quantile(0.95):>8.0f}s  "
            f"| min: {s.min():>6.0f}s"
        )

# ── Network stats printer ────────────────────────────────────
def print_network_stats(name: str, df: pd.DataFrame, prefix: str):
    p = prefix
    active = (df[f"{p}_engager_count"] > 0).sum()
    whale  = df[f"{p}_whale_count"].sum()
    print(f"\n  {name.upper()} network quality:"
          f"\n    posts with engagers: {active:>6,} / {len(df):,}"
          f"\n    total whale engagements: {whale:>6,.0f}")
    for col in [f"{p}_engager_followers_max", f"{p}_engager_followers_mean",
                f"{p}_engager_followers_sum", f"{p}_reach_concentration"]:
        if col in df.columns:
            s = df[col]
            print(f"    {col:<40s}  mean={s.mean():>10.2f}  "
                  f"median={s.median():>8.2f}")

# MAIN

def main():
    print("=" * 70)
    print("FEATURE EXTRACTION — CascadeStore")
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

        # ══════════════════════════════════════════════════════════
    # BLOCK 1: WINDOW COUNTS (Layer 1 — original)
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("BLOCK 1: Window counts (5m / 15m / 30m / 60m)")
    print("=" * 70)

    repost_counts = window_counts(reposts, posts, "repost", WINDOWS_MIN)
    like_counts   = window_counts(likes,   posts, "like",   WINDOWS_MIN)
    reply_counts  = window_counts(replies, posts, "reply",  WINDOWS_MIN)
    quote_counts  = window_counts(quotes,  posts, "quote",  WINDOWS_MIN)

    print_window_stats("reposts", repost_counts, "repost", WINDOWS_MIN)
    print_window_stats("likes",   like_counts,   "like",   WINDOWS_MIN)
    print_window_stats("replies", reply_counts,  "reply",  WINDOWS_MIN)
    print_window_stats("quotes",  quote_counts,  "quote",  WINDOWS_MIN)

    # ══════════════════════════════════════════════════════════
    # BLOCK 2: VELOCITY & ACCELERATION (Layer 1 — original + log)
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("BLOCK 2: Velocity & acceleration features")
    print("=" * 70)

    repost_counts = velocity_features(repost_counts, "repost")
    like_counts   = velocity_features(like_counts,   "like")
    reply_counts  = velocity_features(reply_counts,  "reply")
    quote_counts  = velocity_features(quote_counts,  "quote")

    print_velocity_stats("reposts", repost_counts, "repost")
    print_velocity_stats("likes",   like_counts,   "like")
    print_velocity_stats("replies", reply_counts,  "reply")
    print_velocity_stats("quotes",  quote_counts,  "quote")

    # ══════════════════════════════════════════════════════════
    # BLOCK 3: TIME-TO-FIRST-EVENT (Layer 1 — original)
    # ══════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════
    # BLOCK 4: CONTENT & AUTHOR FEATURES (Layer 3 — expanded)
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("BLOCK 4: Content & author features (expanded)")
    print("=" * 70)

    content = posts[["uri"]].copy()

    # ── Original features ─────────────────────────────────────
    content["text_len"] = posts["text"].fillna("").str.len()
    content["has_embed"] = (
        posts.get("has_embed", pd.Series(0, index=posts.index))
        .fillna(0).astype(int)
    )
    content["author_followers"] = pd.to_numeric(
        posts.get("author_followers", pd.Series(0, index=posts.index)),
        errors="coerce"
    ).fillna(0).astype(int)
    content["author_followers_log"] = np.log1p(content["author_followers"])
    content["author_is_whale"]      = (
        content["author_followers"] >= WHALE_THRESHOLD
    ).astype(int)

    # ── NEW (Layer 3): author following count ─────────────────
    content["author_follows"] = pd.to_numeric(
        posts["author_follows"], errors="coerce"
    ).fillna(0).astype(int)
    content["author_follows_log"] = np.log1p(content["author_follows"])

    # ── NEW (Layer 3): follower-to-following ratio ────────────
    content["author_ff_ratio"] = (
        content["author_followers"] / (content["author_follows"] + 1)
    )

    # ── NEW (Layer 3): author total post count from API ───────
    content["author_posts_count"] = pd.to_numeric(
        posts["author_posts_count"], errors="coerce"
    ).fillna(0).astype(int)
    content["author_posts_count_log"] = np.log1p(content["author_posts_count"])

    # ── NEW (Layer 3): leave-one-out author history ───────────
    print("\n  Computing leave-one-out author history...")
    author_hist = author_history_features(posts)

    # ── Stats ─────────────────────────────────────────────────
    print(f"\n  text_len          mean={content['text_len'].mean():.0f}  "
          f"| median={content['text_len'].median():.0f}  "
          f"| max={content['text_len'].max()}")
    print(f"  has_embed         rate={content['has_embed'].mean()*100:.1f}%  "
          f"| count={content['has_embed'].sum():,}")
    print(f"  author_followers  mean={content['author_followers'].mean():.0f}  "
          f"| median={content['author_followers'].median():.0f}  "
          f"| max={content['author_followers'].max()}")
    print(f"  author_follows    mean={content['author_follows'].mean():.0f}  "
          f"| median={content['author_follows'].median():.0f}")
    print(f"  author_ff_ratio   mean={content['author_ff_ratio'].mean():.2f}  "
          f"| median={content['author_ff_ratio'].median():.2f}")
    print(f"  author_is_whale   rate={content['author_is_whale'].mean()*100:.1f}%  "
          f"| count={content['author_is_whale'].sum():,}")
    print(f"  author_posts_count  mean={content['author_posts_count'].mean():.0f}  "
          f"| median={content['author_posts_count'].median():.0f}")
    print(f"  prior_post_count  mean={author_hist['author_prior_post_count'].mean():.1f}  "
          f"| median={author_hist['author_prior_post_count'].median():.0f}")
    for m in ["likes", "reposts", "replies", "quotes"]:
        col = f"author_hist_mean_{m}"
        if col in author_hist.columns:
            print(f"  {col:<40s}  mean={author_hist[col].mean():.2f}  "
                  f"| median={author_hist[col].median():.2f}")

    # ══════════════════════════════════════════════════════════
    # BLOCK 4B: ENGAGER NETWORK QUALITY (NEW — Layer 2)
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("BLOCK 4B: Engager network quality (30m window)")
    print("=" * 70)

    # Reposts, replies, quotes: follower data is IN the event table
    net_reposts = engager_network_features(
        reposts, posts, "repost",
        did_col="reposter_did",
        followers_col="reposter_followers"
    )
    net_replies = engager_network_features(
        replies, posts, "reply",
        did_col="replier_did",
        followers_col="replier_followers"
    )
    net_quotes = engager_network_features(
        quotes, posts, "quote",
        did_col="quoter_did",
        followers_col="quoter_followers"
    )

    # Likes: no follower data — build cross-table lookup
    print("\n  Building liker follower lookup from other tables...")
    liker_lookup = build_liker_follower_lookup(reposts, replies, quotes)
    print(f"    Lookup size: {len(liker_lookup):,} unique DIDs")

    net_likes = like_network_features(likes, posts, liker_lookup)

    print_network_stats("reposts", net_reposts, "repost")
    print_network_stats("likes",   net_likes,   "like")
    print_network_stats("replies", net_replies,  "reply")
    print_network_stats("quotes",  net_quotes,   "quote")

    # ══════════════════════════════════════════════════════════
    # BLOCK 5: COMBINED ENGAGEMENT FEATURES (original)
    # ══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("BLOCK 5: Combined engagement features (30m window)")
    print("=" * 70)

    combined_30m = (
        posts[["uri"]]
        .merge(repost_counts[["uri", "repost_30m"]], on="uri", how="left")
        .merge(like_counts  [["uri", "like_30m"]],   on="uri", how="left")
        .merge(reply_counts [["uri", "reply_30m"]],  on="uri", how="left")
        .merge(quote_counts [["uri", "quote_30m"]],  on="uri", how="left")
        .fillna(0)
    )
    combined_30m["total_engagement_30m"] = (
        combined_30m["repost_30m"] + combined_30m["like_30m"] +
        combined_30m["reply_30m"]  + combined_30m["quote_30m"]
    )
    combined_30m["like_repost_ratio_30m"] = (
        combined_30m["like_30m"] / (combined_30m["repost_30m"] + 1)
    )
    combined_30m["reply_repost_ratio_30m"] = (
        combined_30m["reply_30m"] / (combined_30m["repost_30m"] + 1)
    )
    combined_30m["quote_repost_ratio_30m"] = (
        combined_30m["quote_30m"] / (combined_30m["repost_30m"] + 1)
    )

    print(f"  total_engagement_30m    "
          f"mean={combined_30m['total_engagement_30m'].mean():.2f}  "
          f"| median={combined_30m['total_engagement_30m'].median():.0f}  "
          f"| max={combined_30m['total_engagement_30m'].max():.0f}")
    print(f"  posts with any engagement: "
          f"{(combined_30m['total_engagement_30m'] > 0).sum():,} / "
          f"{len(combined_30m):,}")
    print(f"  like_repost_ratio_30m   "
          f"mean={combined_30m['like_repost_ratio_30m'].mean():.2f}")
    print(f"  reply_repost_ratio_30m  "
          f"mean={combined_30m['reply_repost_ratio_30m'].mean():.2f}")
    print(f"  quote_repost_ratio_30m  "
          f"mean={combined_30m['quote_repost_ratio_30m'].mean():.2f}")


    
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
        # Layer 1: velocity + TTF
        repost_counts, like_counts, reply_counts, quote_counts,
        ttf_repost, ttf_like, ttf_reply, ttf_quote,
        # Layer 3: author + content
        content,
        author_hist,                                            # NEW
        # Layer 2: network quality
        net_reposts, net_likes, net_replies, net_quotes,        # NEW
        # Combined
        combined_30m.drop(
            columns=["repost_30m", "like_30m", "reply_30m", "quote_30m"]),
        emotion_feat,
    ]:
        feat = feat.merge(block_df, on="uri", how="left")

    print(f"\n  Final feature table: {len(feat):,} rows × "
          f"{len(feat.columns)} columns")
    print(f"  Viral posts:     {feat['is_viral'].sum():,}  "
          f"({feat['is_viral'].mean()*100:.2f}%)")
    print(f"  Non-viral posts: {(feat['is_viral']==0).sum():,}")

    
    # BLOCK 8: VALIDATION
    
    print("\n" + "=" * 70)
    print("BLOCK 8: Validation")
    print("=" * 70)

    # Null check 
    null_counts = feat.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print(f"  ⚠ Columns with nulls:")
        for col, n in null_counts.items():
            print(f"      {col:<40s}: {n:,} nulls")
    else:
        print("  ✓ No null values")

    # Duplicate URI check 
    assert feat["uri"].nunique() == len(feat), "Duplicate URIs in output!"
    print("  ✓ All URIs unique")

    # Monotonic window check 
    # Counts at 15m must be <= counts at 30m, etc — catches time_delta_sec bugs
    for prefix in ["repost", "like", "reply", "quote"]:
        for w1, w2 in [(5, 15), (15, 30), (30, 60)]:
            c1, c2 = f"{prefix}_{w1}m", f"{prefix}_{w2}m"
            if c1 in feat.columns and c2 in feat.columns:
                violations = (feat[c2] < feat[c1]).sum()
                status = "✓" if violations == 0 else f"⚠ {violations} violations"
                print(f"  {status}  {prefix}: {w1}m ≤ {w2}m counts")

    # NEW: Inf check
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    inf_counts   = np.isinf(feat[numeric_cols]).sum()
    inf_cols     = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print(f"   Columns with inf values:")
        for col, n in inf_cols.items():
            print(f"      {col:<45s}: {n:,} infs")
        feat[numeric_cols] = feat[numeric_cols].replace(
            [np.inf, -np.inf], 0)
        print("    Replaced inf values with 0")
    else:
        print("   No inf values")


    # Viral vs Non-Viral mean comparison 
    check_cols = [
       "repost_5m", "repost_30m", "repost_30m_log",           # L1
        "repost_velocity_ratio", "repost_burst_ratio",          # L1
        "like_5m", "like_30m", "like_30m_log",                  # L1
        "like_velocity_ratio", "like_burst_ratio",              # L1
        "reply_30m", "quote_30m",                               # L1
        "ttf_repost_sec", "ttf_like_sec",                       # L1
        "total_engagement_30m",                                 # combined
        "repost_engager_followers_max",                         # L2
        "repost_engager_followers_sum",                         # L2
        "like_engager_followers_max",                           # L2
        "repost_whale_count",                                   # L2
        "like_whale_count",                                     # L2
        "repost_reach_concentration",                           # L2
        "author_followers",                                     # L3
        "author_follows",                                       # L3
        "author_ff_ratio",                                      # L3
        "author_posts_count",                                   # L3
        "author_prior_post_count",                              # L3
        "author_hist_mean_likes",                               # L3
        "author_hist_mean_reposts",                             # L3
        "author_hist_max_reposts", 
    ]
    print(f"\n  {'Feature':<35s} {'Viral mean':>12s} {'Non-viral mean':>14s} {'Ratio':>8s}")
    print(f"  {'-'*73}")
    for col in check_cols:
        if col not in feat.columns:
            continue
        v_mean  = feat[feat["is_viral"] == 1][col].mean()
        nv_mean = feat[feat["is_viral"] == 0][col].mean()
        ratio   = v_mean / (nv_mean + 0.001)
        print(f"  {col:<35s} {v_mean:>12.3f} {nv_mean:>14.3f} {ratio:>7.1f}x")

    
    # BLOCK 9: SAVE
    
    print("\n" + "=" * 70)
    print("BLOCK 9: Saving outputs")
    print("=" * 70)

    parquet_path = OUTPUT_DIR / "training_features.parquet"
    feat.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}")
    print(f"     {len(feat):,} rows × {len(feat.columns)} columns")

    if SAVE_CSV:
        csv_path = OUTPUT_DIR / "training_features.csv"
        feat.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    feature_cols = [c for c in feat.columns if c not in ("uri", "is_viral")]
    print(f"\n  All feature columns ({len(feature_cols)} total):")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")

    print("\n" + "=" * 70)
    print("DONE — training_features.parquet ready for modelling")
    print("=" * 70)
    return feat


if __name__ == "__main__":
    feat = main()