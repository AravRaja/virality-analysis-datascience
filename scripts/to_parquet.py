"""
Convert all raw dataset files to parquet for fast loading.
Run once — safe to re-run, skips files already converted.

Usage:
    python scripts/to_parquet.py
"""

import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    print(f"saved as {path.name} ")


def skip(path: Path) -> bool:
    if path.exists():
        print(f"{path.name} already exists!")
        return True
    return False


# interactions
print("interactions.csv.gz")
dest = OUT / "interactions.parquet"
if not skip(dest):
    df = pd.read_csv(
        RAW / "interactions.csv.gz",
        header=None,
        names=["user_id", "replied_author", "thread_root_author",
               "reposted_author", "quoted_author", "date"],
    )
    save(df, dest)


# followers
print("followers.csv.gz")
dest = OUT / "followers.parquet"
if not skip(dest):
    df = pd.read_csv(RAW / "followers.csv.gz", header=None,
                     names=["follower_id", "followed_id"])
    save(df, dest)


# feed_bookmarks
print("feed_bookmarks.csv")
dest = OUT / "feed_bookmarks.parquet"
if not skip(dest):
    df = pd.read_csv(RAW / "feed_bookmarks.csv", header=None)
    save(df, dest)


# graphs
GRAPH_COLS = {
    "reposts.csv": ["user_id", "reposted_author", "date"],
    "replies.csv": ["user_id", "replied_author", "date"],
    "quotes.csv":  ["user_id", "quoted_author", "date"],
}

for filename, cols in GRAPH_COLS.items():
    print(f"graphs/{filename}")
    dest = OUT / filename.replace(".csv", ".parquet")
    if not skip(dest):
        df = pd.read_csv(RAW / "graphs" / filename, header=None, names=cols)
        save(df, dest)


# feed_posts
feed_posts_dir = RAW / "feed_posts"
for gz_file in sorted(feed_posts_dir.glob("*.jsonl.gz")):
    feed_name = gz_file.stem.replace(".jsonl", "")
    dest = OUT / f"feed_{feed_name}.parquet"
    print(f"feed_posts/{gz_file.name}")
    if skip(dest):
        continue
    records = []
    with gzip.open(gz_file) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    save(pd.DataFrame(records), dest)


# feed_posts_likes
feed_likes_dir = RAW / "feed_posts_likes"
for gz_file in sorted(feed_likes_dir.glob("*.csv.gz")):
    feed_name = gz_file.stem.replace(".csv", "")
    dest = OUT / f"feed_likes_{feed_name}.parquet"
    print(f"feed_posts_likes/{gz_file.name}")
    if skip(dest):
        continue
    df = pd.read_csv(gz_file, header=None,
                     names=["like_id", "user_id", "post_id", "date"])
    save(df, dest)


print("\nAll done.")
