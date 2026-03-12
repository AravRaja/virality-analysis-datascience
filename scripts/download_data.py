"""
Downloads Bluesky (except userposts) to data/raw/ 
Usage:
    python scripts/download_data.py
"""

import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://zenodo.org/records/14669616/files"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

FILES = [
    "interactions.csv.gz",
    "followers.csv.gz",
    "graphs.tar.gz",
    "feed_posts.tar.gz",
    "feed_posts_likes.tar.gz",
    "feed_bookmarks.csv",
]


def download_file(filename: str, dest_dir: Path) -> None:
    url = f"{BASE_URL}/{filename}?download=1"
    dest = dest_dir / filename

    if dest.exists():
        print(f"{filename} already exists!")
        return

    print(f"Getting {filename}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            print(f"\r    {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        print()
        tmp.rename(dest)
        print(f"Saved to {dest}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {OUTPUT_DIR}\n")

    for filename in FILES:
        download_file(filename, OUTPUT_DIR)



if __name__ == "__main__":
    main()
