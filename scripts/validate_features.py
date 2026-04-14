"""
validate_features.py

Run this after feature_extraction.py to verify the output is correct
before handing it to a model.

Run:
    python scripts/validate_features.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — saves PNGs instead
import matplotlib.pyplot as plt
from pathlib import Path

CASCADE_DIR  = Path(__file__).resolve().parent.parent / "datasets" / "bluesky_cascade"
FEATURES_PATH = CASCADE_DIR / "training_features.parquet"
REPORT_DIR    = Path(__file__).resolve().parent.parent / "reports" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SENTINEL = 9_999.0   # must match feature_extraction.py

# =============================================================================
print("=" * 70)
print("VALIDATION — training_features.parquet")
print("=" * 70)

feat = pd.read_parquet(FEATURES_PATH)
print(f"\n  Loaded: {FEATURES_PATH}")
print(f"  Shape:  {feat.shape[0]:,} rows × {feat.shape[1]} columns")

# =============================================================================
# CHECK 1: Schema
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 1: Schema — all expected columns present")
print("=" * 70)

expected_cols = [
    # window counts
    "repost_5m", "repost_15m", "repost_30m", "repost_60m",
    "like_5m",   "like_15m",   "like_30m",   "like_60m",
    "reply_5m",  "reply_15m",  "reply_30m",  "reply_60m",
    "quote_5m",  "quote_15m",  "quote_30m",  "quote_60m",
    # velocity
    "repost_15_to_30m", "repost_acceleration", "repost_velocity_ratio", "repost_burst_ratio",
    "like_15_to_30m",   "like_acceleration",   "like_velocity_ratio",   "like_burst_ratio",
    "reply_15_to_30m",  "reply_acceleration",  "reply_velocity_ratio",  "reply_burst_ratio",
    "quote_15_to_30m",  "quote_acceleration",  "quote_velocity_ratio",  "quote_burst_ratio",
    # time to first
    "ttf_repost_sec", "ttf_like_sec", "ttf_reply_sec", "ttf_quote_sec",
    # content
    "text_len", "has_embed", "author_followers", "author_followers_log", "author_is_whale",
    # combined
    "total_engagement_30m", "like_repost_ratio_30m",
    "reply_repost_ratio_30m", "quote_repost_ratio_30m",
    # target
    "is_viral",
]

missing = [c for c in expected_cols if c not in feat.columns]
extra   = [c for c in feat.columns  if c not in expected_cols and c != "uri"]

if missing:
    print(f"  ⚠ MISSING columns ({len(missing)}):")
    for c in missing:
        print(f"      {c}")
else:
    print(f"  ✓ All {len(expected_cols)} expected columns present")

if extra:
    print(f"  ℹ Extra columns not in expected list ({len(extra)}):")
    for c in extra:
        print(f"      {c}")

# =============================================================================
# CHECK 2: Nulls
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 2: Null values")
print("=" * 70)

null_counts = feat.isnull().sum()
null_counts = null_counts[null_counts > 0]
if len(null_counts) == 0:
    print("  ✓ Zero null values across all columns")
else:
    print(f"  ⚠ {len(null_counts)} columns have nulls:")
    for col, n in null_counts.items():
        pct = n / len(feat) * 100
        print(f"      {col:<40s}: {n:>6,} ({pct:.1f}%)")

# =============================================================================
# CHECK 3: Duplicates
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 3: Duplicate URIs")
print("=" * 70)

n_dupes = len(feat) - feat["uri"].nunique()
if n_dupes == 0:
    print("  ✓ No duplicate URIs")
else:
    print(f"  ⚠ {n_dupes} duplicate URIs — investigate before modelling")

# =============================================================================
# CHECK 4: Class balance
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 4: Class balance")
print("=" * 70)

viral     = feat["is_viral"].sum()
non_viral = (feat["is_viral"] == 0).sum()
ratio     = non_viral / max(viral, 1)
print(f"  Viral:     {viral:>8,}  ({viral/len(feat)*100:.2f}%)")
print(f"  Non-viral: {non_viral:>8,}  ({non_viral/len(feat)*100:.2f}%)")
print(f"  Imbalance ratio: {ratio:.1f}:1  (non-viral : viral)")
if ratio > 20:
    print("  ⚠ Severe imbalance — use class_weight='balanced' or SMOTE when modelling")
elif ratio > 5:
    print("  ℹ Moderate imbalance — consider class weighting")
else:
    print("  ✓ Reasonable class balance")

# =============================================================================
# CHECK 5: Monotonic window counts
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 5: Monotonic window counts (5m ≤ 15m ≤ 30m ≤ 60m)")
print("=" * 70)

all_ok = True
for prefix in ["repost", "like", "reply", "quote"]:
    for w1, w2 in [(5, 15), (15, 30), (30, 60)]:
        c1, c2 = f"{prefix}_{w1}m", f"{prefix}_{w2}m"
        if c1 not in feat.columns or c2 not in feat.columns:
            continue
        violations = (feat[c2] < feat[c1]).sum()
        if violations > 0:
            all_ok = False
            print(f"  ⚠ {prefix}: {c2} < {c1} in {violations} rows")
        else:
            print(f"  ✓ {prefix}: {w1}m ≤ {w2}m")
if all_ok:
    print("  ✓ All window counts are monotonically non-decreasing")

# =============================================================================
# CHECK 6: Sentinel value check for TTF features
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 6: Time-to-first-event sentinel values (9999 = no event)")
print("=" * 70)

for col in ["ttf_repost_sec", "ttf_like_sec", "ttf_reply_sec", "ttf_quote_sec"]:
    if col not in feat.columns:
        continue
    n_sentinel = (feat[col] >= SENTINEL).sum()
    n_real     = (feat[col] < SENTINEL).sum()
    med_real   = feat[feat[col] < SENTINEL][col].median() if n_real > 0 else float("nan")
    print(f"  {col:<20s}  real events: {n_real:>6,}  "
          f"| sentinel (no event): {n_sentinel:>6,}  "
          f"| median real: {med_real:>8.0f}s")

# =============================================================================
# CHECK 7: Feature distribution — viral vs non-viral
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 7: Viral vs Non-viral feature means (signal check)")
print("=" * 70)

compare_cols = [
    "repost_5m", "repost_15m", "repost_30m", "repost_60m",
    "like_5m", "like_15m", "like_30m",
    "reply_30m", "quote_30m",
    "repost_velocity_ratio", "like_velocity_ratio",
    "repost_burst_ratio", "like_burst_ratio",
    "ttf_repost_sec", "ttf_like_sec",
    "total_engagement_30m", "author_followers",
]
compare_cols = [c for c in compare_cols if c in feat.columns]

print(f"\n  {'Feature':<35s} {'Viral':>12s} {'Non-viral':>12s} {'Ratio':>8s}  Signal?")
print(f"  {'-'*80}")
for col in compare_cols:
    v_mean  = feat[feat["is_viral"] == 1][col].mean()
    nv_mean = feat[feat["is_viral"] == 0][col].mean()
    ratio   = v_mean / (nv_mean + 0.001)
    # For TTF, lower = faster = more viral, so ratio is inverted
    is_ttf  = col.startswith("ttf_")
    signal  = "✓ STRONG" if (ratio > 3 or (is_ttf and ratio < 0.5)) else \
              "~ weak"  if (ratio > 1.5 or (is_ttf and ratio < 0.8)) else \
              "✗ none"
    print(f"  {col:<35s} {v_mean:>12.3f} {nv_mean:>12.3f} {ratio:>7.1f}x  {signal}")

# =============================================================================
# CHECK 8: Zero-engagement posts
# =============================================================================
print("\n" + "=" * 70)
print("CHECK 8: Zero-engagement posts (no activity in 60m window)")
print("=" * 70)

if all(c in feat.columns for c in ["repost_60m", "like_60m", "reply_60m", "quote_60m"]):
    zero_mask = (
        (feat["repost_60m"] == 0) &
        (feat["like_60m"]   == 0) &
        (feat["reply_60m"]  == 0) &
        (feat["quote_60m"]  == 0)
    )
    n_zero = zero_mask.sum()
    n_zero_viral = (zero_mask & (feat["is_viral"] == 1)).sum()
    print(f"  Posts with ZERO engagement in 60m: {n_zero:,} ({n_zero/len(feat)*100:.1f}%)")
    print(f"  Of those, labelled viral:          {n_zero_viral:,}")
    if n_zero_viral > 0:
        print(f"  ⚠ Viral posts with no early activity detected — check virality labelling")
    else:
        print(f"  ✓ No viral posts have zero early engagement")

# =============================================================================
# PLOTS — saved to reports/figures/
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING plots → reports/figures/")
print("=" * 70)

viral_df     = feat[feat["is_viral"] == 1]
nonviral_df  = feat[feat["is_viral"] == 0]

# Plot 1: Repost counts over time windows — viral vs non-viral
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, w in zip(axes, [5, 15, 30, 60]):
    col = f"repost_{w}m"
    if col not in feat.columns:
        continue
    cap = feat[col].quantile(0.99)   # cap at 99th pct for readability
    bins = np.linspace(0, cap, 40)
    ax.hist(nonviral_df[col].clip(upper=cap), bins=bins, alpha=0.6,
            label="Non-viral", color="steelblue", density=True)
    ax.hist(viral_df[col].clip(upper=cap),    bins=bins, alpha=0.6,
            label="Viral",     color="tomato",    density=True)
    ax.set_title(f"Reposts in {w}m")
    ax.set_xlabel("Count")
    ax.legend(fontsize=8)
fig.suptitle("Repost count distributions: Viral vs Non-viral", fontsize=13)
plt.tight_layout()
p = REPORT_DIR / "repost_distributions.png"
plt.savefig(p, dpi=120)
plt.close()
print(f"  Saved: {p}")

# Plot 2: Like counts over time windows
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, w in zip(axes, [5, 15, 30, 60]):
    col = f"like_{w}m"
    if col not in feat.columns:
        continue
    cap = feat[col].quantile(0.99)
    bins = np.linspace(0, cap, 40)
    ax.hist(nonviral_df[col].clip(upper=cap), bins=bins, alpha=0.6,
            label="Non-viral", color="steelblue", density=True)
    ax.hist(viral_df[col].clip(upper=cap),    bins=bins, alpha=0.6,
            label="Viral",     color="tomato",    density=True)
    ax.set_title(f"Likes in {w}m")
    ax.set_xlabel("Count")
    ax.legend(fontsize=8)
fig.suptitle("Like count distributions: Viral vs Non-viral", fontsize=13)
plt.tight_layout()
p = REPORT_DIR / "like_distributions.png"
plt.savefig(p, dpi=120)
plt.close()
print(f"  Saved: {p}")

# Plot 3: Time to first repost — viral vs non-viral (exclude sentinel)
fig, ax = plt.subplots(figsize=(8, 5))
col = "ttf_repost_sec"
if col in feat.columns:
    v_ttf  = viral_df[viral_df[col]   < SENTINEL][col]
    nv_ttf = nonviral_df[nonviral_df[col] < SENTINEL][col]
    cap    = feat[col][feat[col] < SENTINEL].quantile(0.99)
    bins   = np.linspace(0, cap, 60)
    ax.hist(nv_ttf.clip(upper=cap), bins=bins, alpha=0.6,
            label=f"Non-viral (n={len(nv_ttf):,})", color="steelblue", density=True)
    ax.hist(v_ttf.clip(upper=cap),  bins=bins, alpha=0.6,
            label=f"Viral (n={len(v_ttf):,})",     color="tomato",    density=True)
    ax.set_xlabel("Seconds to first repost")
    ax.set_title("Time to first repost: Viral vs Non-viral")
    ax.legend()
plt.tight_layout()
p = REPORT_DIR / "ttf_repost.png"
plt.savefig(p, dpi=120)
plt.close()
print(f"  Saved: {p}")

# Plot 4: Total engagement 30m — violin
fig, ax = plt.subplots(figsize=(7, 5))
cap = feat["total_engagement_30m"].quantile(0.99)
data_v  = viral_df["total_engagement_30m"].clip(upper=cap)
data_nv = nonviral_df["total_engagement_30m"].clip(upper=cap)
ax.violinplot([data_nv, data_v], positions=[0, 1], showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Non-viral", "Viral"])
ax.set_ylabel("Total engagement in 30m (capped at 99th pct)")
ax.set_title("Total 30m engagement distribution")
plt.tight_layout()
p = REPORT_DIR / "total_engagement_violin.png"
plt.savefig(p, dpi=120)
plt.close()
print(f"  Saved: {p}")

# Plot 5: Correlation heatmap (top 20 features by abs correlation with is_viral)
fig, ax = plt.subplots(figsize=(7, 8))
numeric_feat = feat.drop(columns=["uri"], errors="ignore").select_dtypes(include=[np.number])
corr = numeric_feat.corr()["is_viral"].drop("is_viral").abs().sort_values(ascending=False)
top20 = corr.head(20)
bars = ax.barh(top20.index[::-1], top20.values[::-1], color="steelblue")
ax.set_xlabel("|Pearson r| with is_viral")
ax.set_title("Top 20 features correlated with virality")
plt.tight_layout()
p = REPORT_DIR / "feature_correlations.png"
plt.savefig(p, dpi=120)
plt.close()
print(f"  Saved: {p}")

# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print(f"\n  All plots saved to: {REPORT_DIR}")
print("  Review reports/figures/ before proceeding to modelling.\n")