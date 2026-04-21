"""
visualise_features.py

Produces 6 publication-ready plots from training_features_{WINDOW_MIN}m.parquet:

    1. Scatter  — early vs sustained engagement (linear + log side-by-side)
    2. Box      — early repost count (raw + log) by viral / non-viral
    3. Box      — velocity ratio by viral / non-viral
    4. Box      — time to first engagement by viral / non-viral
    5. Box      — total likes snapshot (author proxy until follower data fetched)
    6. Line     — engagement rate per time bucket (dynamic window)

Run:
    python scripts/visualise_features.py
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "reports" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Change this to match the --window you ran in feature_extraction.py
WINDOW_MIN    = 30
FEATURES_PATH = (ROOT / "datasets" / "bluesky_cascade"
                 / f"training_features_{WINDOW_MIN}m.parquet")

SENTINEL = 9_999.0

# =============================================================================
# STYLE
# =============================================================================

VIRAL_COLOR    = "#E05C5C"
NONVIRAL_COLOR = "#4A90D9"
VIRAL_LABEL    = "Viral"
NONVIRAL_LABEL = "Non-viral"
ORDER          = [VIRAL_LABEL, NONVIRAL_LABEL]
PALETTE        = {VIRAL_LABEL: VIRAL_COLOR, NONVIRAL_LABEL: NONVIRAL_COLOR}

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.frameon":    False,
})

# =============================================================================
# LOAD
# =============================================================================

print("=" * 65)
print("VIRALITY FEATURE VISUALISATION")
print("=" * 65)
print(f"\n  Loading: {FEATURES_PATH}")

feat = pd.read_parquet(FEATURES_PATH)
feat["label"] = feat["is_viral"].map({1: VIRAL_LABEL, 0: NONVIRAL_LABEL})

viral    = feat[feat["is_viral"] == 1]
nonviral = feat[feat["is_viral"] == 0]

print(f"  Rows:      {len(feat):,}")
print(f"  Viral:     {len(viral):,}  ({len(viral)/len(feat)*100:.1f}%)")
print(f"  Non-viral: {len(nonviral):,}  ({len(nonviral)/len(feat)*100:.1f}%)")

# Detect available time windows dynamically
def detect_windows(df, prefix="repost"):
    cols = [c for c in df.columns if re.match(rf"{prefix}_(\d+)m$", c)]
    return sorted(int(re.search(r"(\d+)m$", c).group(1)) for c in cols)

WINDOWS     = detect_windows(feat, "repost")
W_FIRST     = WINDOWS[0]
W_MID       = WINDOWS[len(WINDOWS) // 2]
W_LAST      = WINDOWS[-1]
EVENT_TYPES = ["repost", "like", "reply", "quote"]

print(f"\n  Detected windows : {WINDOWS}")
print(f"  First={W_FIRST}m  |  Mid={W_MID}m  |  Full={W_LAST}m\n")


# =============================================================================
# HELPERS
# =============================================================================

def cap_at_percentile(s: pd.Series, pct: float = 99) -> pd.Series:
    return s.clip(upper=s.quantile(pct / 100))


def sum_engagement(df, window):
    cols = [f"{e}_{window}m" for e in EVENT_TYPES
            if f"{e}_{window}m" in df.columns]
    return df[cols].sum(axis=1)


def save(fig, name: str):
    path = SAVE_DIR / name
    fig.savefig(path)
    print(f"  Saved -> {path}")
    plt.close(fig)


# =============================================================================
# PLOT 1 — Scatter: early vs sustained engagement
#
# ROOT CAUSE OF PREVIOUS BUG:
#   ~90% of non-viral posts have ZERO early engagement so they all collapse
#   onto (0,0) and are invisible on a linear axis.
#
# FIX: Show linear AND log(1+x) side-by-side.
#   - Linear shows the scale of viral vs non-viral difference
#   - Log reveals the non-viral cluster and its shape
# =============================================================================
print("Plot 1: Scatter — early vs sustained engagement ...")

feat["eng_first"] = sum_engagement(feat, W_FIRST)
feat["eng_last"]  = sum_engagement(feat, W_LAST)

v_mask  = feat["is_viral"] == 1
nv_mask = feat["is_viral"] == 0

v_active  = (feat.loc[v_mask,  "eng_first"] > 0).mean() * 100
nv_active = (feat.loc[nv_mask, "eng_first"] > 0).mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, use_log in zip(axes, [False, True]):
    if use_log:
        x      = np.log1p(feat["eng_first"])
        y      = np.log1p(feat["eng_last"])
        xlabel = f"log(1 + engagement in first {W_FIRST}m)"
        ylabel = f"log(1 + engagement in first {W_LAST}m)"
        title  = "Log Scale — reveals non-viral cluster"
    else:
        x      = cap_at_percentile(feat["eng_first"], 99)
        y      = cap_at_percentile(feat["eng_last"],  99)
        xlabel = f"Total engagement in first {W_FIRST}m (capped 99th pct)"
        ylabel = f"Total engagement in first {W_LAST}m (capped 99th pct)"
        title  = "Linear Scale — shows magnitude of difference"

    # Draw non-viral first (background), viral on top
    ax.scatter(x[nv_mask], y[nv_mask],
               c=NONVIRAL_COLOR, alpha=0.15, s=7, linewidths=0,
               label=f"{NONVIRAL_LABEL} (n={nv_mask.sum():,})",
               rasterized=True)
    ax.scatter(x[v_mask],  y[v_mask],
               c=VIRAL_COLOR,    alpha=0.45, s=13, linewidths=0,
               label=f"{VIRAL_LABEL} (n={v_mask.sum():,})",
               rasterized=True)

    lim = min(x.max(), y.max())
    ax.plot([0, lim], [0, lim], color="grey", lw=1,
            linestyle="--", alpha=0.5,
            label=f"{W_FIRST}m = {W_LAST}m (no growth)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.97, 0.04,
            f"Viral with ≥1 event at {W_FIRST}m:     {v_active:.0f}%\n"
            f"Non-viral with ≥1 event at {W_FIRST}m: {nv_active:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="grey")

fig.suptitle(f"Early ({W_FIRST}m) vs Sustained ({W_LAST}m) Engagement",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, f"01_scatter_engagement_{W_FIRST}m_vs_{W_LAST}m.png")


# =============================================================================
# PLOT 2 — Box: early repost count (raw + log) by viral / non-viral
# =============================================================================
print("Plot 2: Box — early repost count vs virality ...")

col = f"repost_{W_LAST}m"
if col not in feat.columns:
    print(f"  Column '{col}' not found — skipping")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw scale (capped at 99th pct)
    plot_raw = feat[["label", col]].copy()
    plot_raw[col] = cap_at_percentile(plot_raw[col], 99)

    sns.boxplot(data=plot_raw, x="label", y=col,
                order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                flierprops=dict(marker="o", markersize=2.5, alpha=0.2),
                ax=axes[0])

    v_med  = feat[feat["is_viral"] == 1][col].median()
    nv_med = feat[feat["is_viral"] == 0][col].median()

    for i, (grp, med) in enumerate(zip(ORDER, [v_med, nv_med])):
        axes[0].text(i, med, f"  {med:.0f}", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if med > 3 else "grey")

    axes[0].set_xlabel("")
    axes[0].set_ylabel(f"Reposts in first {W_LAST}m (capped 99th pct)")
    axes[0].set_title("Raw Scale")
    axes[0].text(0.97, 0.97,
                 f"Viral median: {v_med:.0f}  |  Non-viral: {nv_med:.0f}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=9, color="grey")

    # Right: log(1+x) scale — shows non-viral distribution
    log_col = "_tmp_log"
    feat[log_col] = np.log1p(feat[col])
    sns.boxplot(data=feat[["label", log_col]], x="label", y=log_col,
                order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                flierprops=dict(marker="o", markersize=2.5, alpha=0.2),
                ax=axes[1])
    feat.drop(columns=[log_col], inplace=True)

    n_zero_nv = ((feat["is_viral"] == 0) & (feat[col] == 0)).sum()
    pct_zero  = n_zero_nv / (feat["is_viral"] == 0).sum() * 100

    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"log(1 + reposts in {W_LAST}m)")
    axes[1].set_title("Log Scale — shows non-viral distribution")
    axes[1].text(0.97, 0.04,
                 f"{pct_zero:.0f}% of non-viral posts have 0 reposts\nin first {W_LAST}m",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=9, color=NONVIRAL_COLOR)

    fig.suptitle(f"Early Repost Activity by Virality (first {W_LAST}m)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_box_early_repost_engagement.png")


# =============================================================================
# PLOT 3 — Box: velocity ratio by viral / non-viral
# Only includes posts with ≥1 event (velocity = 0 for zero-engagement posts
# is uninformative and would dominate the non-viral box).
# =============================================================================
print("Plot 3: Box — velocity ratio vs virality ...")

vel_cols = {
    f"Repost velocity\n({W_MID}→{W_LAST}m / 0→{W_MID}m)": "repost_velocity_ratio",
    f"Like velocity\n({W_MID}→{W_LAST}m / 0→{W_MID}m)":   "like_velocity_ratio",
}
available_vel = {k: v for k, v in vel_cols.items() if v in feat.columns}

if not available_vel:
    print("  No velocity columns found — skipping")
else:
    fig, axes = plt.subplots(1, len(available_vel),
                             figsize=(5.5 * len(available_vel), 5),
                             sharey=False)
    if len(available_vel) == 1:
        axes = [axes]

    for ax, (title, col) in zip(axes, available_vel.items()):
        # Only posts that had at least 1 event — velocity is 0 otherwise
        active = feat[feat[col] > 0][["label", col]].copy()
        active[col] = cap_at_percentile(active[col], 99)

        sns.boxplot(data=active, x="label", y=col,
                    order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.2),
                    ax=ax)

        ax.axhline(1.0, color="grey", lw=1.2, linestyle="--", alpha=0.7,
                   label="Constant rate (ratio = 1)")
        ax.set_xlabel("")
        ax.set_ylabel(f"Velocity ratio\n({W_MID}→{W_LAST}m) / (0→{W_MID}m + 1)")
        ax.set_title(title)
        ax.legend(fontsize=8)

        for i, grp in enumerate(ORDER):
            n_active = (active["label"] == grp).sum()
            n_total  = (feat["label"] == grp).sum()
            ax.annotate(
                f"n={n_active:,} of {n_total:,}\nhad events",
                xy=(i, 0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -46), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="grey"
            )

    fig.suptitle("Engagement Velocity: Viral vs Non-viral\n"
                 "(only posts with ≥1 event shown — zeros excluded)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    save(fig, "03_box_velocity_ratio.png")


# =============================================================================
# PLOT 4 — Box: time to first event by viral / non-viral
# Sentinel (9999) rows excluded — these are posts that never received
# that event type within the window. Coverage shown as annotation.
# =============================================================================
print("Plot 4: Box — time to first event vs virality ...")

ttf_cols = {
    "Time to first\nrepost (s)": "ttf_repost_sec",
    "Time to first\nlike (s)":   "ttf_like_sec",
    "Time to first\nreply (s)":  "ttf_reply_sec",
    "Time to first\nquote (s)":  "ttf_quote_sec",
}
available_ttf = {k: v for k, v in ttf_cols.items() if v in feat.columns}

if not available_ttf:
    print("  No TTF columns found — skipping")
else:
    fig, axes = plt.subplots(1, len(available_ttf),
                             figsize=(4.5 * len(available_ttf), 6),
                             sharey=False)
    if len(available_ttf) == 1:
        axes = [axes]

    for ax, (title, col) in zip(axes, available_ttf.items()):
        real = feat[feat[col] < SENTINEL][["label", col]].copy()
        real[col] = cap_at_percentile(real[col], 99)

        if real.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(data=real, x="label", y=col,
                    order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.15),
                    ax=ax)

        ax.set_xlabel("")
        ax.set_ylabel("Seconds from post creation")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x)}s")
        )

        # Coverage annotations — placed inside axis using transform
        for i, grp in enumerate(ORDER):
            n_total = (feat["label"] == grp).sum()
            n_real  = ((feat["label"] == grp) & (feat[col] < SENTINEL)).sum()
            pct     = n_real / n_total * 100
            ax.annotate(
                f"n={n_real:,}\n({pct:.0f}% received event)",
                xy=(i, 0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -50), textcoords="offset points",
                ha="center", va="top", fontsize=8, color="grey"
            )

        # Speed comparison
        v_med  = real[real["label"] == VIRAL_LABEL][col].median()
        nv_med = real[real["label"] == NONVIRAL_LABEL][col].median()
        if v_med > 0 and nv_med > 0:
            ratio = nv_med / v_med
            ax.text(0.97, 0.97,
                    f"Non-viral {ratio:.1f}x slower",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=NONVIRAL_COLOR)

    fig.suptitle("Time to First Engagement: Viral vs Non-viral\n"
                 "(sentinel posts excluded — only posts that received this event)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    save(fig, "04_box_time_to_first_event.png")


# =============================================================================
# PLOT 5 — Box: total likes snapshot as author-quality proxy
# author_followers is all zeros until API fetch runs.
# total_likes is a valid interim proxy.
# =============================================================================
print("Plot 5: Box — total likes snapshot vs virality ...")

col = "total_likes"
if col not in feat.columns:
    print(f"  Column '{col}' not found — skipping")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_raw = feat[["label", col]].copy()
    plot_raw[col] = cap_at_percentile(plot_raw[col], 99)

    sns.boxplot(data=plot_raw, x="label", y=col,
                order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                flierprops=dict(marker="o", markersize=2.5, alpha=0.2),
                ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total likes at collection time (capped 99th pct)")
    axes[0].set_title("Raw Scale")
    axes[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    log_col = "_tmp_log"
    feat[log_col] = np.log1p(feat[col])
    sns.boxplot(data=feat[["label", log_col]], x="label", y=log_col,
                order=ORDER, palette=PALETTE, width=0.45, linewidth=1.2,
                flierprops=dict(marker="o", markersize=2.5, alpha=0.2),
                ax=axes[1])
    feat.drop(columns=[log_col], inplace=True)

    v_med  = feat[feat["is_viral"] == 1][col].median()
    nv_med = feat[feat["is_viral"] == 0][col].median()

    axes[1].set_xlabel("")
    axes[1].set_ylabel("log(1 + total likes)")
    axes[1].set_title("Log Scale")
    axes[1].text(0.97, 0.97,
                 f"Viral median:     {v_med:,.0f} likes\n"
                 f"Non-viral median: {nv_med:,.0f} likes",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=9, color="grey")

    fig.suptitle("Total Likes at Collection Time by Virality\n"
                 "(placeholder for author_followers — will be replaced after API fetch)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_box_total_likes.png")


# =============================================================================
# PLOT 6 — Line: mean engagement per time bucket (dynamic window)
# =============================================================================
print("Plot 6: Line — engagement rate over time ...")

try:
    sys.path.insert(0, str(ROOT))
    from datasets.bluesky_cascade.cascade_store import CascadeStore

    CASCADE_DIR = ROOT / "datasets" / "bluesky_cascade"
    store = CascadeStore(CASCADE_DIR)

    def make_buckets(window_min, n=5):
        step    = window_min / n
        buckets = [round(step * i) for i in range(1, n + 1)]
        buckets[-1] = window_min
        return buckets

    def bucket_means(events, uris, viral_mask_series, window_min, n=5):
        buckets  = make_buckets(window_min, n)
        v_uris   = set(uris[viral_mask_series])
        nv_uris  = set(uris[~viral_mask_series])
        rows, prev = [], 0
        for b_min in buckets:
            b_sec = b_min * 60
            mask  = events["time_delta_sec"].between(prev, b_sec, inclusive="right")
            chunk = events[mask]
            rows.append({
                "bucket":        b_min,
                "viral_mean":    chunk["root_post_uri"].isin(v_uris).sum()  / max(len(v_uris),  1),
                "nonviral_mean": chunk["root_post_uri"].isin(nv_uris).sum() / max(len(nv_uris), 1),
            })
            prev = b_sec
        return pd.DataFrame(rows)

    viral_mask_s = feat.set_index("uri")["is_viral"].astype(bool)
    uris_s       = pd.Series(feat["uri"].values, index=feat["uri"].values)

    viral_mask_aligned = feat["is_viral"].astype(bool)

    event_tables = {
        "Reposts": store._reposts,
        "Likes":   store._likes,
        "Replies": store._replies,
        "Quotes":  store._quotes,
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes      = axes.flatten()
    buckets   = make_buckets(WINDOW_MIN)
    x_labels  = [str(b) for b in buckets]

    for ax, (name, events) in zip(axes, event_tables.items()):
        if events.empty:
            ax.set_visible(False)
            continue

        df = bucket_means(events, feat["uri"], viral_mask_aligned, WINDOW_MIN)

        ax.plot(x_labels, df["viral_mean"],
                color=VIRAL_COLOR, lw=2.5, marker="o", ms=7,
                label=VIRAL_LABEL)
        ax.plot(x_labels, df["nonviral_mean"],
                color=NONVIRAL_COLOR, lw=2.5, marker="o", ms=7,
                linestyle="--", label=NONVIRAL_LABEL)
        ax.fill_between(x_labels,
                        df["viral_mean"], df["nonviral_mean"],
                        alpha=0.10, color=VIRAL_COLOR)

        ax.set_title(name)
        ax.set_xlabel(f"Minutes after posting")
        ax.set_ylabel(f"Mean events / post per {WINDOW_MIN // 5}m bucket")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Engagement Rate Over Time: Viral vs Non-viral\n"
                 f"Window = {WINDOW_MIN}m  |  {len(buckets)} buckets "
                 f"of {WINDOW_MIN // 5}m each",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, f"06_line_engagement_over_time_{WINDOW_MIN}m.png")

except Exception as e:
    print(f"  Skipping Plot 6 — CascadeStore error: {e}")


# =============================================================================
# DONE
# =============================================================================
print()
print("=" * 65)
print(f"All plots saved to: {SAVE_DIR}")
print("=" * 65)