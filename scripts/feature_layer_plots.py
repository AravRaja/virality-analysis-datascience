import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────
df = pd.read_parquet("../data/training/features_60m.parquet")
TARGET = "is_viral_speed_S"
sns.set_style("whitegrid")
VIRAL_COLORS = {0: "#4A90D9", 1: "#E74C3C"}
VIRAL_LABELS = {0: "Non-Viral", 1: "Viral"}
OUTPUT_DIR = "../figures/evaluation"  # adjust as needed

# =====================================================================
# LAYER 1: Overall Engagement Curves — Viral vs Non-Viral
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

windows = [12, 24, 36, 48, 60]
window_labels = ["12", "24", "36", "48", "60"]
types = {
    "Reposts": "repost",
    "Likes": "like",
    "Replies": "reply",
    "Quotes": "quote",
}

# Calculate overall engagement for each window
for viral_val, color in VIRAL_COLORS.items():
    viral_subset = df[df[TARGET] == viral_val]
    means = []
    for w in windows:
        window_cols = [f"{prefix}_{w}m" for prefix in types.values()]
        window_mean = viral_subset[window_cols].sum(axis=1).mean()
        means.append(window_mean)
    
    ax.plot(
        window_labels,
        means,
        marker="o",
        color=color,
        linewidth=2.5,
        label=VIRAL_LABELS[viral_val],
    )

ax.set_title("Mean Total Engagement Over First 60 Minutes\n(Viral vs Non-Viral)", fontsize=13, fontweight="bold")
ax.set_xlabel("Minutes After Posting")
ax.set_ylabel("Mean Total Count")
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/L1_engagement_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================================
# LAYER 2: Whale Count vs Viral Rate
# =====================================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
fig.suptitle(
    "Viral Rate by Number of Whale Engagers (≥1,000 Followers)",
    fontsize=14,
    fontweight="bold",
)

for ax, (title, prefix) in zip(axes, types.items()):
    col = f"{prefix}_whale_count"
    binned = df[col].clip(upper=3).astype(int)
    binned = binned.replace(3, "3+").astype(str)
    order = ["0", "1", "2", "3+"]

    tmp = pd.DataFrame({"whale_bin": binned, TARGET: df[TARGET]})
    rates = (
        tmp.groupby("whale_bin")[TARGET]
        .mean()
        .reindex(order)
        .fillna(0)
        * 100
    )
    counts = tmp.groupby("whale_bin").size().reindex(order).fillna(0)

    bars = ax.bar(
        rates.index,
        rates.values,
        color=["#4A90D9", "#F5A623", "#E8704A", "#E74C3C"],
        edgecolor="white",
    )
    for bar, n in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={int(n):,}",
            ha="center",
            fontsize=8,
            color="grey",
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Whale Engager Count")
    if ax == axes[0]:
        ax.set_ylabel("Viral Rate (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/L2_whale_vs_viral_rate_is_viral_speed_S.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================================
# LAYER 3: Author Follower Count vs Viral Rate (Binned)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

bins = [0, 10, 100, 1_000, 10_000, np.inf]
labels = ["1–10", "10–100", "100–1k", "1k–10k", "10k+"]
df["_follower_bin"] = pd.cut(
    df["author_followers"].clip(lower=1),
    bins=bins,
    labels=labels,
    right=True,
)

rates = df.groupby("_follower_bin", observed=False)[TARGET].mean() * 100
counts = df.groupby("_follower_bin", observed=False).size()

bars = ax.bar(
    labels,
    rates.values,
    color=["#4A90D9", "#5DADE2", "#F5A623", "#E8704A", "#E74C3C"],
    edgecolor="white",
)
for bar, n in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"n={int(n):,}",
        ha="center",
        fontsize=9,
        color="grey",
    )

ax.set_title(
    "Viral Rate by Author Follower Count",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Author Follower Bin")
ax.set_ylabel("Viral Rate (%)")

df.drop(columns="_follower_bin", inplace=True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/L3_author_followers_viral_rate_is_viral_speed_S.png", dpi=300, bbox_inches="tight")
plt.show()