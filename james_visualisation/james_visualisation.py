import pandas as pd
from pathlib import Path
root_posts = pd.read_parquet(Path(__file__).parent / "root_posts_subset.parquet")
root_posts["viral1"]=root_posts["viral_TI_S_score"]>3
root_posts["viral2"]=root_posts["viral_speed_S_score"]>89
import matplotlib.pyplot as plt
label_order = ["joy", "moral_outrage", "neutral", "sadness", "anger"]

for viral_col, label in [("viral1", "TI_S > 3"), ("viral2", "speed_S > 89")]:
    tier_means = root_posts.groupby(viral_col)[[f"emotion_{e}" for e in label_order]].mean()
    tier_means.index = ["Non-viral", "Viral"]
    tier_means.columns = label_order

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(2)
    width = 0.15

    for i, emotion in enumerate(label_order):
        ax.bar([j + i * width for j in x], tier_means[emotion], width, label=emotion, alpha=0.85)

    ax.set_xticks([j + width * 2 for j in x])
    ax.set_xticklabels(["Non-viral", "Viral"], fontsize=12)
    ax.set_xlabel("Virality", fontsize=12)
    ax.set_ylabel("Mean emotion probability", fontsize=12)
    ax.set_title(f"Emotion probabilities — {label}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
