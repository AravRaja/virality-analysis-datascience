"""
emotion_features.py

Runs the fine-tuned RoBERTa emotion classifier over root post text and
appends emotion probability columns to training_features.parquet.

Model:  models/emotion_classifier/
Labels: joy | moral_outrage | neutral | sadness | anger

Inputs:
    datasets/bluesky_cascade/training_features.parquet  (must exist)
    models/emotion_classifier/                          (model weights)

Output:
    datasets/bluesky_cascade/training_features.parquet  (overwritten in place)
    datasets/bluesky_cascade/training_features.csv      (optional)

New columns added:
    emotion_joy           – P(joy)
    emotion_moral_outrage – P(moral_outrage)
    emotion_neutral       – P(neutral)
    emotion_sadness       – P(sadness)
    emotion_anger         – P(anger)
    emotion_label         – argmax label (string)

Run:
    python scripts/emotion_features.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT       = Path(__file__).resolve().parent.parent
HF_MODEL   = "bobloker/emotion-classifier-roberta"
DATA_DIR   = ROOT / "datasets" / "bluesky_cascade"
FEATURES_PATH = DATA_DIR / "training_features.parquet"
SAVE_CSV   = True
BATCH_SIZE = 64
MAX_LENGTH = 128


def load_model(hf_model: str):
    print(f"  Loading tokenizer from: {hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    print(f"  Loading model from:     {hf_model}")
    model = AutoModelForSequenceClassification.from_pretrained(hf_model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")
    return tokenizer, model, device


def predict_emotions(texts: list[str],
                     tokenizer,
                     model,
                     device,
                     batch_size: int = BATCH_SIZE,
                     max_length: int = MAX_LENGTH) -> np.ndarray:
    """
    Returns an (N, num_labels) array of softmax probabilities.
    Empty / null texts are replaced with an empty string.
    """
    texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

        done = min(i + batch_size, len(texts))
        print(f"\r  Progress: {done:>6,} / {len(texts):,}", end="", flush=True)

    print()
    return np.vstack(all_probs)


def main():
    print("=" * 70)
    print("EMOTION FEATURES — RoBERTa classifier")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURES_PATH}\n"
            f"Run feature_extraction.py first."
        )

    print(f"\n  Loading feature table: {FEATURES_PATH}")
    feat = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(feat):,} rows × {len(feat.columns)} columns")

    # Retrieve post text from the cascade store
    root_posts_path = DATA_DIR / "root_posts.parquet"
    if not root_posts_path.exists():
        raise FileNotFoundError(f"root_posts.parquet not found: {root_posts_path}")

    posts = pd.read_parquet(root_posts_path, columns=["uri", "text"])
    feat  = feat.merge(posts, on="uri", how="left")
    texts = feat["text"].tolist()
    print(f"  Text column: {feat['text'].notna().sum():,} non-null / {len(feat):,} total")

    # Load model and run inference
    print()
    tokenizer, model, device = load_model(HF_MODEL)

    id2label = model.config.id2label  # {0: 'joy', 1: 'moral_outrage', ...}
    labels   = [id2label[i] for i in range(len(id2label))]

    print(f"\n  Running inference on {len(texts):,} posts (batch={BATCH_SIZE}) ...")
    probs = predict_emotions(texts, tokenizer, model, device)

    # Attach probability columns
    for i, label in enumerate(labels):
        feat[f"emotion_{label}"] = probs[:, i].astype(np.float32)

    feat["emotion_label"] = [labels[i] for i in probs.argmax(axis=1)]

    # Drop the merged text column (it's already in root_posts.parquet)
    feat.drop(columns=["text"], inplace=True)

    # Summary
    print("\n  Emotion distribution (argmax):")
    dist = feat["emotion_label"].value_counts()
    for lbl, cnt in dist.items():
        print(f"    {lbl:<20s}: {cnt:>6,}  ({cnt/len(feat)*100:.1f}%)")

    print("\n  Mean probabilities:")
    for label in labels:
        col = f"emotion_{label}"
        print(f"    {col:<30s}: {feat[col].mean():.4f}")

    # Save
    print(f"\n  Saving: {FEATURES_PATH}")
    feat.to_parquet(FEATURES_PATH, index=False)
    print(f"     {len(feat):,} rows × {len(feat.columns)} columns")

    if SAVE_CSV:
        csv_path = DATA_DIR / "training_features.csv"
        feat.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    emotion_cols = [f"emotion_{l}" for l in labels] + ["emotion_label"]
    print(f"\n  New emotion columns: {emotion_cols}")

    print("\n" + "=" * 70)
    print("DONE — emotion features appended to training_features.parquet")
    print("=" * 70)
    return feat


if __name__ == "__main__":
    feat = main()
