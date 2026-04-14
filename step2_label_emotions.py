"""
Step 2 — Emotion labeling via Gemini Flash 2.0 (7-class Ekman)
===============================================================
Usage (in terminal):
    pip install google-genai pandas tqdm
    export GEMINI_API_KEY=your-key-here

Usage (in Jupyter — run this cell last):
    await main()

Get your free API key at: https://aistudio.google.com/apikey

Input:  label_sample.csv   (from Step 1, must have a "text" column)
Output: label_sample_emotions.csv  (same rows + "emotion" + "emotion_confidence" columns)
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm_asyncio

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = "label_anger_boost3.csv"
OUTPUT_CSV  = "label_anger_boost3_emotions.csv"
MODEL = "gemini-2.5-flash"
CONCURRENCY = 10           # Gemini Flash has generous rate limits
RETRY_LIMIT = 3
RETRY_DELAY = 2.0

EMOTIONS = {"joy", "anger", "sadness", "moral_outrage", "neutral"}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """You are an expert emotion analyst for social media content.

Classify the Bluesky post into EXACTLY ONE of these 5 emotions:
  joy | anger | sadness | moral_outrage | neutral

Definitions:
  joy           — happiness, delight, excitement, pride, warmth, humour, love, amusement, gratitude
  anger         — frustration, irritation, or annoyance directed at a PERSONAL inconvenience or specific individual; the target is YOU or someone who wronged YOU directly
  sadness       — grief, loss, disappointment, melancholy, longing, hopelessness about one's own situation
  moral_outrage — anger or disgust directed at a SYSTEMIC injustice, institutional failure, wrongdoing affecting others, or a violation of shared values; the target is a system, policy, group, or abstract wrong
  neutral       — strictly factual reporting with zero emotional charge; the author has no stake or feeling in what they are describing

Critical distinctions:
  anger vs moral_outrage — Ask: is the author upset on their OWN behalf (anger) or on behalf of OTHERS / society (moral_outrage)?
    "My landlord is impossible" → anger (personal grievance)
    "Landlords are destroying housing for everyone" → moral_outrage (systemic)
  sadness vs neutral — Sadness requires personal emotional investment. A news headline with no authorial feeling = neutral.
  joy vs neutral — Dry humour or mild satisfaction without enthusiasm = neutral. Clear delight or excitement = joy.

Rules:
- Pick the DOMINANT emotion even if multiple are present.
- neutral ONLY when there is genuinely zero emotional charge. If in doubt, pick the closest emotion instead.
- Do NOT default to neutral for ambiguous posts. Commit to the closest emotion.
- Emojis are strong signals — weight them accordingly.
- Return ONLY a JSON object with two keys:
    {"emotion": "<label>", "confidence": <0.0–1.0>}
  No explanation. No markdown. No extra text."""

# 14 diverse few-shot examples — 2-3 per class, including hard borderline cases
FEW_SHOTS = [
    # ── joy ──────────────────────────────────────────────────────────────────
    (
        "Just got the job offer!! I've been waiting 3 months for this 😭🎉",
        {"emotion": "joy", "confidence": 0.97}
    ),
    (
        "Can't believe how good this turned out AND how cheap the ingredients were 😂 recipe in thread",
        {"emotion": "joy", "confidence": 0.90}
    ),
    (
        "My daughter took her first steps today. I ugly cried in the kitchen for ten minutes.",
        {"emotion": "joy", "confidence": 0.93}
    ),
    # ── sadness ───────────────────────────────────────────────────────────────
    (
        "Miss my dog so much. Two years since she passed and I still reach for her lead by the door.",
        {"emotion": "sadness", "confidence": 0.96}
    ),
    (
        "Got the rejection email this morning. Third time applying. Starting to think this career isn't for me.",
        {"emotion": "sadness", "confidence": 0.92}
    ),
    (
        "Town centre where I grew up is all payday loan shops and empty units now. Just feels like loss.",
        {"emotion": "sadness", "confidence": 0.88}
    ),
    # ── anger (personal) ──────────────────────────────────────────────────────
    (
        "They're raising tube fares AGAIN while service gets worse every year. Absolutely done.",
        {"emotion": "anger", "confidence": 0.91}
    ),
    (
        "My neighbour has had a leaf blower going for 45 minutes. I am going to lose my mind.",
        {"emotion": "anger", "confidence": 0.95}
    ),
    # ── moral_outrage (systemic) ──────────────────────────────────────────────
    (
        "This is what happens when corporations are allowed to write their own regulations. People died. Nobody will go to prison.",
        {"emotion": "moral_outrage", "confidence": 0.96}
    ),
    (
        "The landlord raised rent 40% and the council did nothing. The system is designed to protect them, not us.",
        {"emotion": "moral_outrage", "confidence": 0.93}
    ),
    (
        "A child died waiting 14 hours in A&E. The Health Secretary went on holiday. This is not normal and we should not accept it.",
        {"emotion": "moral_outrage", "confidence": 0.97}
    ),
    # ── neutral ───────────────────────────────────────────────────────────────
    (
        "The UK base rate was held at 4.5% today by the Bank of England.",
        {"emotion": "neutral", "confidence": 0.97}
    ),
    (
        "Parliament votes on the amended bill tomorrow at 14:00. Live coverage on BBC Two.",
        {"emotion": "neutral", "confidence": 0.96}
    ),
    # ── hard borderline: anger vs moral_outrage ───────────────────────────────
    (
        "My GP surgery has a 3-week wait for a routine appointment. I just need to be seen.",
        {"emotion": "anger", "confidence": 0.82}
    ),
]


def build_contents(post_text: str) -> list:
    """Build Gemini contents list with few-shot examples then the target post."""
    contents = []
    for text, label in FEW_SHOTS:
        contents.append(types.Content(role="user",  parts=[types.Part(text=f'Post: "{text}"')]))
        contents.append(types.Content(role="model", parts=[types.Part(text=json.dumps(label))]))
    contents.append(types.Content(role="user", parts=[types.Part(text=f'Post: "{post_text}"')]))
    return contents


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, float]:
    """Extract emotion + confidence, falling back gracefully on bad output."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        obj = json.loads(clean)
        emotion    = obj.get("emotion", "").lower().strip()
        confidence = float(obj.get("confidence", 0.0))
        if emotion in EMOTIONS:
            return emotion, confidence
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: find any emotion keyword in raw text
    for e in EMOTIONS:
        if e in raw.lower():
            return e, 0.5

    return "neutral", 0.0


# ── API call with retry ───────────────────────────────────────────────────────

async def label_post(
    client: genai.Client,
    sem: asyncio.Semaphore,
    idx: int,
    text: str,
) -> dict:
    """Call Gemini for a single post, retrying on transient errors."""
    async with sem:
        for attempt in range(RETRY_LIMIT):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=build_contents(text),
                    config=types.GenerateContentConfig(
    system_instruction=SYSTEM,
    temperature=0.1,
    max_output_tokens=60,
    thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable thinking
)
                )
                raw = response.text or ""
                emotion, confidence = parse_response(raw)
                return {"idx": idx, "emotion": emotion, "emotion_confidence": confidence}

            except Exception as e:
                err = str(e).lower()
                is_rate_limit = "429" in err or "resource_exhausted" in err or "quota" in err
                wait = RETRY_DELAY * (2 ** attempt) if is_rate_limit else RETRY_DELAY

                if attempt == RETRY_LIMIT - 1:
                    print(f"[WARN] idx={idx} failed after {RETRY_LIMIT} attempts: {e}")
                    return {"idx": idx, "emotion": "neutral", "emotion_confidence": 0.0}

                await asyncio.sleep(wait)

    return {"idx": idx, "emotion": "neutral", "emotion_confidence": 0.0}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    input_path  = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")

    print(f"Loaded {len(df):,} posts from {input_path}")

    # Resume support: skip rows already labeled
    if output_path.exists():
        done = pd.read_csv(output_path, usecols=["idx"])["idx"].tolist()
        remaining = df[~df.index.isin(done)]
        print(f"Resuming — {len(done):,} already labeled, {len(remaining):,} remaining")
    else:
        remaining = df

    if remaining.empty:
        print("All posts already labeled. Nothing to do.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY environment variable — get one free at https://aistudio.google.com/apikey")

    client = genai.Client(api_key=api_key)
    sem    = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        label_post(client, sem, idx, str(row["text"]))
        for idx, row in remaining.iterrows()
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Labeling")

    results_df = pd.DataFrame(results).set_index("idx")
    df["emotion"]            = results_df["emotion"]
    df["emotion_confidence"] = results_df["emotion_confidence"]

    df.to_csv(output_path, index=True, index_label="idx")
    print(f"\nDone. Saved {len(df):,} rows to {output_path}")

    # Class distribution summary
    print("\nEmotion distribution:")
    dist  = df["emotion"].value_counts()
    total = len(df)
    for emotion, count in dist.items():
        bar = "█" * int(count / total * 40)
        print(f"  {emotion:<10} {count:>6,}  {count/total*100:5.1f}%  {bar}")

    low_conf = (df["emotion_confidence"] < 0.7).sum()
    print(f"\nLow-confidence labels (<0.7): {low_conf:,} ({low_conf/total*100:.1f}%)")
    print("Consider reviewing these manually in Step 3.")


# ── Entry point ───────────────────────────────────────────────────────────────
# In Jupyter, use:  await main()
# In terminal, use: python step2_label_emotions.py

if __name__ == "__main__":
    asyncio.run(main())
