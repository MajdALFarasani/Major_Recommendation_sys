# feature_engineering.py
import sys, string, numpy as np
import pandas as pd
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer

def main():
    PROJECT_ROOT = Path(__file__).parent
    raw_path   = PROJECT_ROOT / "data" / "student_profiles.jsonl"
    output_csv = PROJECT_ROOT / "data" / "student_profiles_with_features.csv"

    print("1) Loading raw JSONL from:", raw_path)
    df = pd.read_json(raw_path, lines=True)
    print(f"   → {len(df)} records loaded\n")

    # ── Clean story text ────────────────────────────────────────────────────
    df["Story_clean"] = (
        df["Story"].str.lower()
        .str.translate(str.maketrans("", "", string.punctuation))
        .str.strip()
    )

    # ── academic_confidence from GPA ───────────────────────────────────────
    def map_conf(g):
        return 5 if g>=3.6 else 4 if g>=3.2 else 3 if g>=2.8 else 2 if g>=2.0 else 1
    df["academic_confidence"] = df["GPA"].apply(map_conf)
    df = df.drop(columns=["GPA", "Age"], errors="ignore")

    # ── Normalise hobbies list ─────────────────────────────────────────────
    df["Hobbies_list"] = df["Hobbies"].apply(lambda lst: [h.strip().lower() for h in lst])

    # hobby-category flags
    ROOT_PARENT = PROJECT_ROOT.parent.resolve()
    sys.path.insert(0, str(ROOT_PARENT))
    from category_map import CATEGORY_MAP
    for cat, hset in CATEGORY_MAP.items():
        df[f"cat__{cat}"] = df["Hobbies_list"].apply(lambda lst: int(bool(set(lst) & hset)))

    # sex flags
    df["sex__male"]   = (df["Sex"].str.lower()=="male").astype(int)
    df["sex__female"] = (df["Sex"].str.lower()=="female").astype(int)

    # Unique-Quality one-hots
    qual_dummies = pd.get_dummies(df["Unique Quality"].fillna(""), prefix="qual")
    df = pd.concat([df, qual_dummies], axis=1)

    # ─────────── NEW FEATURES ──────────────────────────────────────────────
    ## Country & State dummies
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["Country"].fillna(""),           prefix="reg"),
            pd.get_dummies(df["State/Province"].fillna(""),    prefix="state")
        ],
        axis=1
    )

    ## Hobby count
    df["hobby_count"] = df["Hobbies_list"].str.len()

    ## Sentiment score (VADER)
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["Story_clean"].apply(lambda t: sia.polarity_scores(t)["compound"])

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    print("✅ Saved enriched CSV →", output_csv)

if __name__ == "__main__":
    # ensure VADER lexicon is present once
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon")
    main()
