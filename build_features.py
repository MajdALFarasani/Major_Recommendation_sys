# build_features.py
from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

PROJECT_ROOT        = Path(__file__).parent
ENRICHED_CSV        = PROJECT_ROOT / "data" / "student_profiles_with_features.csv"
VECTORIZER_OUT      = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
FEATURES_TARGET_OUT = PROJECT_ROOT / "data" / "features_and_target.joblib"
VECTORIZER_OUT.parent.mkdir(parents=True, exist_ok=True)
FEATURES_TARGET_OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(ENRICHED_CSV)
    print(f"Loaded {df.shape}")

    # 1) TF-IDF on story
    tfidf = TfidfVectorizer(max_features=5000)
    X_text = tfidf.fit_transform(df["Story_clean"])
    joblib.dump(tfidf, VECTORIZER_OUT)

    # 2) grab column groups
    cat_cols   = [c for c in df if c.startswith("cat__")]
    sex_cols   = [c for c in df if c.startswith("sex__")]
    qual_cols  = [c for c in df if c.startswith("qual_")]
    reg_cols   = [c for c in df if c.startswith("reg_")]
    state_cols = [c for c in df if c.startswith("state_")]
    misc_num   = ["academic_confidence", "hobby_count", "sentiment"]

    X_cat   = df[cat_cols].values
    X_sex   = df[sex_cols].values
    X_qual  = df[qual_cols].values
    X_reg   = df[reg_cols].values
    X_state = df[state_cols].values
    X_misc  = df[misc_num].values

    # 3) stack
    X = hstack([X_text, X_cat, X_sex, X_qual, X_reg, X_state, X_misc])

    # 4) labels
    y = df["Major"]

    joblib.dump((X, y), FEATURES_TARGET_OUT)
    print("Finished ->", FEATURES_TARGET_OUT)

if __name__ == "__main__":
    main()
