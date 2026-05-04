# train_explainer_logreg.py
# -------------------------------------------------------------------
# Train a lightweight Logistic Regression model for explanations only

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_PATH    = PROJECT_ROOT / "data" / "student_profiles_with_features.csv"
VEC_PATH     = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_OUT    = PROJECT_ROOT / "models" / "major_explainer_logreg.joblib"
LE_OUT       = PROJECT_ROOT / "models" / "label_encoder.joblib"

def main():
    # 1) Load enriched dataset
    df = pd.read_csv(DATA_PATH)
    texts = df["Story_clean"].fillna("")
    y = df["Major"]

    # 2) Encode majors
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LE_OUT)

    # 3) Load TF‑IDF vectorizer (same as RF uses)
    tfidf = joblib.load(VEC_PATH)
    X = tfidf.transform(texts)

    # 4) Train Logistic Regression
    logreg = LogisticRegression(max_iter=2000, multi_class="ovr", n_jobs=-1)
    print("Training Logistic Regression explainer …")
    logreg.fit(X, y_enc)

    # 5) Save model
    joblib.dump(logreg, MODEL_OUT)
    print("✅ Logistic explainer saved →", MODEL_OUT)

if __name__ == "__main__":
    main()
