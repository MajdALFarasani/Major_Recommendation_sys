# train_model.py
from pathlib import Path
import joblib, json
import matplotlib.pyplot as plt  # ### NEW: For plotting
import seaborn as sns            # ### NEW: For nicer heatmaps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # ### NEW: Added confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
FT_IN        = PROJECT_ROOT / "data"   / "features_and_target.joblib"
MODEL_OUT    = PROJECT_ROOT / "models" / "major_recommender_rf.joblib"
LE_OUT       = PROJECT_ROOT / "models" / "label_encoder.joblib"
IMP_JSON     = PROJECT_ROOT / "models" / "rf_feature_importance.json"

# ### NEW: Path for saving the visualization
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
CM_PLOT_OUT  = FIGURES_DIR / "confusion_matrix.png"

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True) # Ensure reports folder exists

# ── Hyper-parameters ─────────────────────────────────────────────────────
N_ESTIMATORS = 500          
RANDOM_SEED  = 42

def main():

    # 1) Load sparse X and string y
    X, y = joblib.load(FT_IN)
    print(f"Loaded X shape={X.shape} | y len={len(y)}")

    # 2) Encode majors -> integers
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LE_OUT)
    print("LabelEncoder saved →", LE_OUT)

    # 3) Split 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=RANDOM_SEED, stratify=y_enc
    )

    # 4) RF with OOB score
    rf = RandomForestClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = None,
        min_samples_leaf = 1,
        oob_score        = True,
        class_weight     = "balanced_subsample",
        n_jobs           = -1,
        random_state     = RANDOM_SEED,
        verbose          = 0
    )

    print(f"Training RandomForest (n_estimators={N_ESTIMATORS}) …")
    rf.fit(X_tr, y_tr)
    print(f"   ↳ OOB accuracy: {rf.oob_score_:.3f}")

    # 5) Test performance
    preds = rf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"\nHold-out accuracy: {acc:.3f}\n")
    
    # Print the text report (This covers Bullet 1 of your requirements)
    print(classification_report(y_te, preds, target_names=le.classes_))

   # ─────────────────────────────────────────────────────────────────────────
    # ### NEW: 5.5) Generate Summary Visualization (Option B)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating Summary Confusion Matrix Plot...")
    
    cm = confusion_matrix(y_te, preds)
    
    # Use a standard, manageable size (no need for huge dimensions now)
    plt.figure(figsize=(10, 8)) 
    
    # Create the heatmap without labels
    sns.heatmap(
        cm, 
        annot=False,          # No numbers inside boxes
        fmt='d', 
        cmap='Blues', 
        xticklabels=False,    # ### KEY CHANGE: Hide specific major names
        yticklabels=False     # ### KEY CHANGE: Hide specific major names
    )
    
    plt.xlabel('Predicted Majors', fontsize=12)
    plt.ylabel('Actual Majors', fontsize=12)
    plt.title(f'Confusion Matrix Heatmap\n(Accuracy: {acc:.3f} - Labels hidden for clarity)', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(CM_PLOT_OUT)
    print(f"Visualization saved → {CM_PLOT_OUT}")
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    # 6) Save model
    joblib.dump(rf, MODEL_OUT)
    print("Model saved →", MODEL_OUT)

    # 7) Save top-30 feature importances (optional)
    importances = rf.feature_importances_
    top_idx     = importances.argsort()[::-1][:30]
    imp_record  = {int(i): float(importances[i]) for i in top_idx}
    IMP_JSON.write_text(json.dumps(imp_record, indent=2))
    print(f"Top-30 importances written → {IMP_JSON}")

if __name__ == "__main__":
    main()