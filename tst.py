# app.py – Major Recommendation System  (duplicate-proof rev 2025-08-06)
# ---------------------------------------------------------------------------
# * Heavy artefacts cached
# * Multi-word + single-word keyword extraction
# * _canon() removes "a/an/the", collapses spaces → no duplicate badges
# * Editable statement on Step 3
# ---------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import sys, json, io, csv, re

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack

# ─── Paths & basic config ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from category_map import CATEGORY_MAP

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Major Recommendation", page_icon="🎓", layout="centered")

# simple dark-theme tweaks
st.markdown(
    """
    <style>
      :root {
        --primary-color:#4C6EF5;
        --background-color:#0F1117;
        --secondary-background-color:#1E2530;
        --text-color:#FFFFFF;
      }
      a { color:#93C5FD; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Heavy artefacts (cached once) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_heavy():
    import joblib, spacy, nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sentence_transformers import SentenceTransformer

    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    clf   = joblib.load(MODELS_DIR / "major_recommender_rf.joblib")
    le    = joblib.load(MODELS_DIR / "label_encoder.joblib")

    nlp   = spacy.load("en_core_web_sm")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()

    return tfidf, clf, le, nlp, sbert, sia

tfidf, clf, le, nlp, sbert_model, sia = load_heavy()

# ─── Static meta (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_static():
    hdr = pd.read_csv(DATA_DIR / "student_profiles_with_features.csv", nrows=0)

    cfile = DATA_DIR / "country_states.json"
    try:
        country_states = json.loads(cfile.read_text("utf-8"))
    except FileNotFoundError:
        raw = pd.read_json(DATA_DIR / "student_profiles.jsonl", lines=True)
        country_states = (
            raw.groupby("Country")["State/Province"]
               .apply(lambda s: sorted({str(x) for x in s}))
               .to_dict()
        )
        cfile.write_text(json.dumps(country_states, ensure_ascii=False), "utf-8")

    qual_cols  = [c for c in hdr if c.startswith("qual_")]
    reg_cols   = [c for c in hdr if c.startswith("reg_")]
    state_cols = [c for c in hdr if c.startswith("state_")]
    return country_states, qual_cols, reg_cols, state_cols

COUNTRY_STATES, QUAL_COLS, REG_COLS, STATE_COLS = load_static()

MAJOR_INFO = json.loads((DATA_DIR / "major_info.json").read_text("utf-8"))
MAJORS     = list(MAJOR_INFO)

@st.cache_resource(show_spinner=False)
def precompute_major_emb():
    return {m: sbert_model.encode(m, convert_to_tensor=True) for m in MAJORS}

EMB_MAJOR = precompute_major_emb()

# ─── Keyword helpers ────────────────────────────────────────────────────────
_GENERIC = {"thing","stuff","time","life","activity","project","work",
            "class","subject","study","homework","task","daily"}

def _canon(txt: str) -> str:
    """Normalise for dedup: strip leading a/an/the, lower, collapse spaces."""
    txt = txt.strip().lower()
    txt = re.sub(r"^(a|an|the)\s+", "", txt)
    return re.sub(r"\s+", " ", txt)

@st.cache_data(show_spinner=False)
def extract_phrases(story: str) -> list[str]:
    """Return multi-word phrases & salient single tokens, deduplicated."""
    doc = nlp(story)
    phrases: set[str] = set()

    # noun-chunks
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip().lower()
        if len(txt.split()) > 1 and txt not in _GENERIC:
            phrases.add(txt)

    # liking verbs
    prefer = {"love","enjoy","like","prefer","adore"}
    for i, tok in enumerate(doc[:-1]):
        if tok.lemma_.lower() in prefer and tok.pos_ == "VERB":
            nxt = doc[i+1]
            if nxt.pos_ in {"NOUN","PROPN","VERB"}:
                phrases.add(f"{tok.lemma_.lower()} {nxt.lemma_.lower()}")

    # single nouns / PROPN ≥4 letters
    for tok in doc:
        if tok.is_stop or tok.is_punct:     continue
        if tok.lemma_.lower() in _GENERIC:  continue
        if tok.pos_ in {"PROPN","NOUN"} and len(tok.text) >= 4:
            phrases.add(tok.lemma_.lower())

    # final dedup via canonical form
    canon_to_orig = {}
    for p in phrases:
        c = _canon(p)
        if c not in canon_to_orig or len(p) < len(canon_to_orig[c]):
            canon_to_orig[c] = p

    return list(canon_to_orig.values())

def explain_keywords(story, majors, topn=3):
    from sentence_transformers import util
    phr = extract_phrases(story)
    if not phr: return {m: [] for m in majors}
    emb_phr = sbert_model.encode(phr, convert_to_tensor=True)
    out={}
    for m in majors:
        sims = util.cos_sim(EMB_MAJOR[m], emb_phr)[0].cpu().tolist()
        ranked = sorted(zip(phr, sims), key=lambda x: x[1], reverse=True)
        out[m] = [p for p,_ in ranked[:topn]]
    return out

# ─── Misc helpers & UI bits ─────────────────────────────────────────────────
def _rerun():
    if hasattr(st, "experimental_rerun"): st.experimental_rerun()
    elif hasattr(st, "rerun"):             st.rerun()
    else:                                  st.stop()

def major_card(major, pct, blurb, url, kw):
    badge = " ".join(
        f"<span style='background:#4C6EF5;border-radius:8px;padding:2px 6px;font-size:0.8em;margin-right:4px;color:#F8FAFF;'>{w}</span>"
        for w in kw
    )
    st.markdown(
        f"""
        <div style='border:1px solid #4C6EF5;background:#1E2530;color:#F8FAFF;
                    border-radius:12px;padding:1rem;margin-bottom:1rem;'>
          <h4 style='margin:0 0 .4em 0;color:#FFFFFF;'>{major}
              <span style='font-size:.8em;font-weight:400;color:#C4C9D2;'> ({pct:.1f}% match)</span></h4>
          <p style='margin:.3em 0 .6em 0;line-height:1.5em;'>{blurb}</p>
          {badge}<br>
          <a href='{url}' target='_blank' style='text-decoration:none;font-weight:600;color:#93C5FD;'>Learn more →</a>
        </div>
        """, unsafe_allow_html=True)

def csv_bytes(top3, probs, expl):
    denom = probs.sum()
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["Major","Match %","Keywords","Link"])
    for m,p in zip(top3, probs):
        w.writerow([m, f"{(p/denom)*100:.1f}", ", ".join(expl[m]), MAJOR_INFO[m]["url"]])
    return buf.getvalue().encode()

# ─── Wizard state ───────────────────────────────────────────────────────────
if "step" not in st.session_state: st.session_state["step"] = 1
st.title("🎓 Major Recommendation System")

# ---------------- STEP 1 ----------------------------------------------------
if st.session_state.step == 1:
    st.header("Step 1 — Tell us about yourself")
    conf = st.selectbox("Academic confidence", ["Very Low","Low","Medium","High","Very High"], index=2)
    CONF = {"Very Low":1,"Low":2,"Medium":3,"High":4,"Very High":5}
    sex = st.radio("Sex", ["Male","Female"], horizontal=True)
    country = st.selectbox("Country", sorted(COUNTRY_STATES))
    state   = st.selectbox("State / Province", COUNTRY_STATES.get(country, ["N/A"]))
    all_hobbies = sorted({h for lst in CATEGORY_MAP.values() for h in lst})
    hobbies = st.multiselect("Hobbies / Interests", all_hobbies)
    unique = st.selectbox("Unique quality", [c[5:] for c in QUAL_COLS])

    if st.button("Next →", type="primary"):
        st.session_state.profile = {
            "academic_conf": CONF[conf], "sex": sex,
            "country": country, "state": state,
            "hobbies": hobbies, "unique_qual": unique,
        }
        st.session_state.step = 2
        _rerun()

# ---------------- STEP 2 ----------------------------------------------------
elif st.session_state.step == 2:
    st.header("Step 2 — Paste your personal statement")
    story = st.text_area("Personal statement", height=220,
                         placeholder="Share your interests, achievements…")
    col1,col2 = st.columns(2)
    with col1:
        if st.button("← Back"): st.session_state.step=1; _rerun()
    with col2:
        if st.button("Get recommendations", type="primary") and story.strip():
            st.session_state.story=story.strip(); st.session_state.step=3; _rerun()

# ---------------- STEP 3 ----------------------------------------------------
else:
    profile = st.session_state.profile
    story   = st.session_state.story

    with st.expander("📝 Personal statement (edit & re-run)", expanded=True):
        new_story = st.text_area("", value=story, height=200, key="edit_story")
        if st.button("Update recommendations"):
            st.session_state.story = new_story.strip(); _rerun()

    with st.spinner("Crunching the numbers…"):
        vec = tfidf.transform([st.session_state.story.lower()])
        cat_f = [int(any(h in CATEGORY_MAP[k] for h in profile["hobbies"])) for k in CATEGORY_MAP]
        sex_f = [1,0] if profile["sex"]=="Male" else [0,1]
        reg_f = [int(profile["country"] == c[4:]) for c in REG_COLS]
        st_f  = [int(profile["state"]   == c[6:]) for c in STATE_COLS]
        qual_f= [int(profile["unique_qual"] == c[5:]) for c in QUAL_COLS]
        misc  = [profile["academic_conf"], len(profile["hobbies"]),
                 sia.polarity_scores(st.session_state.story)["compound"]]

        X = hstack([vec,
                    np.array([cat_f],dtype=np.int8),
                    np.array([sex_f],dtype=np.int8),
                    np.array([qual_f],dtype=np.int8),
                    np.array([reg_f],dtype=np.int8),
                    np.array([st_f], dtype=np.int8),
                    np.array([misc], dtype=np.float32)])
        probs = clf.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top3    = [le.inverse_transform([i])[0] for i in top_idx]
        keywords = explain_keywords(st.session_state.story, top3)

    st.header("Step 3 — Your personalised recommendations")
    for m, idx in zip(top3, top_idx):
        major_card(m, (probs[idx]/probs[top_idx].sum())*100,
                   MAJOR_INFO[m]["blurb"], MAJOR_INFO[m]["url"],
                   keywords.get(m, []))

    st.download_button("Download my recommendations (CSV)",
                       csv_bytes(top3, probs[top_idx], keywords),
                       file_name="my_majors.csv", mime="text/csv")

    if st.button("← Start over"):
        st.session_state.clear(); _rerun()
