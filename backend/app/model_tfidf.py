import pickle, os
from functools import lru_cache

PATHS = ["models/tfidf.pkl", "../models/tfidf.pkl", "model.pkl"]

@lru_cache
def get_tfidf():
    for p in PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

def predict_tfidf(text: str):
    bundle = get_tfidf()
    if bundle is None:
        return {"label": "UNKNOWN", "confidence": 0.0, "fake_words": [], "real_words": [], "available": False}
    vec, clf = bundle["vectorizer"], bundle["model"]
    proba = clf.predict_proba(vec.transform([text]))[0]
    idx = int(proba.argmax())
    raw = clf.classes_[idx]
    label = {0: "FAKE", 1: "REAL", "0": "FAKE", "1": "REAL"}.get(raw, str(raw))
    # top contributing words
    try:
        names = vec.get_feature_names_out()
        coefs = clf.coef_[0]
        row = vec.transform([text]).tocoo()
        scored = sorted(((names[c], float(coefs[c] * v)) for c, v in zip(row.col, row.data)), key=lambda x: -abs(x[1]))[:10]
        fake = [(w, s) for w, s in scored if s > 0][:5]
        real = [(w, -s) for w, s in scored if s < 0][:5]
    except Exception:
        fake, real = [], []
    return {"label": label, "confidence": float(proba[idx]), "fake_words": fake, "real_words": real, "available": True}
