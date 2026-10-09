from functools import lru_cache

@lru_cache
def get_bert():
    try:
        from transformers import pipeline
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None

def predict_bert(text: str):
    clf = get_bert()
    if clf is None:
        return {"label": "UNKNOWN", "confidence": 0.0, "fake_words": [], "real_words": [], "available": False}
    try:
        r = clf(text[:512])[0]
        # SST-2 POSITIVE ~ REAL, NEGATIVE ~ FAKE (demo mapping)
        label = "REAL" if r["label"] == "POSITIVE" else "FAKE"
        return {"label": label, "confidence": float(r["score"]), "fake_words": [], "real_words": [], "available": True}
    except Exception:
        return {"label": "UNKNOWN", "confidence": 0.0, "fake_words": [], "real_words": [], "available": False}
