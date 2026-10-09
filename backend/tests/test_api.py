from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app

c = TestClient(app)
LONG_FAKE = "breaking aliens endorse candidate shock claim " * 10
LONG_REAL = "city council approves budget for schools roads " * 10

def test_health():
    assert c.get("/health").json() == {"ok": True}

def test_short_rejected():
    assert c.post("/predict", json={"text": "too short"}).status_code == 422

def test_predict_shape():
    r = c.post("/predict", json={"text": LONG_REAL})
    assert r.status_code == 200
    assert set(r.json()) == {"tfidf", "bert"}
