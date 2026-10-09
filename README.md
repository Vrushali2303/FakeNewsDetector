# 📰 Fake News Detector

**Check a story before you share it.**

Paste a headline and article, get a red-pen verdict in under a second. Fake News Detector grades the *writing style* of a piece (shouty caps, hype words, exclamation bursts, thin copy) and tells you whether it reads like real reporting or like clickbait.

> **Style signal only.** It does not fact-check claims, sources, or dates. Treat every verdict as a reason to verify, not as proof.

---

## ✨ Features

- **Instant REAL / FAKE verdict** with a confidence score and gauge
- **Zero-setup demo**: one Python file, one command, no build step
- **Seven sample articles** (council report, miracle cure, shouty rant, vaccine rumor, election claim, crypto windfall, research study) to try the tool without writing anything
- **Dark mode** toggle
- **Two models side by side** in the full version: a TF-IDF baseline and DistilBERT
- **Word-level signals** from the TF-IDF model in the full backend: which words push toward FAKE and which toward REAL
- **Input guard**: requires about 20 words so verdicts never rest on a single line

---

## 🚀 Quick start

The fastest way in is the single-file demo.

**1. Install dependencies** (Python 3.10+):

```bash
pip install fastapi uvicorn pydantic
```

**2. Start the server** from this folder:

```bash
python -m uvicorn demo:app --port 8000
```

**3. Open** [http://localhost:8000](http://localhost:8000) in your browser.

**4. Paste** a headline (optional) and an article of 20+ words, then press **Check this article**. Or press one of the sample buttons first.

To stop the server, press `Ctrl+C` in the terminal.

---

## 🧩 Full version (FastAPI backend + React frontend)

The full version splits the app into an API and a UI, and shows the TF-IDF and DistilBERT results side by side.

**Backend** (terminal 1):

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

**Frontend** (terminal 2):

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints, usually [http://localhost:5173](http://localhost:5173).

---

## 🔌 API

The backend exposes two endpoints.

| Method | Path       | Purpose                                             |
|--------|------------|-----------------------------------------------------|
| GET    | `/health`  | Liveness check, returns `{"ok": true}`              |
| POST   | `/predict` | Scores a text and returns the `tfidf` and `bert` results |

`/predict` rejects inputs under 20 words with HTTP `422`.

---

## 🧠 How it works

| Model        | Where it runs              | What it does                                                                 |
|--------------|----------------------------|------------------------------------------------------------------------------|
| **TF-IDF**   | `demo.py` and `backend/`   | Linear classifier over word weights (scikit-learn, `predict_proba`). Lightweight, fast, and the default. |
| **DistilBERT** | `backend/` only          | Sentiment-based mapping (`POSITIVE` → REAL, `NEGATIVE` → FAKE). Demo only. |

**Without a trained model**, `demo.py` falls back to a keyword heuristic. It counts loaded words, ALL-CAPS runs, and exclamation marks. The app runs either way.

---

## 📦 Bring your own model

For full accuracy, place a trained scikit-learn bundle at:

```
backend/models/tfidf.pkl
```

The pickle must hold a dict with two keys:

```python
{"vectorizer": fitted_tfidf_vectorizer, "model": fitted_classifier}
```

`demo.py` also looks in `models/tfidf.pkl`, `model.pkl`, and `tfidf.pkl`.

The BERT comparison needs `torch` and `transformers`. The first run downloads about 250 MB.

---

## 🗂️ Project layout

```
.
├── demo.py              # Single-file app: API + inline UI
├── backend/
│   ├── app/
│   │   ├── main.py      # FastAPI routes
│   │   ├── model_tfidf.py
│   │   ├── model_bert.py
│   │   └── schemas.py
│   ├── models/          # Place tfidf.pkl here
│   └── tests/           # pytest suite
└── frontend/
    └── src/             # React + Vite UI (VerdictCard, api client)
```

---

## 🧪 Tests

```bash
cd backend
pytest
```

---

## ⚠️ Limits

- Reads style, not truth. Polished, calm writing can still be false.
- A short or unusual text can get a weak verdict. Low confidence means get a second opinion before you share.
- The keyword fallback is a demo heuristic, not a classifier. Load a trained model for real results.
