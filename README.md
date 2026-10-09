# Fake News Detector — open the app

## Fastest (single file, no build)
1. Install Python 3.10+ then:
   ```
   pip install fastapi uvicorn pydantic
   ```
2. Unzip, open a terminal in this folder, run:
   ```
   python -m uvicorn demo:app --port 8000
   ```
3. Open **http://localhost:8000** in your browser.
4. Paste headline (optional) + article (20+ words) → **Analyze**.
   Try the sample buttons: Council budget / Miracle cure / ALL-CAPS rant.
5. Stop: press Ctrl+C in the terminal.

## Full version (FastAPI backend + React frontend)
Backend:
```
cd backend && pip install -r requirements.txt && python -m uvicorn app.main:app --port 8000
```
Frontend (new terminal):
```
cd frontend && npm install && npm run dev
```
Open the Vite URL it prints (usually http://localhost:5173).

## Notes
- `demo.py` runs alone; keyword fallback works with no model file.
- For full accuracy, place your trained pickle at `backend/models/tfidf.pkl`.
- BERT compare lives only in `backend/` (needs torch + transformers, ~250MB download).
- Style signal only — not fact-checking.
