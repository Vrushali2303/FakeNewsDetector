from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TextIn, PredictOut
from .model_tfidf import predict_tfidf
from .model_bert import predict_bert

app = FastAPI(title="Fake News Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictOut)
def predict(inp: TextIn):
    if len(inp.text.split()) < 20:
        raise HTTPException(422, "Paste at least ~20 words")
    return {"tfidf": predict_tfidf(inp.text), "bert": predict_bert(inp.text)}
