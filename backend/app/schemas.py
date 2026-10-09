from pydantic import BaseModel
from typing import List, Optional, Tuple

class TextIn(BaseModel):
    text: str

class Verdict(BaseModel):
    label: str
    confidence: float
    fake_words: List[Tuple[str, float]] = []
    real_words: List[Tuple[str, float]] = []
    available: bool = True

class PredictOut(BaseModel):
    tfidf: Verdict
    bert: Verdict
