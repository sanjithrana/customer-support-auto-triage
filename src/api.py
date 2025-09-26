# src/api.py
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Optional

MODEL_PATH = os.environ.get("MODEL_PATH", "models/ticket_model.joblib")

app = FastAPI(title="Ticket Auto-Triage API")

class PredictRequest(BaseModel):
    Subject: Optional[str] = ""
    Description: Optional[str] = ""
    Priority: Optional[str] = None

class PredictResponse(BaseModel):
    category: str
    confidence: Optional[float] = None
    latency_ms: float

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train and save model first.")
    model = joblib.load(MODEL_PATH)
    # If model is pipeline -> it will handle vectorization
    print("Model loaded from", MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.Subject or "") + " " + (req.Description or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Either Subject or Description must be provided")
    # Minimal inline cleaning to match training
    # NOTE: full cleaning identical to training is recommended; here we rely on the pipeline (which expects cleaned text)
    # If you used the preprocess.clean_text during training on combined text, you may want to apply it here too.
    from preprocess import clean_text
    cleaned = clean_text(text)
    start = time.time()
    preds = model.predict([cleaned])
    latency_ms = (time.time() - start) * 1000.0
    # Attempt to get probability/confidence if supported
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])
            # choose max probability
            confidence = float(proba.max())
    except Exception:
        confidence = None
    return PredictResponse(category=str(preds[0]), confidence=confidence, latency_ms=latency_ms)

@app.get("/")
def root():
    return {"message": "Ticket Auto-Triage API. Use POST /predict"}
## src/api.py
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Optional

from src.preprocess import clean_text

MODEL_PATH = os.environ.get("MODEL_PATH", "models/ticket_model.joblib")

app = FastAPI(title="Ticket Auto-Triage API")

class PredictRequest(BaseModel):
    Subject: Optional[str] = ""
    Description: Optional[str] = ""
    Priority: Optional[str] = None

class PredictResponse(BaseModel):
    category: str
    confidence: Optional[float] = None
    latency_ms: float

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train and save model first.")
    model = joblib.load(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.Subject or "") + " " + (req.Description or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Either Subject or Description must be provided")
    cleaned = clean_text(text)
    start = time.time()
    preds = model.predict([cleaned])
    latency_ms = (time.time() - start) * 1000.0
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])
            confidence = float(proba.max())
    except Exception:
        confidence = None
    return PredictResponse(category=str(preds[0]), confidence=confidence, latency_ms=latency_ms)

@app.get("/")
def root():
    return {"message": "Ticket Auto-Triage API. Use POST /predict"}

# 👇 This is the block you asked for
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
