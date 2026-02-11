import os
from threading import Lock
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS_3 = ["negative", "neutral", "positive"]
LABELS_2 = ["negative", "positive"]

MODEL_DIR = os.getenv("MODEL_DIR", "outputs/seed_stability_stage2_best/seed_52")
MODEL_ID = os.getenv("MODEL_ID")
MAX_LENGTH_DEFAULT = int(os.getenv("MAX_LENGTH", "256"))

_model = None
_tokenizer = None
_num_labels = None
_load_lock = Lock()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    max_length: int = Field(default=MAX_LENGTH_DEFAULT, ge=8, le=512)
    uncertainty_neutral_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)
    max_length: int = Field(default=MAX_LENGTH_DEFAULT, ge=8, le=512)
    uncertainty_neutral_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _load_model_once() -> None:
    global _model, _tokenizer, _num_labels
    if _model is not None and _tokenizer is not None:
        return
    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return
        try:
            if MODEL_ID:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            else:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        except Exception as exc:
            source = f"MODEL_ID='{MODEL_ID}'" if MODEL_ID else f"MODEL_DIR='{MODEL_DIR}'"
            raise RuntimeError(f"Failed to load model from {source}.") from exc
        _model.eval()
        _num_labels = int(_model.config.num_labels)


def _predict(texts: List[str], max_length: int, uncertainty_neutral_threshold: Optional[float]):
    _load_model_once()
    enc = _tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(**enc).logits.cpu().numpy()
    probs = _softmax(logits)
    pred_ids = probs.argmax(axis=1)

    labels = LABELS_3 if _num_labels == 3 else LABELS_2
    outputs = []
    for idx, p in zip(pred_ids, probs):
        pred_label = labels[int(idx)]
        if _num_labels == 3 and uncertainty_neutral_threshold is not None:
            if float(np.max(p)) < uncertainty_neutral_threshold:
                pred_label = "neutral"

        outputs.append(
            {
                "label": pred_label,
                "probabilities": {labels[i]: float(p[i]) for i in range(len(labels))},
                "confidence": float(np.max(p)),
            }
        )
    return outputs


app = FastAPI(title="Sentiment English API", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_source": MODEL_ID if MODEL_ID else MODEL_DIR,
        "model_dir": MODEL_DIR,
        "model_loaded": _model is not None,
        "device": "cpu",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = _predict(
            texts=[req.text],
            max_length=req.max_length,
            uncertainty_neutral_threshold=req.uncertainty_neutral_threshold,
        )[0]
        return {"text": req.text, **result}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    clean_texts = [t.strip() for t in req.texts if t and t.strip()]
    if not clean_texts:
        raise HTTPException(status_code=400, detail="No valid texts provided.")

    try:
        results = _predict(
            texts=clean_texts,
            max_length=req.max_length,
            uncertainty_neutral_threshold=req.uncertainty_neutral_threshold,
        )
        return {"count": len(clean_texts), "results": [{"text": t, **r} for t, r in zip(clean_texts, results)]}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
