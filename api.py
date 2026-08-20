"""FastAPI app serving both gender+emotion classifiers side by side.

    uvicorn api:app --reload

    curl -X POST http://localhost:8000/predict/cnn-lstm -F "file=@sample.wav"
    curl -X POST http://localhost:8000/predict/mlp -F "file=@sample.wav"
"""

import os
import tempfile

import joblib
import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.labels import LABELS
from preprocessing.features_flat import featurize_file

app = FastAPI(title="Speech Emotion Recognition API")

CNN_LSTM_CHECKPOINT = os.getenv("CNN_LSTM_CHECKPOINT", "checkpoints/model.keras")
CNN_LSTM_SCALER = os.getenv("CNN_LSTM_SCALER", "checkpoints/scaler.joblib")
MLP_MODEL_PATH = os.getenv("MLP_MODEL_PATH", "model/emotion_model.joblib")

_cnn_lstm_model = None
_cnn_lstm_scaler = None
_mlp_bundle = None


def get_cnn_lstm():
    global _cnn_lstm_model, _cnn_lstm_scaler
    if _cnn_lstm_model is None:
        if not os.path.exists(CNN_LSTM_CHECKPOINT):
            raise HTTPException(
                status_code=503,
                detail=f"No trained CNN+LSTM checkpoint at {CNN_LSTM_CHECKPOINT}. Run train_cnn_lstm.py first.",
            )
        _cnn_lstm_model = keras.models.load_model(CNN_LSTM_CHECKPOINT)  # trusted: trained locally
        _cnn_lstm_scaler = joblib.load(CNN_LSTM_SCALER)["scaler"]  # trusted: trained locally
    return _cnn_lstm_model, _cnn_lstm_scaler


def get_mlp_bundle():
    global _mlp_bundle
    if _mlp_bundle is None:
        if not os.path.exists(MLP_MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"No trained MLP model at {MLP_MODEL_PATH}. Run train_mlp.py first.",
            )
        _mlp_bundle = joblib.load(MLP_MODEL_PATH)  # trusted: trained locally by train_mlp.py
    return _mlp_bundle


class PredictResponse(BaseModel):
    top_label: str
    confidence: float
    scores: dict


async def _read_upload_to_tempfile(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        return tmp.name


@app.post("/predict/cnn-lstm", response_model=PredictResponse)
async def predict_cnn_lstm(file: UploadFile = File(...)):
    model, scaler = get_cnn_lstm()

    tmp_path = await _read_upload_to_tempfile(file)
    try:
        features = featurize_file(tmp_path).reshape(1, -1)
    finally:
        os.unlink(tmp_path)

    features = scaler.transform(features).reshape(1, -1, 1)
    probs = model.predict(features, verbose=0)[0]

    scores = {label: round(float(p), 4) for label, p in zip(LABELS, probs)}
    top_idx = int(probs.argmax())

    return PredictResponse(top_label=LABELS[top_idx], confidence=round(float(probs[top_idx]), 4), scores=scores)


@app.post("/predict/mlp", response_model=PredictResponse)
async def predict_mlp(file: UploadFile = File(...)):
    bundle = get_mlp_bundle()
    model, scaler, labels = bundle["model"], bundle["scaler"], bundle["labels"]

    tmp_path = await _read_upload_to_tempfile(file)
    try:
        features = featurize_file(tmp_path).reshape(1, -1)
    finally:
        os.unlink(tmp_path)

    features = scaler.transform(features)
    probs = model.predict_proba(features)[0]

    scores = {label: round(float(p), 4) for label, p in zip(labels, probs)}
    top_idx = int(probs.argmax())

    return PredictResponse(top_label=labels[top_idx], confidence=round(float(probs[top_idx]), 4), scores=scores)


@app.get("/health")
async def health():
    cnn_lstm_loaded = os.path.exists(CNN_LSTM_CHECKPOINT)
    mlp_loaded = os.path.exists(MLP_MODEL_PATH)
    mlp_accuracy = None
    if mlp_loaded:
        mlp_accuracy = round(float(joblib.load(MLP_MODEL_PATH)["accuracy"]), 4)  # trusted: locally-trained
    cnn_lstm_accuracy = None
    if cnn_lstm_loaded and os.path.exists(CNN_LSTM_SCALER):
        cnn_lstm_accuracy = round(float(joblib.load(CNN_LSTM_SCALER)["accuracy"]), 4)  # trusted: locally-trained
    return {
        "status": "ok",
        "models": {
            "cnn_lstm": {"loaded": cnn_lstm_loaded, "test_accuracy": cnn_lstm_accuracy},
            "mlp": {"loaded": mlp_loaded, "test_accuracy": mlp_accuracy},
        },
    }


# Serves static/index.html at http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")
