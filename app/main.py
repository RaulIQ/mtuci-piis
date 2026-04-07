import base64
import io
import logging
import os
import time
from datetime import datetime, timezone

import librosa
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.model import KwsInferenceService
from app.monitoring import INFERENCE_LATENCY_MS, REQUEST_LATENCY_MS, REQUESTS_TOTAL
from app.schemas import PredictAudioRequest, PredictResponse
from app.storage import RequestLogStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("kws-service")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/kwc_cnn.pt")
DB_PATH = os.getenv("DB_PATH", "storage/requests.db")

app = FastAPI(title="KWS Inference Service", version="1.0.0")
store = RequestLogStore(DB_PATH)
service = KwsInferenceService(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "model_version": service.model_version,
        "labels": service.labels,
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        audio, sr = librosa.load(io.BytesIO(content), sr=None, mono=True)
        pred = service.predict(audio, sr=sr)

        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY_MS.labels(endpoint="/predict").observe(latency_ms)
        INFERENCE_LATENCY_MS.observe(pred["inference_ms"])
        REQUESTS_TOTAL.labels(endpoint="/predict", status="ok").inc()

        now = datetime.now(timezone.utc).isoformat()
        store.write(
            created_at=now,
            filename=file.filename,
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            latency_ms=latency_ms,
            model_version=service.model_version,
        )
        logger.info(
            "predict_ok file=%s class=%s conf=%.4f latency_ms=%.2f",
            file.filename,
            pred["predicted_class"],
            pred["confidence"],
            latency_ms,
        )

        return PredictResponse(
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            latency_ms=latency_ms,
            model_version=service.model_version,
            top_k=pred["top_k"],
        )
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint="/predict", status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint="/predict", status="error").inc()
        logger.exception("predict_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-base64", response_model=PredictResponse)
def predict_base64(payload: PredictAudioRequest):
    started = time.perf_counter()
    try:
        content = base64.b64decode(payload.audio_base64)
        if not content:
            raise HTTPException(status_code=400, detail="Empty payload")

        audio, sr = librosa.load(io.BytesIO(content), sr=None, mono=True)
        pred = service.predict(audio, sr=sr)

        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY_MS.labels(endpoint="/predict-base64").observe(latency_ms)
        INFERENCE_LATENCY_MS.observe(pred["inference_ms"])
        REQUESTS_TOTAL.labels(endpoint="/predict-base64", status="ok").inc()

        now = datetime.now(timezone.utc).isoformat()
        store.write(
            created_at=now,
            filename="base64_payload.wav",
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            latency_ms=latency_ms,
            model_version=service.model_version,
        )
        return PredictResponse(
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            latency_ms=latency_ms,
            model_version=service.model_version,
            top_k=pred["top_k"],
        )
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint="/predict-base64", status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint="/predict-base64", status="error").inc()
        logger.exception("predict_base64_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

