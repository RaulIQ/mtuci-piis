import base64
import io
import logging
import os
import time
from datetime import datetime, timezone

import librosa
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.model import KwsInferenceService
from app.monitoring import INFERENCE_LATENCY_MS, REQUEST_LATENCY_MS, REQUESTS_TOTAL
from app.schemas import (
    PredictAudioRequest,
    PredictResponse,
    PredictSpectrogramRequest,
    StreamPredictResponse,
)
from app.storage import RequestLogStore
from app.streaming import SlidingWindowProcessor, StreamParams
from app.ws_kws import handle_kws_ws


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("kws-service")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/kws_resnet.pt")
DB_PATH = os.getenv("DB_PATH", "storage/requests.db")

app = FastAPI(title="KWS Inference Service", version="1.0.0")
store = RequestLogStore(DB_PATH)
service = KwsInferenceService(MODEL_PATH)
streaming = SlidingWindowProcessor(service)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "model_version": service.model_version,
        "labels": service.labels,
        "spec": {
            "sample_rate": service.sample_rate,
            "n_fft": service.n_fft,
            "hop_length": service.hop_length,
            "n_mels": service.n_mels,
            "frames": service.spec_frames,
        },
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/kws")
async def ws_kws(websocket: WebSocket):
    """Поток: первое сообщение — JSON config (sample_rate обязателен), далее binary float32 mono PCM."""
    await handle_kws_ws(websocket, service)


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


@app.post("/predict-spectrogram", response_model=PredictResponse)
def predict_spectrogram(payload: PredictSpectrogramRequest):
    started = time.perf_counter()
    try:
        if not payload.log_mels:
            raise HTTPException(status_code=400, detail="Empty log_mels")
        pred = service.predict_log_mels(payload.log_mels)

        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY_MS.labels(endpoint="/predict-spectrogram").observe(latency_ms)
        INFERENCE_LATENCY_MS.observe(pred["inference_ms"])
        REQUESTS_TOTAL.labels(endpoint="/predict-spectrogram", status="ok").inc()

        return PredictResponse(
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            latency_ms=latency_ms,
            model_version=service.model_version,
            top_k=pred["top_k"],
        )
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint="/predict-spectrogram", status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint="/predict-spectrogram", status="error").inc()
        logger.exception("predict_spectrogram_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-stream", response_model=StreamPredictResponse)
async def predict_stream(
    file: UploadFile = File(...),
    stride_sec: float = Form(0.25),
    refractory_sec: float = Form(0.8),
    confidence_threshold: float = Form(0.55),
    target_labels: str = Form("one,two,three,four,five,six,seven,eight,nine"),
):
    started = time.perf_counter()
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        audio, sr = librosa.load(io.BytesIO(content), sr=None, mono=True)
        labels_set = {x.strip() for x in target_labels.split(",") if x.strip()}
        params = StreamParams(
            stride_sec=stride_sec,
            refractory_sec=refractory_sec,
            confidence_threshold=confidence_threshold,
            target_labels=labels_set,
        )
        result = streaming.run(audio, sr, params)

        latency_ms = (time.perf_counter() - started) * 1000
        REQUEST_LATENCY_MS.labels(endpoint="/predict-stream").observe(latency_ms)
        REQUESTS_TOTAL.labels(endpoint="/predict-stream", status="ok").inc()
        logger.info(
            "predict_stream_ok file=%s windows=%s detections=%s latency_ms=%.2f",
            file.filename,
            result["windows_processed"],
            result["detections_count"],
            latency_ms,
        )
        return StreamPredictResponse(**result)
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint="/predict-stream", status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint="/predict-stream", status="error").inc()
        logger.exception("predict_stream_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

