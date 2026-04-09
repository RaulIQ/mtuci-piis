import io
import logging
import time
from datetime import datetime, timezone

import librosa

from app.model import KwsInferenceService
from app.monitoring import INFERENCE_LATENCY_MS, REQUEST_LATENCY_MS, REQUESTS_TOTAL
from app.schemas import PredictResponse
from app.storage import RequestLogStore


def run_predict_from_audio_bytes(
    *,
    service: KwsInferenceService,
    store: RequestLogStore,
    raw_bytes: bytes,
    filename: str | None,
    endpoint: str,
    started_perf: float,
    logger: logging.Logger,
    empty_detail: str,
    log_success: bool = True,
) -> PredictResponse:
    if not raw_bytes:
        raise ValueError(empty_detail)

    audio, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
    pred = service.predict(audio, sr=sr)

    latency_ms = (time.perf_counter() - started_perf) * 1000
    REQUEST_LATENCY_MS.labels(endpoint=endpoint).observe(latency_ms)
    INFERENCE_LATENCY_MS.observe(pred["inference_ms"])
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="ok").inc()

    now = datetime.now(timezone.utc).isoformat()
    store.write(
        created_at=now,
        filename=filename,
        predicted_class=pred["predicted_class"],
        confidence=pred["confidence"],
        latency_ms=latency_ms,
        model_version=service.model_version,
    )
    if log_success:
        logger.info(
            "predict_ok file=%s class=%s conf=%.4f latency_ms=%.2f",
            filename,
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
