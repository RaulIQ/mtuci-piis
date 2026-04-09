import logging
from datetime import datetime, timezone

from app.audio_bytes import load_mono_from_bytes
from app.model import KwsInferenceService
from app.monitoring import INFERENCE_LATENCY_MS, observe_request_success_ms
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
    audio, sr = load_mono_from_bytes(raw_bytes, empty_detail=empty_detail)
    pred = service.predict(audio, sr=sr)

    latency_ms = observe_request_success_ms(endpoint, started_perf)
    INFERENCE_LATENCY_MS.observe(pred["inference_ms"])

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
