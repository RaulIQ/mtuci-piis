import io
import logging
from datetime import datetime, timezone

import numpy as np

from app.model import KwsInferenceService
from app.monitoring import INFERENCE_LATENCY_MS, observe_request_success_ms
from app.schemas import PredictResponse
from app.storage import RequestLogStore


def run_predict_from_logmel_npy(
    *,
    service: KwsInferenceService,
    store: RequestLogStore,
    raw_bytes: bytes,
    filename: str | None,
    endpoint: str,
    started_perf: float,
    logger: logging.Logger,
    empty_detail: str,
) -> PredictResponse:
    if not raw_bytes:
        raise ValueError(empty_detail)

    bio = io.BytesIO(raw_bytes)
    log_mel = np.load(bio, allow_pickle=False)

    pred = service.predict_from_normalized_logmel(log_mel)

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
    logger.info(
        "predict_logmel_ok file=%s class=%s conf=%.4f latency_ms=%.2f",
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
