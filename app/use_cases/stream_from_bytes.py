import io
import logging
import time

import librosa

from app.monitoring import REQUEST_LATENCY_MS, REQUESTS_TOTAL
from app.schemas import StreamPredictResponse
from app.streaming import SlidingWindowProcessor, StreamParams


def run_stream_from_bytes(
    *,
    streaming: SlidingWindowProcessor,
    raw_bytes: bytes,
    filename: str | None,
    endpoint: str,
    started_perf: float,
    params: StreamParams,
    logger: logging.Logger,
) -> StreamPredictResponse:
    if not raw_bytes:
        raise ValueError("Empty file")

    audio, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
    result = streaming.run(audio, sr, params)

    latency_ms = (time.perf_counter() - started_perf) * 1000
    REQUEST_LATENCY_MS.labels(endpoint=endpoint).observe(latency_ms)
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="ok").inc()

    logger.info(
        "predict_stream_ok file=%s windows=%s detections=%s latency_ms=%.2f",
        filename,
        result["windows_processed"],
        result["detections_count"],
        latency_ms,
    )
    return StreamPredictResponse(**result)
