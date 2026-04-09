import logging

from app.audio_bytes import load_mono_from_bytes
from app.monitoring import observe_request_success_ms
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
    audio, sr = load_mono_from_bytes(raw_bytes, empty_detail="Empty file")
    result = streaming.run(audio, sr, params)

    latency_ms = observe_request_success_ms(endpoint, started_perf)

    logger.info(
        "predict_stream_ok file=%s windows=%s detections=%s latency_ms=%.2f",
        filename,
        result["windows_processed"],
        result["detections_count"],
        latency_ms,
    )
    return StreamPredictResponse(**result)
