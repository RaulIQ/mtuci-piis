import io
import logging

import numpy as np

from app.model import KwsInferenceService
from app.monitoring import observe_request_success_ms
from app.schemas import StreamPredictResponse
from app.streaming import StreamParams
from app.streaming_logmel import process_logmel_npz_windows, window_predictions_for_response


def run_stream_from_logmel_npz(
    *,
    service: KwsInferenceService,
    raw_bytes: bytes,
    filename: str | None,
    endpoint: str,
    started_perf: float,
    params: StreamParams,
    logger: logging.Logger,
    empty_detail: str,
) -> StreamPredictResponse:
    if not raw_bytes:
        raise ValueError(empty_detail)

    bio = io.BytesIO(raw_bytes)
    data = np.load(bio, allow_pickle=False)
    t_sec = np.asarray(data["t_sec"], dtype=np.float64)
    log_mel = np.asarray(data["log_mel"], dtype=np.float32)
    is_silence = np.asarray(data["is_silence"], dtype=np.bool_)

    windows_enriched, detections, _ = process_logmel_npz_windows(
        service,
        params,
        t_sec,
        log_mel,
        is_silence,
        refractory_until=-1.0,
    )
    window_predictions = window_predictions_for_response(windows_enriched)

    latency_ms = observe_request_success_ms(endpoint, started_perf)

    logger.info(
        "predict_stream_logmel_ok file=%s windows=%s detections=%s latency_ms=%.2f",
        filename,
        len(window_predictions),
        len(detections),
        latency_ms,
    )

    return StreamPredictResponse(
        windows_processed=len(window_predictions),
        detections_count=len(detections),
        detections=detections,
        window_predictions=window_predictions,
    )
