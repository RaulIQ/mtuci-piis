import io
import logging
from typing import Any

import numpy as np

from app.model import KwsInferenceService
from app.monitoring import observe_request_success_ms
from app.schemas import StreamPredictResponse
from app.streaming import StreamParams


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

    if t_sec.ndim != 1 or is_silence.ndim != 1 or log_mel.ndim != 5:
        raise ValueError("invalid npz layout: expected t_sec [N], is_silence [N], log_mel [N,1,1,n_mels,time]")
    n = int(t_sec.shape[0])
    if is_silence.shape[0] != n or log_mel.shape[0] != n:
        raise ValueError("npz arrays length mismatch")

    if params.target_labels is None:
        params.target_labels = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

    expected_mels = int(service.model.mel.n_mels)
    windows: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    refractory_until = -1.0

    for i in range(n):
        t = float(t_sec[i])
        if bool(is_silence[i]):
            pred = service.silence_prediction()
        else:
            pred = service.predict_from_normalized_logmel(log_mel[i])

        label = pred["predicted_class"]
        conf = float(pred["confidence"])
        windows.append(
            {
                "t_sec": round(t, 3),
                "predicted_class": label,
                "confidence": round(conf, 4),
            }
        )

        is_trigger = (
            label in params.target_labels
            and conf >= params.confidence_threshold
            and t >= refractory_until
        )
        if is_trigger:
            detections.append(
                {
                    "t_sec": round(t, 3),
                    "label": label,
                    "confidence": round(conf, 4),
                }
            )
            refractory_until = t + params.refractory_sec

    latency_ms = observe_request_success_ms(endpoint, started_perf)

    logger.info(
        "predict_stream_logmel_ok file=%s windows=%s detections=%s latency_ms=%.2f",
        filename,
        len(windows),
        len(detections),
        latency_ms,
    )

    return StreamPredictResponse(
        windows_processed=len(windows),
        detections_count=len(detections),
        detections=detections,
        window_predictions=windows,
    )
