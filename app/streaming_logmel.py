"""Shared log-mel NPZ window loop for HTTP stream and WebSocket."""

from typing import Any

import numpy as np

from app.model import KwsInferenceService
from app.streaming import StreamParams


def validate_logmel_npz_arrays(
    t_sec: np.ndarray,
    log_mel: np.ndarray,
    is_silence: np.ndarray,
) -> int:
    t_sec = np.asarray(t_sec, dtype=np.float64)
    log_mel = np.asarray(log_mel, dtype=np.float32)
    is_silence = np.asarray(is_silence, dtype=np.bool_)
    if t_sec.ndim != 1 or is_silence.ndim != 1 or log_mel.ndim != 5:
        raise ValueError(
            "invalid npz layout: expected t_sec [N], is_silence [N], log_mel [N,1,1,n_mels,time]"
        )
    n = int(t_sec.shape[0])
    if is_silence.shape[0] != n or log_mel.shape[0] != n:
        raise ValueError("npz arrays length mismatch")
    return n


def process_logmel_npz_windows(
    service: KwsInferenceService,
    params: StreamParams,
    t_sec: np.ndarray,
    log_mel: np.ndarray,
    is_silence: np.ndarray,
    *,
    refractory_until: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """
    Run inference for each window in arrays. Updates refractory state across windows.

    Returns (window_prediction_rows, detections_in_batch, new_refractory_until).
    Each row in the first list includes fields needed for WebSocket payloads.
    """
    if params.target_labels is None:
        params.target_labels = {
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        }

    n = validate_logmel_npz_arrays(t_sec, log_mel, is_silence)

    windows: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    r_until = refractory_until

    for i in range(n):
        t = float(t_sec[i])
        if bool(is_silence[i]):
            pred = service.silence_prediction()
        else:
            pred = service.predict_from_normalized_logmel(log_mel[i])

        label = pred["predicted_class"]
        conf = float(pred["confidence"])
        is_trigger = (
            label in params.target_labels
            and conf >= params.confidence_threshold
            and t >= r_until
        )
        detection: dict[str, float | str] | None = None
        if is_trigger:
            detection = {"t_sec": t, "label": label, "confidence": conf}
            detections.append(
                {
                    "t_sec": round(t, 3),
                    "label": label,
                    "confidence": round(conf, 4),
                }
            )
            r_until = t + params.refractory_sec

        windows.append(
            {
                "t_sec": round(t, 3),
                "predicted_class": label,
                "confidence": round(conf, 4),
                "inference_ms": pred["inference_ms"],
                "top_k": pred.get("top_k"),
                "trigger": is_trigger,
                "detection": detection,
            }
        )

    return windows, detections, r_until


def window_predictions_for_response(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip to StreamPredictResponse window_predictions shape."""
    return [
        {
            "t_sec": w["t_sec"],
            "predicted_class": w["predicted_class"],
            "confidence": w["confidence"],
        }
        for w in windows
    ]
