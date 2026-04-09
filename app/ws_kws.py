"""WebSocket: поток float32 PCM с клиента, окно 1 с, опрос каждые poll_interval_sec (как realtime KWS)."""

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.model import KwsInferenceService

logger = logging.getLogger("kws-service")

DEFAULT_POLL = 0.25
DEFAULT_REFRACTORY = 0.8
DEFAULT_CONF = 0.55
DEFAULT_LABELS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
MAX_BUFFER_SEC = 4.0
INPUT_AUDIO = "audio"
INPUT_LOG_MEL = "log_mel"
INPUT_AUDIO_VIA_LOG_MEL = "audio_via_log_mel"


def _parse_config(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if "sample_rate" not in data:
        raise ValueError("sample_rate required")
    sr = int(data["sample_rate"])
    if sr < 8000 or sr > 96000:
        raise ValueError("sample_rate out of range")
    labels_raw = data.get("target_labels")
    if isinstance(labels_raw, list):
        target_labels = {str(x).strip() for x in labels_raw if str(x).strip()}
    elif isinstance(labels_raw, str):
        target_labels = {x.strip() for x in labels_raw.split(",") if x.strip()}
    else:
        target_labels = set(DEFAULT_LABELS)
    return {
        "sample_rate": sr,
        "input_type": str(data.get("input_type", INPUT_AUDIO)),
        "poll_interval_sec": float(data.get("poll_interval_sec", DEFAULT_POLL)),
        "refractory_sec": float(data.get("refractory_sec", DEFAULT_REFRACTORY)),
        "confidence_threshold": float(data.get("confidence_threshold", DEFAULT_CONF)),
        "target_labels": target_labels or set(DEFAULT_LABELS),
    }


async def handle_kws_ws(websocket: WebSocket, service: KwsInferenceService) -> None:
    await websocket.accept()
    buf = np.array([], dtype=np.float32)
    cfg: dict[str, Any] | None = None
    last_infer_mono = 0.0
    stream_start_mono: float | None = None
    refractory_until = -1.0
    frame_idx = 0

    try:
        first = await websocket.receive_text()
        cfg = _parse_config(first)
        await websocket.send_json(
            {
                "type": "ready",
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
        )
    except WebSocketDisconnect:
        logger.info("ws_kws disconnected before initial config")
        return
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=4400, reason=str(e)[:120])
        return

    assert cfg is not None
    in_sr = cfg["sample_rate"]
    input_type = cfg["input_type"]
    poll_iv = max(0.05, min(1.0, cfg["poll_interval_sec"]))
    refractory_sec = cfg["refractory_sec"]
    conf_thr = cfg["confidence_threshold"]
    targets = cfg["target_labels"]
    max_samples = int(MAX_BUFFER_SEC * in_sr)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue

            if "text" in message and message["text"] is not None:
                try:
                    cfg = _parse_config(message["text"])
                    in_sr = cfg["sample_rate"]
                    input_type = cfg["input_type"]
                    poll_iv = max(0.05, min(1.0, cfg["poll_interval_sec"]))
                    refractory_sec = cfg["refractory_sec"]
                    conf_thr = cfg["confidence_threshold"]
                    targets = cfg["target_labels"]
                    max_samples = int(MAX_BUFFER_SEC * in_sr)
                    buf = np.array([], dtype=np.float32)
                    last_infer_mono = 0.0
                    stream_start_mono = None
                    refractory_until = -1.0
                    await websocket.send_json({"type": "reconfigured"})
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                continue

            if "bytes" not in message or message["bytes"] is None:
                continue

            raw = message["bytes"]
            if len(raw) % 4 != 0:
                await websocket.send_json({"type": "error", "message": "binary length not multiple of 4"})
                continue

            chunk = np.frombuffer(raw, dtype=np.float32).copy()
            if chunk.size == 0:
                continue

            if stream_start_mono is None:
                stream_start_mono = time.monotonic()

            now_m = time.monotonic()
            frame_idx += 1
            t_sec = round(now_m - stream_start_mono, 3)

            if input_type in {INPUT_AUDIO, INPUT_AUDIO_VIA_LOG_MEL}:
                buf = np.concatenate([buf, chunk])
                if buf.size > max_samples:
                    buf = buf[-max_samples:]

                win = int(1.0 * in_sr)
                if buf.size < win or now_m - last_infer_mono < poll_iv:
                    continue
                last_infer_mono = now_m
                window_audio = buf[-win:].astype(np.float32, copy=False)
                try:
                    if input_type == INPUT_AUDIO:
                        pred = await asyncio.to_thread(service.predict, window_audio, in_sr)
                    else:
                        pred = await asyncio.to_thread(service.predict_audio_via_log_mels, window_audio, in_sr)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("ws_kws_predict_failed")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc), "frame_idx": frame_idx}
                        )
                    continue
            elif input_type == INPUT_LOG_MEL:
                expected = service.n_mels * service.spec_frames
                if chunk.size != expected:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"log_mel size mismatch: got {chunk.size}, expected {expected}",
                        }
                    )
                    continue
                if now_m - last_infer_mono < poll_iv:
                    continue
                last_infer_mono = now_m
                log_mels = chunk.reshape(service.n_mels, service.spec_frames)
                try:
                    pred = await asyncio.to_thread(service.predict_log_mels, log_mels)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("ws_kws_predict_log_mel_failed")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc), "frame_idx": frame_idx}
                        )
                    continue
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"unsupported input_type={input_type}"}
                )
                continue

            label = pred["predicted_class"]
            conf = float(pred["confidence"])
            trigger = (
                label in targets
                and conf >= conf_thr
                and t_sec >= refractory_until
            )
            detection = None
            if trigger:
                detection = {"t_sec": t_sec, "label": label, "confidence": conf}
                refractory_until = t_sec + refractory_sec

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            await websocket.send_json(
                {
                    "type": "prediction",
                    "frame_idx": frame_idx,
                    "t_sec": t_sec,
                    "predicted_class": label,
                    "confidence": conf,
                    "inference_ms": pred["inference_ms"],
                    "trigger": trigger,
                    "detection": detection,
                    "top_k": pred.get("top_k"),
                }
            )

    except WebSocketDisconnect:
        logger.info("ws_kws client disconnected")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws_kws error=%s", exc)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
