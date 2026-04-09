"""WebSocket: поток float32 PCM с клиента, окно 1 с, опрос каждые poll_interval_sec (как realtime KWS)."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.label_parsing import parse_comma_separated_labels
from app.model import KwsInferenceService

logger = logging.getLogger("kws-service")

DEFAULT_POLL = 0.25
DEFAULT_REFRACTORY = 0.8
DEFAULT_CONF = 0.55
DEFAULT_LABELS = frozenset(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)
MAX_BUFFER_SEC = 4.0


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
        target_labels = parse_comma_separated_labels(labels_raw)
    else:
        target_labels = set(DEFAULT_LABELS)
    return {
        "sample_rate": sr,
        "poll_interval_sec": float(data.get("poll_interval_sec", DEFAULT_POLL)),
        "refractory_sec": float(data.get("refractory_sec", DEFAULT_REFRACTORY)),
        "confidence_threshold": float(data.get("confidence_threshold", DEFAULT_CONF)),
        "target_labels": target_labels or set(DEFAULT_LABELS),
    }


@dataclass
class KwsWsState:
    buf: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_infer_mono: float = 0.0
    stream_start_mono: float | None = None
    refractory_until: float = -1.0
    frame_idx: int = 0
    in_sr: int = 0
    poll_iv: float = DEFAULT_POLL
    refractory_sec: float = DEFAULT_REFRACTORY
    conf_thr: float = DEFAULT_CONF
    targets: set[str] = field(default_factory=lambda: set(DEFAULT_LABELS))
    max_samples: int = 0

    def apply_parsed_config(self, cfg: dict[str, Any]) -> None:
        self.in_sr = cfg["sample_rate"]
        self.poll_iv = max(0.05, min(1.0, cfg["poll_interval_sec"]))
        self.refractory_sec = cfg["refractory_sec"]
        self.conf_thr = cfg["confidence_threshold"]
        self.targets = cfg["target_labels"]
        self.max_samples = int(MAX_BUFFER_SEC * self.in_sr)

    def reset_stream_buffers(self) -> None:
        self.buf = np.array([], dtype=np.float32)
        self.last_infer_mono = 0.0
        self.stream_start_mono = None
        self.refractory_until = -1.0


def _decode_float32_chunk(raw: bytes) -> tuple[np.ndarray | None, str | None]:
    if len(raw) % 4 != 0:
        return None, "binary length not multiple of 4"
    chunk = np.frombuffer(raw, dtype=np.float32).copy()
    if chunk.size == 0:
        return None, None
    return chunk, None


def _append_chunk(state: KwsWsState, chunk: np.ndarray) -> None:
    if state.stream_start_mono is None:
        state.stream_start_mono = time.monotonic()
    state.buf = np.concatenate([state.buf, chunk])
    if state.buf.size > state.max_samples:
        state.buf = state.buf[-state.max_samples :]


def _evaluate_trigger(
    label: str,
    conf: float,
    t_sec: float,
    *,
    targets: set[str],
    conf_thr: float,
    refractory_until: float,
    refractory_sec: float,
) -> tuple[bool, dict[str, float | str] | None, float]:
    trigger = label in targets and conf >= conf_thr and t_sec >= refractory_until
    if not trigger:
        return False, None, refractory_until
    detection: dict[str, float | str] = {"t_sec": t_sec, "label": label, "confidence": conf}
    return True, detection, t_sec + refractory_sec


async def _send_ready(websocket: WebSocket, service: KwsInferenceService) -> None:
    await websocket.send_json(
        {
            "type": "ready",
            "model_version": service.model_version,
            "labels": service.labels,
        }
    )


async def _handle_text_reconfigure(websocket: WebSocket, state: KwsWsState, text: str) -> None:
    try:
        cfg = _parse_config(text)
        state.apply_parsed_config(cfg)
        state.reset_stream_buffers()
        await websocket.send_json({"type": "reconfigured"})
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        await websocket.send_json({"type": "error", "message": str(e)})


async def _run_predict_and_send(
    websocket: WebSocket,
    service: KwsInferenceService,
    state: KwsWsState,
    window_audio: np.ndarray,
    t_sec: float,
) -> bool:
    state.frame_idx += 1
    frame_idx = state.frame_idx
    try:
        pred = await asyncio.to_thread(service.predict, window_audio, state.in_sr)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws_kws_predict_failed")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "frame_idx": frame_idx}
            )
        return True

    label = pred["predicted_class"]
    conf = float(pred["confidence"])
    trigger, detection, state.refractory_until = _evaluate_trigger(
        label,
        conf,
        t_sec,
        targets=state.targets,
        conf_thr=state.conf_thr,
        refractory_until=state.refractory_until,
        refractory_sec=state.refractory_sec,
    )

    if websocket.client_state != WebSocketState.CONNECTED:
        return False

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
    return True


async def _handle_binary_chunk(
    websocket: WebSocket,
    service: KwsInferenceService,
    state: KwsWsState,
    raw: bytes,
) -> bool:
    chunk, err = _decode_float32_chunk(raw)
    if err is not None:
        await websocket.send_json({"type": "error", "message": err})
        return True
    if chunk is None:
        return True

    _append_chunk(state, chunk)

    now_m = time.monotonic()
    win = int(state.in_sr)
    if state.buf.size < win or now_m - state.last_infer_mono < state.poll_iv:
        return True

    state.last_infer_mono = now_m
    assert state.stream_start_mono is not None
    t_sec = round(now_m - state.stream_start_mono, 3)
    window_audio = state.buf[-win:].astype(np.float32, copy=False)
    return await _run_predict_and_send(websocket, service, state, window_audio, t_sec)


async def _receive_loop(websocket: WebSocket, service: KwsInferenceService, state: KwsWsState) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            break
        if message["type"] != "websocket.receive":
            continue

        if message.get("text") is not None:
            await _handle_text_reconfigure(websocket, state, message["text"])
            continue

        raw = message.get("bytes")
        if raw is None:
            continue

        if not await _handle_binary_chunk(websocket, service, state, raw):
            break


async def handle_kws_ws(websocket: WebSocket, service: KwsInferenceService) -> None:
    await websocket.accept()
    state = KwsWsState()

    try:
        first = await websocket.receive_text()
        cfg = _parse_config(first)
        state.apply_parsed_config(cfg)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        await websocket.close(code=4400, reason=str(e)[:120])
        return

    await _send_ready(websocket, service)

    try:
        await _receive_loop(websocket, service, state)
    except WebSocketDisconnect:
        logger.info("ws_kws client disconnected")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws_kws error=%s", exc)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
