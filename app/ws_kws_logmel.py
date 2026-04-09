"""WebSocket: incremental log-mel NPZ packets (same layout as /predict-stream-logmel)."""

import asyncio
import io
import json
import logging
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.label_parsing import parse_comma_separated_labels
from app.model import KwsInferenceService
from app.streaming import StreamParams
from app.streaming_logmel import process_logmel_npz_windows

logger = logging.getLogger("kws-service")

DEFAULT_REFRACTORY = 0.8
DEFAULT_CONF = 0.55
DEFAULT_LABELS = frozenset(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)


@dataclass
class KwsLogmelWsState:
    refractory_until: float = -1.0
    frame_idx: int = 0
    conf_thr: float = DEFAULT_CONF
    refractory_sec: float = DEFAULT_REFRACTORY
    targets: set[str] = field(default_factory=lambda: set(DEFAULT_LABELS))

    def apply_config(self, cfg: dict) -> None:
        self.targets = cfg["target_labels"]
        self.conf_thr = cfg["confidence_threshold"]
        self.refractory_sec = cfg["refractory_sec"]

    def reset_session(self) -> None:
        self.refractory_until = -1.0
        self.frame_idx = 0


def _parse_logmel_ws_config(data: dict) -> dict:
    labels_raw = data.get("target_labels")
    if isinstance(labels_raw, list):
        target_labels = {str(x).strip() for x in labels_raw if str(x).strip()}
    elif isinstance(labels_raw, str):
        target_labels = parse_comma_separated_labels(labels_raw)
    else:
        target_labels = set(DEFAULT_LABELS)
    return {
        "target_labels": target_labels or set(DEFAULT_LABELS),
        "confidence_threshold": float(data.get("confidence_threshold", DEFAULT_CONF)),
        "refractory_sec": float(data.get("refractory_sec", DEFAULT_REFRACTORY)),
    }


async def _send_ready(websocket: WebSocket, service: KwsInferenceService) -> None:
    await websocket.send_json(
        {
            "type": "ready",
            "model_version": service.model_version,
            "labels": service.labels,
        }
    )


def _load_and_process_npz(
    service: KwsInferenceService,
    state: KwsLogmelWsState,
    raw: bytes,
) -> tuple[list[dict], float]:
    bio = io.BytesIO(raw)
    data = np.load(bio, allow_pickle=False)
    t_sec = np.asarray(data["t_sec"], dtype=np.float64)
    log_mel = np.asarray(data["log_mel"], dtype=np.float32)
    is_silence = np.asarray(data["is_silence"], dtype=np.bool_)

    params = StreamParams(
        stride_sec=0.25,
        refractory_sec=state.refractory_sec,
        confidence_threshold=state.conf_thr,
        target_labels=set(state.targets),
    )
    windows, _detections, new_r = process_logmel_npz_windows(
        service,
        params,
        t_sec,
        log_mel,
        is_silence,
        refractory_until=state.refractory_until,
    )
    return windows, new_r


async def _handle_binary_npz(
    websocket: WebSocket,
    service: KwsInferenceService,
    state: KwsLogmelWsState,
    raw: bytes,
) -> bool:
    if not raw:
        return True
    try:
        windows, new_r = await asyncio.to_thread(_load_and_process_npz, service, state, raw)
        state.refractory_until = new_r
    except (ValueError, KeyError, OSError) as exc:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "message": str(exc)})
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws_kws_logmel_process_failed")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "message": str(exc)})
        return True

    for w in windows:
        state.frame_idx += 1
        frame_idx = state.frame_idx
        if websocket.client_state != WebSocketState.CONNECTED:
            return False
        await websocket.send_json(
            {
                "type": "prediction",
                "frame_idx": frame_idx,
                "t_sec": w["t_sec"],
                "predicted_class": w["predicted_class"],
                "confidence": w["confidence"],
                "inference_ms": w["inference_ms"],
                "trigger": w["trigger"],
                "detection": w["detection"],
                "top_k": w.get("top_k"),
            }
        )
    return True


async def _handle_text(websocket: WebSocket, state: KwsLogmelWsState, text: str) -> None:
    try:
        data = json.loads(text)
        cfg = _parse_logmel_ws_config(data)
        state.apply_config(cfg)
        state.reset_session()
        await websocket.send_json({"type": "reconfigured"})
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        await websocket.send_json({"type": "error", "message": str(e)})


async def _receive_loop(websocket: WebSocket, service: KwsInferenceService, state: KwsLogmelWsState) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            break
        if message["type"] != "websocket.receive":
            continue

        if message.get("text") is not None:
            await _handle_text(websocket, state, message["text"])
            continue

        raw = message.get("bytes")
        if raw is None:
            continue

        if not await _handle_binary_npz(websocket, service, state, raw):
            break


async def handle_kws_logmel_ws(websocket: WebSocket, service: KwsInferenceService) -> None:
    await websocket.accept()
    state = KwsLogmelWsState()

    try:
        first = await websocket.receive_text()
        cfg = _parse_logmel_ws_config(json.loads(first))
        state.apply_config(cfg)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        await websocket.close(code=4400, reason=str(e)[:120])
        return

    await _send_ready(websocket, service)

    try:
        await _receive_loop(websocket, service, state)
    except WebSocketDisconnect:
        logger.info("ws_kws_logmel client disconnected")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws_kws_logmel error=%s", exc)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
