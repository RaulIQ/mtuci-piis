import json
import threading
import time
from collections import deque
from typing import Any

import av
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from ui.services.kws_model import compute_canonical_log_mels

DIGIT_MAP = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "zero": "0",
}


def push_log(logs: list[str], message: str) -> None:
    logs.insert(0, f"[{time.strftime('%H:%M:%S')}] {message}")
    del logs[10:]


def render_center(container: Any, digit: str | None) -> None:
    if digit is None:
        container.markdown(
            """
            <div style="min-height:140px;display:flex;align-items:center;justify-content:center;">
              <p style="margin:0;text-align:center;font-size:1.5rem;color:#888;">Скажите число</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    container.markdown(
        f"""
        <div style="min-height:140px;display:flex;align-items:center;justify-content:center;">
          <p style="margin:0;text-align:center;font-size:6rem;font-weight:700;color:#111;">{digit}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def update_center_digit(
    predicted_class: str,
    confidence: float,
    current_digit: str | None,
    bad_since: float | None,
    confidence_threshold: float,
    target_labels: list[str],
    bad_window_sec: float = 2.0,
) -> tuple[str | None, float | None]:
    low = str(predicted_class or "").lower()
    targets = {x.lower() for x in target_labels}
    valid = low in targets and confidence >= confidence_threshold

    if valid:
        return DIGIT_MAP.get(low, low), None

    if low in {"silence", "unknown"}:
        if current_digit is None:
            return None, None
        if bad_since is None:
            return current_digit, time.monotonic()
        if time.monotonic() - bad_since >= bad_window_sec:
            return None, None
        return current_digit, bad_since

    return current_digit, None


def audio_frame_to_mono_float32(frame: av.AudioFrame) -> np.ndarray:
    arr = frame.to_ndarray()
    if arr.ndim == 2:
        mono = arr.mean(axis=0)
    else:
        mono = arr.reshape(-1)

    if np.issubdtype(mono.dtype, np.integer):
        info = np.iinfo(mono.dtype)
        scale = max(abs(info.min), info.max)
        mono = mono.astype(np.float32) / float(scale)
    else:
        mono = mono.astype(np.float32)
    return mono.reshape(-1)


def run_realtime_bridge(
    *,
    webrtc_ctx: Any,
    audio_lock: threading.Lock,
    audio_chunks: deque[np.ndarray],
    audio_meta: dict[str, Any],
    ws_url: str,
    ws_input_type: str,
    ws_config: dict[str, Any],
    model_sr: int,
    poll_interval_sec: float,
    confidence_threshold: float,
    target_labels: list[str],
    mel_tf: Any,
    frames: int,
    status_placeholder: Any,
    traffic_placeholder: Any,
    center_placeholder: Any,
    log_placeholder: Any,
) -> None:
    input_sr = int(audio_meta["sample_rate"] or model_sr)
    tx_bytes = 0
    rx_bytes = 0
    traffic_start = time.monotonic()
    center_digit = None
    bad_since = None
    logs: list[str] = []
    backend_buf = np.array([], dtype=np.float32)
    last_feature_sent = 0.0
    max_buffer_samples = max(input_sr * 4, model_sr * 4)

    bridge_cfg = dict(ws_config)
    bridge_cfg["sample_rate"] = input_sr if ws_input_type == "audio" else model_sr

    with connect(ws_url, open_timeout=10, close_timeout=3) as conn:
        cfg_text = json.dumps(bridge_cfg)
        conn.send(cfg_text)
        tx_bytes += len(cfg_text.encode("utf-8"))

        try:
            ready_msg = conn.recv(timeout=5)
            if isinstance(ready_msg, str):
                rx_bytes += len(ready_msg.encode("utf-8"))
                ready_payload = json.loads(ready_msg)
                push_log(logs, f"ready {ready_payload.get('model_version', '?')} input_type={ws_input_type}")
            else:
                push_log(logs, "ready: unexpected binary message")
        except TimeoutError:
            push_log(logs, "timeout waiting ready from API")

        while webrtc_ctx.state.playing:
            new_chunks: list[np.ndarray] = []
            with audio_lock:
                while audio_chunks:
                    new_chunks.append(audio_chunks.popleft())
                current_sr = int(audio_meta["sample_rate"] or input_sr)

            if new_chunks:
                if ws_input_type == "audio":
                    for chunk in new_chunks:
                        payload = np.asarray(chunk, dtype=np.float32)
                        conn.send(payload.tobytes())
                        tx_bytes += payload.nbytes
                else:
                    joined = np.concatenate(new_chunks).astype(np.float32, copy=False)
                    backend_buf = np.concatenate([backend_buf, joined])
                    if backend_buf.size > max_buffer_samples:
                        backend_buf = backend_buf[-max_buffer_samples:]

            if ws_input_type == "log_mel":
                now = time.monotonic()
                if backend_buf.size >= current_sr and now - last_feature_sent >= poll_interval_sec:
                    last_feature_sent = now
                    window_audio = backend_buf[-current_sr:]
                    log_mels = compute_canonical_log_mels(
                        window_audio,
                        sr=current_sr,
                        mel_tf=mel_tf,
                        frames=frames,
                    )
                    if log_mels is None:
                        center_digit, bad_since = update_center_digit(
                            "silence",
                            1.0,
                            center_digit,
                            bad_since,
                            confidence_threshold=confidence_threshold,
                            target_labels=target_labels,
                        )
                        push_log(logs, "local silence gate")
                    else:
                        conn.send(log_mels.astype(np.float32).tobytes())
                        tx_bytes += int(log_mels.nbytes)

            while True:
                try:
                    msg = conn.recv(timeout=0.01)
                except TimeoutError:
                    break
                except ConnectionClosed:
                    push_log(logs, "API websocket closed")
                    break

                if isinstance(msg, bytes):
                    rx_bytes += len(msg)
                    push_log(logs, f"binary message {len(msg)} B")
                    continue

                rx_bytes += len(msg.encode("utf-8"))
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    push_log(logs, f"raw {msg}")
                    continue

                if payload.get("type") == "prediction":
                    label = str(payload.get("predicted_class", ""))
                    conf = float(payload.get("confidence", 0.0))
                    center_digit, bad_since = update_center_digit(
                        label,
                        conf,
                        center_digit,
                        bad_since,
                        confidence_threshold=confidence_threshold,
                        target_labels=target_labels,
                    )
                    if payload.get("trigger") and payload.get("detection"):
                        push_log(logs, f"TRIGGER {payload['detection']}")
                    else:
                        push_log(logs, f"pred {label} ({conf:.3f}) inf_ms={payload.get('inference_ms', 0)}")
                elif payload.get("type") == "error":
                    push_log(logs, f"ERR {payload.get('message', '')}")
                elif payload.get("type") == "reconfigured":
                    push_log(logs, "reconfigured")
                else:
                    push_log(logs, msg)

            elapsed_sec = max(0.001, time.monotonic() - traffic_start)
            up_kbps = (tx_bytes * 8) / elapsed_sec / 1000
            down_kbps = (rx_bytes * 8) / elapsed_sec / 1000

            status_placeholder.success(f"Стрим {current_sr} Hz -> UI backend -> API ({ws_input_type})")
            traffic_placeholder.info(
                "UI <-> API: "
                f"tx {tx_bytes / 1024:.1f} KiB, "
                f"rx {rx_bytes / 1024:.1f} KiB, "
                f"up {up_kbps:.1f} kbps, "
                f"down {down_kbps:.1f} kbps"
            )
            render_center(center_placeholder, center_digit)
            log_placeholder.code("\n".join(logs) if logs else "Ожидание предсказаний...")
            time.sleep(0.05)
