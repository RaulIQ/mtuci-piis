"""Shared realtime KWS: mic → Python log-mel NPZ → WebSocket ``/ws/kws-logmel``."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from helpers.ws_traffic import WsTrafficCounter
import sounddevice as sd
import websockets

from helpers.client_logmel import pcm16k_window_to_logmel_npz_bytes
from services.api import fetch_mel_config
from services.urls import build_ws_kws_logmel_url


async def run_realtime_logmel_async(
    api_url: str,
    *,
    confidence: float,
    refractory: float,
    target_labels: list[str],
    stride: float,
    buffer_sec: float,
    blocksize: int,
    stop_event: threading.Event,
    on_message: Callable[[dict], None],
    traffic: WsTrafficCounter | None = None,
) -> None:
    """
    Capture 16 kHz mono, send NPZ packets until ``stop_event`` is set.
    ``on_message`` receives server JSON dicts (``type`` ready | prediction | error | reconfigured).
    """
    try:
        mel_cfg = fetch_mel_config(api_url)
    except Exception as exc:  # pylint: disable=broad-except
        on_message({"type": "runner_error", "message": f"mel-config: {exc}"})
        return

    ws_url = build_ws_kws_logmel_url(api_url)
    ws_payload = {
        "confidence_threshold": confidence,
        "refractory_sec": refractory,
        "target_labels": target_labels,
    }

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _t, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            on_message({"type": "audio_status", "message": str(status)})
        audio_q.put(indata.copy())

    stream: sd.InputStream | None = None
    try:
        stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            callback=_callback,
            blocksize=blocksize,
        )
        stream.start()
    except (OSError, sd.PortAudioError) as exc:
        on_message(
            {
                "type": "runner_error",
                "message": (
                    f"микрофон: {exc}. В Docker нет ALSA по умолчанию — "
                    "смонтируйте /dev/snd или запускайте Streamlit на хосте."
                ),
            }
        )
        on_message({"type": "runner_stopped"})
        return

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            payload_text = json.dumps(ws_payload)
            await ws.send(payload_text)
            if traffic is not None:
                traffic.add_up(len(payload_text.encode("utf-8")))
            first = await ws.recv()
            if isinstance(first, bytes):
                raise RuntimeError("expected JSON hello, got binary")
            hello = json.loads(first)
            if hello.get("type") != "ready":
                raise RuntimeError(f"expected ready, got: {hello}")
            on_message(hello)
            if traffic is not None and isinstance(first, str):
                traffic.add_down(len(first.encode("utf-8")))

            recv_task = asyncio.create_task(_forward_ws_messages(ws, on_message, traffic))

            buf = np.array([], dtype=np.float32)
            stream_start: float | None = None
            last_sent = 0.0

            try:
                while not stop_event.is_set():
                    drained = False
                    while True:
                        try:
                            chunk = audio_q.get_nowait()
                            drained = True
                            buf = np.concatenate([buf, chunk.reshape(-1)])
                        except queue.Empty:
                            break
                    max_keep = int(16000 * buffer_sec)
                    if buf.size > max_keep:
                        buf = buf[-max_keep:]

                    now = time.monotonic()
                    if buf.size < 16000:
                        if not drained:
                            await asyncio.sleep(0.02)
                        continue

                    if stream_start is None:
                        stream_start = now

                    if now - last_sent < stride:
                        if not drained:
                            await asyncio.sleep(0.01)
                        continue

                    last_sent = now
                    t_sec = round(now - stream_start, 3)
                    window = buf[-16000:].astype(np.float32, copy=False)

                    npz_bytes, prep_err = pcm16k_window_to_logmel_npz_bytes(
                        window,
                        t_sec,
                        sample_rate=int(mel_cfg["sample_rate"]),
                        n_fft=int(mel_cfg["n_fft"]),
                        hop_length=int(mel_cfg["hop_length"]),
                        n_mels=int(mel_cfg["n_mels"]),
                    )
                    if prep_err:
                        on_message({"type": "prep_warning", "message": prep_err})
                        continue

                    await ws.send(npz_bytes)
                    if traffic is not None:
                        traffic.add_up(len(npz_bytes))
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
    except Exception as exc:  # pylint: disable=broad-except
        on_message({"type": "runner_error", "message": str(exc)})
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
    on_message({"type": "runner_stopped"})


async def _forward_ws_messages(
    ws,
    on_message: Callable[[dict], None],
    traffic: WsTrafficCounter | None,
) -> None:
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                on_message({"type": "runner_error", "message": "unexpected binary from server"})
                continue
            if traffic is not None:
                traffic.add_down(len(raw.encode("utf-8")))
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                on_message({"type": "runner_error", "message": f"invalid json: {raw!r}"})
                continue
            on_message(msg)
    except asyncio.CancelledError:
        raise


def run_realtime_logmel_in_thread(
    *,
    api_url: str,
    confidence: float,
    refractory: float,
    target_labels: list[str],
    stride: float,
    buffer_sec: float,
    blocksize: int,
    stop_event: threading.Event,
    on_message: Callable[[dict], None],
    traffic: WsTrafficCounter | None = None,
) -> threading.Thread:
    """Runs ``run_realtime_logmel_async`` in a daemon thread."""

    def _target() -> None:
        asyncio.run(
            run_realtime_logmel_async(
                api_url,
                confidence=confidence,
                refractory=refractory,
                target_labels=target_labels,
                stride=stride,
                buffer_sec=buffer_sec,
                blocksize=blocksize,
                stop_event=stop_event,
                on_message=on_message,
                traffic=traffic,
            )
        )

    t = threading.Thread(target=_target, daemon=True, name="realtime_logmel_kws")
    t.start()
    return t
