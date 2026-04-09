#!/usr/bin/env python3
"""Realtime KWS: capture 16 kHz mono mic, build log-mel NPZ locally, stream to ``/ws/kws-logmel``."""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import websockets

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UI_ROOT = _REPO_ROOT / "ui"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from helpers.client_logmel import pcm16k_window_to_logmel_npz_bytes
from helpers.labels import DEFAULT_TARGET_LABELS, parse_target_labels


def _rest_to_ws_base(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}"


def _fetch_mel_config(api_url: str) -> dict:
    r = requests.get(f"{api_url.rstrip('/')}/model/mel-config", timeout=10)
    r.raise_for_status()
    return r.json()


async def _receive_loop(ws) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            print("unexpected binary from server", flush=True)
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            print(raw, flush=True)
            continue
        mtype = msg.get("type")
        if mtype == "prediction":
            trig = msg.get("trigger")
            line = (
                f"t={msg.get('t_sec')} {msg.get('predicted_class')} "
                f"conf={msg.get('confidence'):.3f} inf_ms={msg.get('inference_ms', 0):.1f}"
            )
            if trig:
                line += f" TRIGGER {msg.get('detection')}"
            print(line, flush=True)
        elif mtype == "error":
            print("ERR", msg.get("message"), flush=True)
        elif mtype in ("ready", "reconfigured"):
            print(msg, flush=True)
        else:
            print(msg, flush=True)


async def _run(args: argparse.Namespace) -> None:
    mel_cfg = _fetch_mel_config(args.api_url)
    ws_base = _rest_to_ws_base(args.api_url)
    ws_url = f"{ws_base}/ws/kws-logmel"
    targets = parse_target_labels(args.targets)
    ws_payload = {
        "confidence_threshold": args.confidence,
        "refractory_sec": args.refractory,
        "target_labels": targets,
    }

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _t, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print("audio status:", status, flush=True)
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        callback=_callback,
        blocksize=args.blocksize,
    )

    stream.start()
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            await ws.send(json.dumps(ws_payload))
            first = await ws.recv()
            if isinstance(first, bytes):
                raise RuntimeError("expected JSON hello, got binary")
            hello = json.loads(first)
            if hello.get("type") != "ready":
                raise RuntimeError(f"expected ready, got: {hello}")

            print("connected", hello, flush=True)

            recv_task = asyncio.create_task(_receive_loop(ws))

            buf = np.array([], dtype=np.float32)
            stream_start: float | None = None
            last_sent = 0.0

            try:
                while True:
                    drained = False
                    while True:
                        try:
                            chunk = audio_q.get_nowait()
                            drained = True
                            buf = np.concatenate([buf, chunk.reshape(-1)])
                        except queue.Empty:
                            break
                    max_keep = int(16000 * args.buffer_sec)
                    if buf.size > max_keep:
                        buf = buf[-max_keep:]

                    now = time.monotonic()
                    if buf.size < 16000:
                        if not drained:
                            await asyncio.sleep(0.02)
                        continue

                    if stream_start is None:
                        stream_start = now

                    if now - last_sent < args.stride:
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
                        print("prep:", prep_err, flush=True)
                        continue

                    await ws.send(npz_bytes)
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
    finally:
        stream.stop()
        stream.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Stream log-mel NPZ to /ws/kws-logmel")
    p.add_argument("--api-url", default="http://localhost:8000", help="REST base URL")
    p.add_argument("--stride", type=float, default=0.25, help="Seconds between NPZ sends")
    p.add_argument("--refractory", type=float, default=0.8, help="Refractory period (seconds)")
    p.add_argument("--confidence", type=float, default=0.55, help="Detection confidence threshold")
    p.add_argument("--targets", default=DEFAULT_TARGET_LABELS, help="Comma-separated target labels")
    p.add_argument("--blocksize", type=int, default=4096, help="sounddevice block size")
    p.add_argument("--buffer-sec", type=float, default=4.0, help="Max audio ring buffer (seconds)")
    args = p.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nexit", flush=True)


if __name__ == "__main__":
    main()
