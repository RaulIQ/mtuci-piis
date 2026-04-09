#!/usr/bin/env python3
"""Realtime KWS: capture 16 kHz mono mic, build log-mel NPZ locally, stream to ``/ws/kws-logmel``."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

_UI_ROOT = Path(__file__).resolve().parent.parent / "ui"
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from helpers.labels import DEFAULT_TARGET_LABELS, parse_target_labels
from helpers.realtime_logmel_runner import run_realtime_logmel_in_thread


def _print_sink(msg: dict) -> None:
    mtype = msg.get("type")
    if mtype == "prediction":
        trig = msg.get("trigger")
        line = (
            f"t={msg.get('t_sec')} {msg.get('predicted_class')} "
            f"conf={float(msg.get('confidence') or 0):.3f} "
            f"inf_ms={float(msg.get('inference_ms') or 0):.1f}"
        )
        if trig:
            line += f" TRIGGER {msg.get('detection')}"
        print(line, flush=True)
    elif mtype == "error":
        print("ERR", msg.get("message"), flush=True)
    elif mtype in ("ready", "reconfigured"):
        print(msg, flush=True)
    elif mtype == "runner_error":
        print("runner:", msg.get("message"), flush=True)
    elif mtype == "prep_warning":
        print("prep:", msg.get("message"), flush=True)
    elif mtype == "audio_status":
        print("audio:", msg.get("message"), flush=True)
    elif mtype == "runner_stopped":
        print("stopped", flush=True)
    else:
        print(msg, flush=True)


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

    stop_event = threading.Event()

    def _on_sigint(_sig, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    t = run_realtime_logmel_in_thread(
        api_url=args.api_url,
        confidence=args.confidence,
        refractory=args.refractory,
        target_labels=parse_target_labels(args.targets),
        stride=args.stride,
        buffer_sec=args.buffer_sec,
        blocksize=args.blocksize,
        stop_event=stop_event,
        on_message=_print_sink,
    )
    t.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nexit", flush=True)
