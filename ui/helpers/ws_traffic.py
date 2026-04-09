"""Thread-safe WebSocket traffic counters (application payload bytes, UTF-8 for text)."""

from __future__ import annotations

import threading


class WsTrafficCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.bytes_up = 0
        self.bytes_down = 0

    def add_up(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.bytes_up += n

    def add_down(self, n: int) -> None:
        if n <= 0:
            return
        with self._lock:
            self.bytes_down += n

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.bytes_up, self.bytes_down

    def reset(self) -> None:
        with self._lock:
            self.bytes_up = 0
            self.bytes_down = 0


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024 / 1024:.2f} MiB"


def format_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec < 0:
        return "—"
    return f"{format_bytes(int(bytes_per_sec))}/s"
