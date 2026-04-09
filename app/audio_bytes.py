import io
from typing import Any

import librosa


def load_mono_from_bytes(raw_bytes: bytes, *, empty_detail: str) -> tuple[Any, int]:
    if not raw_bytes:
        raise ValueError(empty_detail)
    return librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
