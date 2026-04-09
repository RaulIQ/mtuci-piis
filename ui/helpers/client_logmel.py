import io
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kws_shared.waveform import SR, prepare_audio, to_model_waveform
from models.resnet_kws import normalized_log_mel


def _one_sec_chunk_to_logmel_row(
    chunk: np.ndarray,
    *,
    mel: torchaudio.transforms.MelSpectrogram,
    placeholder: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, bool]:
    """One second of PCM at ``sr`` → normalized log-mel [1,1,n_mels,time] or silence placeholder."""
    win_size = int(1.0 * sr)
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if len(chunk) < win_size:
        chunk = np.pad(chunk, (0, win_size - len(chunk)))
    elif len(chunk) > win_size:
        chunk = chunk[:win_size]
    prepared = prepare_audio(chunk)
    if prepared is None:
        return placeholder.copy(), True
    wave = to_model_waveform(prepared, sr=sr)
    x = torch.from_numpy(wave).unsqueeze(0).float()
    with torch.no_grad():
        lm = normalized_log_mel(mel, x)
    return lm.cpu().numpy().astype(np.float32), False


def wav_bytes_to_normalized_logmel_npy(
    wav_bytes: bytes,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> tuple[bytes, str | None]:
    """Match server ``ResNetKWS``: MelSpectrogram → log → per-utterance mean/std → .npy [1,1,n_mels,time]."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    prepared = prepare_audio(y)
    if prepared is None:
        return b"", "signal too quiet (treated as silence)"

    wave = to_model_waveform(prepared, sr=sr)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    x = torch.from_numpy(wave).unsqueeze(0).float()
    with torch.no_grad():
        lm = normalized_log_mel(mel, x)
    arr = lm.cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue(), None


def pcm16k_window_to_logmel_npz_bytes(
    y_16k: np.ndarray,
    t_sec: float,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> tuple[bytes, str | None]:
    """
    Single 1 s window at model rate → NPZ with N=1 (``t_sec``, ``log_mel``, ``is_silence``).
    Same layout as chunks from ``wav_bytes_to_sliding_logmel_npz_bytes`` for ``/ws/kws-logmel``.
    """
    if sample_rate != SR:
        return b"", "mel sample_rate must match model SR (16 kHz)"

    win_size = int(1.0 * SR)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    with torch.no_grad():
        ref_lm = normalized_log_mel(mel, torch.zeros(1, win_size, dtype=torch.float32))
    placeholder = np.zeros(ref_lm.shape, dtype=np.float32)

    row, is_silence = _one_sec_chunk_to_logmel_row(
        y_16k, mel=mel, placeholder=placeholder, sr=SR
    )
    stack = np.expand_dims(row, axis=0)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        t_sec=np.asarray([t_sec], dtype=np.float64),
        log_mel=stack,
        is_silence=np.asarray([is_silence], dtype=np.bool_),
    )
    return buf.getvalue(), None


def wav_bytes_to_sliding_logmel_npz_bytes(
    wav_bytes: bytes,
    *,
    stride_sec: float,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> tuple[bytes, str | None]:
    """Sliding 1 s windows aligned with ``SlidingWindowProcessor.run``; NPZ for ``/predict-stream-logmel``."""
    y, sr_in = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = SR
    if sr_in != sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr).astype(np.float32)

    if sample_rate != sr:
        return b"", "mel sample_rate must match model SR (16 kHz)"

    win_size = int(1.0 * sr)
    step = max(1, int(stride_sec * sr))
    if len(y) <= win_size:
        step = win_size

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    with torch.no_grad():
        ref_lm = normalized_log_mel(mel, torch.zeros(1, win_size, dtype=torch.float32))
    placeholder = np.zeros(ref_lm.shape, dtype=np.float32)

    rows: list[np.ndarray] = []
    t_list: list[float] = []
    silence_flags: list[bool] = []

    for start in range(0, max(1, len(y) - win_size + 1), step):
        end = start + win_size
        chunk = y[start:end]
        if len(chunk) < win_size:
            chunk = np.pad(chunk, (0, win_size - len(chunk)))
        row, is_sil = _one_sec_chunk_to_logmel_row(chunk, mel=mel, placeholder=placeholder, sr=sr)
        t_list.append(start / float(sr))
        silence_flags.append(is_sil)
        rows.append(row)

    stack = np.stack(rows, axis=0)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        t_sec=np.asarray(t_list, dtype=np.float64),
        log_mel=stack,
        is_silence=np.asarray(silence_flags, dtype=np.bool_),
    )
    return buf.getvalue(), None
