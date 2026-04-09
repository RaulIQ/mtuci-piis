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
        prepared = prepare_audio(chunk)
        t_list.append(start / float(sr))
        if prepared is None:
            silence_flags.append(True)
            rows.append(placeholder.copy())
        else:
            wave = to_model_waveform(prepared, sr=sr)
            x = torch.from_numpy(wave).unsqueeze(0).float()
            with torch.no_grad():
                lm = normalized_log_mel(mel, x)
            silence_flags.append(False)
            rows.append(lm.cpu().numpy().astype(np.float32))

    stack = np.stack(rows, axis=0)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        t_sec=np.asarray(t_list, dtype=np.float64),
        log_mel=stack,
        is_silence=np.asarray(silence_flags, dtype=np.bool_),
    )
    return buf.getvalue(), None
