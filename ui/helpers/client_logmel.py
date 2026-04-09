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

from kws_shared.waveform import prepare_audio, to_model_waveform
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
