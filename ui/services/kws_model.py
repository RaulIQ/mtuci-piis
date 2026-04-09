import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms

_MODEL_PATH = Path(__file__).resolve().parents[2] / "app" / "model.py"
_PROJECT_ROOT = _MODEL_PATH.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location("kws_app_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load model helpers from {_MODEL_PATH}")
_KWS_APP_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KWS_APP_MODEL)

prepare_audio = _KWS_APP_MODEL.prepare_audio
to_model_waveform = _KWS_APP_MODEL.to_model_waveform


def build_mel_transform(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )


def compute_canonical_log_mels(
    audio: np.ndarray,
    sr: int,
    mel_tf: torchaudio.transforms.MelSpectrogram,
    frames: int,
) -> np.ndarray | None:
    prepared = prepare_audio(audio)
    if prepared is None:
        return None

    wave = to_model_waveform(prepared, sr=sr)
    x = torch.from_numpy(wave).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        mel = mel_tf(x)
        log_mels = torch.log(mel + 1e-6).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    if log_mels.shape[1] > frames:
        log_mels = log_mels[:, :frames]
    elif log_mels.shape[1] < frames:
        pad_w = frames - log_mels.shape[1]
        log_mels = np.pad(log_mels, ((0, 0), (0, pad_w)))
    return log_mels
