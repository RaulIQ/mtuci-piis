import numpy as np
import librosa

SR = 16000


def to_model_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)))
    elif len(y) > SR:
        y = y[:SR]
    return y


def prepare_audio(y: np.ndarray, silence_peak: float = 0.02, silence_rms: float = 0.006):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y))))

    if peak < silence_peak and rms < silence_rms:
        return None
    y = (y / (peak + 1e-8)) * 0.95
    return y
