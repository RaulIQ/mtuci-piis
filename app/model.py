import time
from pathlib import Path

import librosa
import numpy as np
import torch
from models.small_kws_cnn import SmallKwsCNN


SR = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160


def extract_logmel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    elif len(audio) > sr:
        audio = audio[:sr]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


def prepare_audio(y: np.ndarray, silence_peak: float = 0.02, silence_rms: float = 0.006):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y))))

    if peak < silence_peak and rms < silence_rms:
        return None
    y = (y / (peak + 1e-8)) * 0.95
    return y


class KwsInferenceService:
    def __init__(self, model_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self._resolve_model_path(model_path)
        self.model, self.labels = self._load_checkpoint(self.model_path)
        self.model_version = self.model_path.name

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path)
        if path.is_file():
            return path
        # Preferred artifact for inference
        fallback = Path("artifacts/colab_kws_cnn.pt")
        if fallback.is_file():
            return fallback
        # Legacy fallback
        legacy = Path("artifacts/kws_cnn.pt")
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"Model file not found: {model_path}")

    def _load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        labels = list(checkpoint["labels"])
        model = SmallKwsCNN(num_classes=len(labels)).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, labels

    def predict(self, audio: np.ndarray, sr: int = SR):
        prepared = prepare_audio(audio)
        if prepared is None:
            ranked = {label: (1.0 if label == "silence" else 0.0) for label in self.labels}
            return {
                "predicted_class": "silence",
                "confidence": 1.0,
                "top_k": self._top_k(ranked, 5),
                "inference_ms": 0.0,
            }

        feature = extract_logmel(prepared, sr=sr)
        x = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        infer_ms = (time.perf_counter() - start) * 1000

        order = np.argsort(-probs)
        ranked = {self.labels[int(i)]: float(probs[int(i)]) for i in order}
        best = self.labels[int(order[0])]
        return {
            "predicted_class": best,
            "confidence": float(probs[int(order[0])]),
            "top_k": self._top_k(ranked, 5),
            "inference_ms": infer_ms,
        }

    @staticmethod
    def _top_k(ranked: dict[str, float], k: int) -> list[dict[str, float]]:
        out = []
        for i, (label, score) in enumerate(ranked.items()):
            if i >= k:
                break
            out.append({label: float(score)})
        return out

