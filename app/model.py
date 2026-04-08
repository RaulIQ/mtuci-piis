import time
from pathlib import Path

import librosa
import numpy as np
import torch

from models.resnet_kws import ResNetKWS


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
        for cand in (Path("artifacts/kws_resnet.pt"), Path("artifacts/colab_kws_resnet.pt")):
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train and save ResNet to artifacts/kws_resnet.pt or set MODEL_PATH."
        )

    def _load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        labels = list(checkpoint["labels"])
        n_mels = int(checkpoint.get("n_mels", 128))
        n_fft = int(checkpoint.get("n_fft", 512))
        hop_length = int(checkpoint.get("hop_length", 160))
        sample_rate = int(checkpoint.get("sample_rate", SR))

        model = ResNetKWS(
            num_classes=len(labels),
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(self.device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
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

        wave = to_model_waveform(prepared, sr=sr)
        x = torch.from_numpy(wave).unsqueeze(0).to(self.device)

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
