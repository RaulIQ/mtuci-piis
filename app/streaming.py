from dataclasses import dataclass

import numpy as np

from app.model import KwsInferenceService, SR


@dataclass
class StreamParams:
    stride_sec: float = 0.25
    refractory_sec: float = 0.8
    confidence_threshold: float = 0.55
    target_labels: set[str] | None = None


class SlidingWindowProcessor:
    def __init__(self, inference_service: KwsInferenceService) -> None:
        self.inference = inference_service

    def run(self, audio: np.ndarray, sr: int, params: StreamParams) -> dict:
        if params.target_labels is None:
            params.target_labels = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

        # Bring the stream to model sample rate
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            sr = SR

        window_sec = 1.0
        win_size = int(window_sec * sr)
        step = max(1, int(params.stride_sec * sr))

        if len(audio) <= win_size:
            step = win_size

        windows = []
        detections = []
        refractory_until = -1.0

        for start in range(0, max(1, len(audio) - win_size + 1), step):
            end = start + win_size
            chunk = audio[start:end]
            if len(chunk) < win_size:
                chunk = np.pad(chunk, (0, win_size - len(chunk)))

            pred = self.inference.predict(chunk, sr=sr)
            t_sec = start / sr
            label = pred["predicted_class"]
            conf = float(pred["confidence"])

            windows.append(
                {
                    "t_sec": round(t_sec, 3),
                    "predicted_class": label,
                    "confidence": round(conf, 4),
                }
            )

            is_trigger = (
                label in params.target_labels
                and conf >= params.confidence_threshold
                and t_sec >= refractory_until
            )
            if is_trigger:
                detections.append(
                    {
                        "t_sec": round(t_sec, 3),
                        "label": label,
                        "confidence": round(conf, 4),
                    }
                )
                refractory_until = t_sec + params.refractory_sec

        return {
            "windows_processed": len(windows),
            "detections_count": len(detections),
            "detections": detections,
            "window_predictions": windows,
        }

