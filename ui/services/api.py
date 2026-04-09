import os

import requests

from helpers.labels import format_target_labels


def get_api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8000")


def fetch_mel_config(api_url: str, timeout: int = 10) -> dict:
    r = requests.get(f"{api_url}/model/mel-config", timeout=timeout)
    r.raise_for_status()
    return r.json()


def predict_audio(api_url: str, audio_name: str, audio_bytes: bytes, timeout: int = 30) -> requests.Response:
    files = {"file": (audio_name, audio_bytes, "audio/wav")}
    return requests.post(f"{api_url}/predict", files=files, timeout=timeout)


def predict_logmel_npy(api_url: str, npy_bytes: bytes, timeout: int = 30) -> requests.Response:
    files = {"file": ("logmel.npy", npy_bytes, "application/octet-stream")}
    return requests.post(f"{api_url}/predict-logmel", files=files, timeout=timeout)


def predict_stream(
    api_url: str,
    audio_name: str,
    audio_bytes: bytes,
    *,
    stride_sec: float,
    refractory_sec: float,
    confidence_threshold: float,
    target_labels_raw: str,
    timeout: int = 60,
) -> requests.Response:
    files = {"file": (audio_name, audio_bytes, "audio/wav")}
    data = {
        "stride_sec": str(stride_sec),
        "refractory_sec": str(refractory_sec),
        "confidence_threshold": str(confidence_threshold),
        "target_labels": format_target_labels(target_labels_raw),
    }
    return requests.post(f"{api_url}/predict-stream", files=files, data=data, timeout=timeout)
