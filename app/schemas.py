from pydantic import BaseModel


class KwsMelConfigResponse(BaseModel):
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int


class PredictAudioRequest(BaseModel):
    audio_base64: str


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    latency_ms: float
    model_version: str
    top_k: list[dict[str, float]]


class StreamPredictResponse(BaseModel):
    windows_processed: int
    detections_count: int
    detections: list[dict[str, float | str]]
    window_predictions: list[dict[str, float | str]]

