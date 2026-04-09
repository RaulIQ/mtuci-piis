from pydantic import BaseModel


class PredictAudioRequest(BaseModel):
    audio_base64: str


class PredictSpectrogramRequest(BaseModel):
    log_mels: list[list[float]]


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

