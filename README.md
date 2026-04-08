# MTUCI PIIS - KWS Inference API

## Project Overview
Simple Keyword Spotting inference service based on Lab 3 model.

**Target deployment (ESP32):** In the intended setup, audio is captured on an **ESP32** and sent to the API—not from the browser. **Preprocessing** (e.g. feature extraction) can run **either on the ESP32 or on the server**, depending on the hybrid vs cloud-only mode. **Neural network inference** runs **on the server** (this service). The Streamlit UI and browser/curl examples are for development and demos.

Core components:
- Inference API (`FastAPI`)
- Web UI (`Streamlit`)
- Model artifact (`artifacts/kws_cnn.pt`, fallback from `artifacts/kwc_cnn.pt`)
- Request logging (SQLite)

## Minimum Requirements
- Docker
- Docker Compose

## Run

```bash
docker compose up --build
```

API is available at:
- `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## API Endpoints
- `GET /health` - liveness
- `GET /ready` - model loaded status and labels
- `POST /predict` - WAV file upload (multipart)
- `POST /predict-base64` - WAV bytes in base64 JSON
- `POST /predict-stream` - server-side sliding window + refractory
- `GET /metrics` - basic service metrics

## How to Send Audio

### Option 1: Upload WAV file

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.wav"
```

### Option 2: Send base64 WAV in JSON

```bash
python3 - <<'PY'
import base64, json, requests

with open("sample.wav", "rb") as f:
    payload = {"audio_base64": base64.b64encode(f.read()).decode("utf-8")}

r = requests.post("http://localhost:8000/predict-base64", json=payload, timeout=30)
print(r.status_code, r.json())
PY
```

## Response Example

```json
{
  "predicted_class": "stop",
  "confidence": 0.92,
  "latency_ms": 23.4,
  "model_version": "kws_cnn.pt",
  "top_k": [
    {"stop": 0.92},
    {"unknown": 0.04},
    {"silence": 0.02}
  ]
}
```

## Documentation
- Lab 4 report: `docs/lab4.md`

