# MTUCI PIIS - KWS Inference API

## Общее описание проекта

Проект представляет собой ML-сервис для распознавания голосовых команд (Keyword Spotting, KWS) с фокусом на команды `one`-`nine`.

Основная идея архитектуры:
- аудио поступает от клиентского устройства (в целевом сценарии - ESP32);
- инференс выполняется на сервере в API-сервисе;
- сервис поддерживает одиночные и потоковые запросы;
- предусмотрены логирование запросов и метрики для мониторинга.

Ключевые компоненты:
- API инференса на `FastAPI` (`app/`);
- UI на `Streamlit` (`ui/`) для проверки и демонстрации API;
- артефакт модели в `artifacts/`;
- журнал запросов в SQLite (`DB_PATH`);
- метрики в формате Prometheus (`/metrics`).

## Установка и запуск (Docker-first)

### Требования

- `Docker` и `Docker Compose` (обязательно для запуска API);
- `Python 3.11+` и `pip` (для запуска UI на хосте).

### 1) Запуск API в Docker

Из корня репозитория:

```bash
docker compose up --build
```

После старта API доступен по адресу:
- `http://localhost:8000/docs` - Swagger UI;
- `http://localhost:8000/health` - liveness;
- `http://localhost:8000/ready` - readiness;
- `http://localhost:8000/metrics` - метрики Prometheus.

### 2) Запуск UI отдельно (опционально, как example-клиент)

UI не является обязательной частью контейнерного запуска API и используется как демонстрационный интерфейс для проверки работы сервиса.

```bash
pip install -r requirements.txt
streamlit run ui/app.py --server.port 8501
```

UI будет доступен на `http://localhost:8501`.
По умолчанию UI обращается к API на `http://localhost:8000` (через `API_URL`).

## Примеры использования

### Ручной запрос через `curl` (WAV-файл)

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample.wav"
```

### Ручной запрос через JSON base64

```bash
python3 - <<'PY'
import base64
import requests

with open("sample.wav", "rb") as f:
    payload = {"audio_base64": base64.b64encode(f.read()).decode("utf-8")}

r = requests.post("http://localhost:8000/predict-base64", json=payload, timeout=30)
print(r.status_code)
print(r.json())
PY
```

### Взаимодействие через UI

1. Убедитесь, что API запущен в Docker (`docker compose up --build`).
2. Запустите `Streamlit` UI отдельной командой.
3. Откройте `http://localhost:8501`.
4. Выберите нужную страницу:
   - `KWS · server (WAV)` - отправка WAV в API;
   - `KWS · edge (log-mel)` - отправка признаков log-mel;
   - `Realtime KWS` / `Realtime KWS · log-mel` - потоковые сценарии.

### Пример ответа API

```json
{
  "predicted_class": "one",
  "confidence": 0.92,
  "latency_ms": 23.4,
  "model_version": "kws_resnet.pt",
  "top_k": [
    {"one": 0.92},
    {"unknown": 0.04},
    {"silence": 0.02}
  ]
}
```

## Эндпоинты API

- `GET /health` - проверка, что сервис запущен;
- `GET /ready` - проверка готовности модели;
- `POST /predict` - предсказание по WAV-файлу;
- `POST /predict-logmel` - предсказание по log-mel (`.npy`);
- `POST /predict-base64` - предсказание по base64-аудио;
- `POST /predict-stream` - потоковый режим для WAV;
- `POST /predict-stream-logmel` - потоковый режим для log-mel (`.npz`);
- `GET /metrics` - экспорт метрик Prometheus.

## Документация и архитектура

- Архитектура и обоснование: `docs/lab1.md`
- Масштабирование и метрики (ЛР2): `docs/lab2.md`
- Обучение/модель (ЛР3): `docs/lab3.md`
- Интеграция и развёртывание (ЛР4): `docs/lab4.md`

