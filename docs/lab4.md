# Лабораторная работа 4

**ФИО:** Алимбеков Рауль Азатович  
**Группа:** БВТ2201  
**Тема:** Интеграция, инференс, мониторинг и контейнеризация ML-сервиса KWS

## Шаг 1. Разработка инференс-модуля
- Реализован микросервис предсказаний на `FastAPI`: `app/main.py`.
- Модель загружается при старте из артефакта ЛР3 (`artifacts/kws_cnn.pt`, поддержан fallback на `artifacts/kwc_cnn.pt`).
- Добавлены endpoint'ы:
  - `POST /predict` - инференс по загруженному WAV,
  - `GET /health` - liveness,
  - `GET /ready` - readiness и информация о модели.
- Предобработка сигнала соответствует схеме из ЛР3: `log-mel` и нормализация.

## Шаг 2. Интеграция компонентов
- Система объединяет:
  - инференс (`FastAPI` + PyTorch),
  - интерфейс (`Streamlit`),
  - хранилище логов (`SQLite`).
- Поток: UI/API-клиент -> inference -> логирование + метрики.
- Реализованы 2 формата передачи аудио:
  - `multipart/form-data` (`POST /predict`, поле `file`),
  - JSON base64 (`POST /predict-base64`, поле `audio_base64`).
- Такой формат подходит как для ручного теста WAV-файла, так и для интеграции с внешним клиентом/стрим-процессом.

## Шаг 3. Базовый мониторинг
- Логирование запросов реализовано в:
  - stdout-логи приложения,
  - SQLite таблица `requests` (время, файл, класс, confidence, latency, версия модели).
- Метрики:
  - `kws_requests_total` (с метками endpoint/status),
  - `kws_request_latency_ms`,
  - `kws_inference_latency_ms`.
- Экспорт метрик через `GET /metrics` (формат Prometheus).

## Шаг 4. Контейнеризация и оркестрация
- Создан `Dockerfile.inference` для API.
- Создан `Dockerfile.ui` для интерфейса.
- Создан `docker-compose.yml` для сервисов `inference` + `ui`:
  - переменные окружения,
  - healthcheck,
  - volume для SQLite логов,
  - зависимость UI от готовности API.
- Полный запуск:

```bash
docker compose up --build
```

## Шаг 5. Демонстрация
- Демо-сценарий:
  1. Запустить сервисы `docker compose up --build`.
  2. Открыть UI `http://localhost:8501`, отправить WAV и получить ответ модели.
  3. Проверить API через `http://localhost:8000/docs`.
  4. Отправить WAV в `POST /predict`.
  5. Отправить base64-аудио в `POST /predict-base64`.
  6. Проверить метрики на `http://localhost:8000/metrics`.

## Источники
- FastAPI docs
- PyTorch docs

