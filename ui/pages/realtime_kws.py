import sys
from pathlib import Path

import streamlit as st

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from components.offline_inference import render_offline_inference
from components.realtime_widget import render_realtime_widget
from helpers.labels import DEFAULT_TARGET_LABELS, parse_target_labels
from services.api import get_api_url
from services.urls import build_ws_url

st.set_page_config(page_title="Realtime KWS", layout="centered")
st.title("Realtime KWS")

api_url = get_api_url()
ws_url = build_ws_url(api_url)

st.write(f"REST: `{api_url}` · WebSocket: `{ws_url}`")

st.markdown("### Параметры потока (отправляются при старте)")
col_a, col_b = st.columns(2)
with col_a:
    poll_interval_sec = st.slider(
        "Интервал опроса (с)",
        0.10,
        0.50,
        0.25,
        0.05,
        help="Раз в столько секунд сервер берёт последнюю 1 с аудио и вызывает модель.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.99, 0.55, 0.01)
with col_b:
    refractory_sec = st.slider("Refractory (sec)", 0.2, 2.0, 0.8, 0.1)
    target_labels_raw = st.text_input(
        "Target labels (comma-separated)",
        value=DEFAULT_TARGET_LABELS,
    )

target_labels = parse_target_labels(target_labels_raw)

ws_config = {
    "poll_interval_sec": poll_interval_sec,
    "confidence_threshold": confidence_threshold,
    "refractory_sec": refractory_sec,
    "target_labels": target_labels,
}

st.markdown("### Поток с микрофона (WebSocket)")
st.caption(
    "Нажмите «Старт» в блоке ниже, разрешите микрофон. "
    "Аудио уходит на сервер; ответы — в реальном времени. "
    "Нужен запущенный API (uvicorn) на том же хосте, что в REST URL."
)

render_realtime_widget(ws_url, ws_config)

st.markdown("---")
st.markdown("### Запись файлом (как раньше)")
st.caption("Одиночный Predict и predict-stream по завершённой записи — тот же API.")

recorded = st.audio_input("Записать для офлайн-режима")

audio_bytes = None
audio_name = "recorded.wav"

if recorded is not None:
    audio_bytes = recorded.getvalue()
    st.audio(audio_bytes, format="audio/wav")

render_offline_inference(
    api_url=api_url,
    audio_bytes=audio_bytes,
    audio_name=audio_name,
    mode_label="Режим",
    mode_options=["Один запрос (Predict)", "Скользящее окно на сервере"],
    predict_button_label="Predict",
    predict_success_text="Готово",
    request_failed_text="Ошибка",
    error_prefix="Ошибка",
    stream_params_header="### Параметры окна",
    stream_button_label="Запустить predict-stream",
    stream_success_text="Готово",
    windows_label="Окон",
    detections_label="Детекций",
    empty_detections_text="Нет детекций.",
    window_predictions_header="### По окнам",
    widget_key_prefix="off_",
)
