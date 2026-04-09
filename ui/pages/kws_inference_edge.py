import sys
from pathlib import Path

import streamlit as st

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from components.offline_inference import render_offline_inference
from helpers.wav_duration import wav_duration_seconds
from services.api import get_api_url

st.set_page_config(page_title="KWS — edge preprocessing", layout="centered")
st.title("KWS — client / edge preprocessing")
st.caption(
    "Log-mel (same recipe as the model) is built in the Streamlit app process, then only tensors "
    "are sent: ≤1 s → `/predict-logmel`; longer → sliding windows as `.npz` → `/predict-stream-logmel`."
)

api_url = get_api_url()
st.write(f"API endpoint: `{api_url}`")

st.markdown("### Microphone")
recorded = st.audio_input("Record audio (≤1 s for single shot, >1 s for stream mode)")

audio_bytes = None
inference_audio_bytes = None
forced_mode: str | None = None
audio_name = "recorded.wav"

if recorded is not None:
    audio_bytes = recorded.getvalue()
    st.download_button(
        "Download recorded WAV",
        data=audio_bytes,
        file_name=audio_name,
        mime="audio/wav",
    )
    duration_sec = wav_duration_seconds(audio_bytes)
    if duration_sec is not None:
        st.caption(f"Duration: {duration_sec:.2f} s.")
        inference_audio_bytes = audio_bytes
        forced_mode = "predict" if duration_sec <= 1.0 else "stream"
        if duration_sec <= 1.0 and duration_sec < 0.85:
            st.info("Aim for about 1 s for best alignment with the model.")
        if duration_sec > 1.0:
            st.info("Audio longer than 1 s — use «Run stream simulation» (client log-mel windows).")
    else:
        st.warning("Could not read WAV duration; inference is disabled.")

render_offline_inference(
    api_url=api_url,
    audio_bytes=inference_audio_bytes,
    audio_name=audio_name,
    mode_label="Mode",
    mode_options=["Single prediction", "Sliding window stream simulation"],
    predict_button_label="Predict",
    predict_success_text="Prediction complete",
    request_failed_text="Request failed",
    error_prefix="Error",
    stream_params_header="### Sliding window parameters",
    stream_button_label="Run stream simulation",
    stream_success_text="Stream simulation completed",
    windows_label="Windows processed",
    detections_label="Detections after refractory",
    empty_detections_text="No detections passed target/confidence/refractory conditions.",
    detections_header="### Final detections",
    window_predictions_header="### Window-by-window predictions",
    widget_key_prefix="kws_edge_",
    forced_mode=forced_mode,
    use_client_logmel=True,
)
