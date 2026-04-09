import streamlit as st

from components.offline_inference import render_offline_inference
from services.api import get_api_url


st.set_page_config(page_title="KWS Demo", layout="centered")
st.title("KWS Inference UI")
st.caption("Upload WAV and run single-shot or streaming-like KWS inference.")

api_url = get_api_url()
st.write(f"API endpoint: `{api_url}`")

st.markdown("### Audio source")
uploaded = st.file_uploader("Upload .wav", type=["wav"])
recorded = st.audio_input("Or record audio from microphone")

audio_bytes = None
audio_name = "recorded.wav"

if recorded is not None:
    audio_bytes = recorded.getvalue()
    audio_name = "recorded.wav"
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        "Download recorded WAV",
        data=audio_bytes,
        file_name=audio_name,
        mime="audio/wav",
    )
elif uploaded is not None:
    audio_bytes = uploaded.getvalue()
    audio_name = uploaded.name

render_offline_inference(
    api_url=api_url,
    audio_bytes=audio_bytes,
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
)

