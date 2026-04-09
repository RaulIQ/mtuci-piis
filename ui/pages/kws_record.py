import sys
from pathlib import Path

import streamlit as st

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from components.offline_inference import render_offline_inference
from helpers.wav_duration import wav_duration_seconds
from services.api import get_api_url

st.set_page_config(page_title="KWS Record", layout="centered")
st.title("KWS record (~1 s)")
st.caption(
    "Record about one second from the microphone. The UI builds the same log-mel as the "
    "model, then sends it to /predict-logmel (no raw WAV over inference for this page)."
)

api_url = get_api_url()
st.write(f"API endpoint: `{api_url}`")

st.markdown("### Microphone")
recorded = st.audio_input("Record ~1 s of audio")

audio_bytes = None
inference_audio_bytes = None
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
        if duration_sec > 1.0:
            st.warning("Inference is disabled: audio must be at most 1.00 s.")
        else:
            inference_audio_bytes = audio_bytes
            if duration_sec < 0.85:
                st.info("Aim for about 1 s for best alignment with the model.")
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
    stream_params_header="",
    stream_button_label="",
    stream_success_text="",
    windows_label="",
    detections_label="",
    empty_detections_text="",
    detections_header=None,
    window_predictions_header=None,
    widget_key_prefix="kws_record_",
    forced_mode="predict",
    use_client_logmel=True,
)
