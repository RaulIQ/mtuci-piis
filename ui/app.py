import os

import requests
import streamlit as st


st.set_page_config(page_title="KWS Demo", layout="centered")
st.title("KWS Inference UI")
st.caption("Upload WAV and run single-shot or streaming-like KWS inference.")

api_url = os.getenv("API_URL", "http://localhost:8000")
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

if audio_bytes is not None:
    mode = st.radio(
        "Mode",
        ["Single prediction", "Sliding window stream simulation"],
        index=0,
    )

    if mode == "Single prediction":
        if st.button("Predict"):
            try:
                files = {"file": (audio_name, audio_bytes, "audio/wav")}
                response = requests.post(f"{api_url}/predict", files=files, timeout=30)
                if response.ok:
                    data = response.json()
                    st.success("Prediction complete")
                    st.json(data)
                else:
                    st.error(f"Request failed: {response.status_code}")
                    st.code(response.text)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error: {exc}")
    else:
        st.markdown("### Sliding window parameters")
        stride_sec = st.slider("Stride (seconds)", min_value=0.10, max_value=0.50, value=0.25, step=0.05)
        refractory_sec = st.slider("Refractory period (seconds)", min_value=0.20, max_value=2.00, value=0.80, step=0.10)
        confidence_threshold = st.slider("Confidence threshold", min_value=0.10, max_value=0.99, value=0.55, step=0.01)
        target_labels_raw = st.text_input(
            "Target labels (comma-separated)",
            value="one,two,three,four,five,six,seven,eight,nine",
        )
        target_labels = {x.strip() for x in target_labels_raw.split(",") if x.strip()}

        if st.button("Run stream simulation"):
            try:
                files = {"file": (audio_name, audio_bytes, "audio/wav")}
                data = {
                    "stride_sec": str(stride_sec),
                    "refractory_sec": str(refractory_sec),
                    "confidence_threshold": str(confidence_threshold),
                    "target_labels": ",".join(sorted(target_labels)),
                }
                response = requests.post(
                    f"{api_url}/predict-stream",
                    files=files,
                    data=data,
                    timeout=60,
                )
                if not response.ok:
                    st.error(f"Request failed: {response.status_code}")
                    st.code(response.text)
                else:
                    payload = response.json()
                    detections = payload.get("detections", [])
                    rows = payload.get("window_predictions", [])

                    st.success("Stream simulation completed")
                    st.write(f"Windows processed: {payload.get('windows_processed', len(rows))}")
                    st.write(f"Detections after refractory: {payload.get('detections_count', len(detections))}")

                    if detections:
                        st.markdown("### Final detections")
                        st.dataframe(detections, use_container_width=True)
                    else:
                        st.info("No detections passed target/confidence/refractory conditions.")

                    st.markdown("### Window-by-window predictions")
                    st.dataframe(rows, use_container_width=True, height=350)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error: {exc}")

