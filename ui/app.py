import base64
import io
import os

import librosa
import numpy as np
import requests
import soundfile as sf
import streamlit as st


st.set_page_config(page_title="KWS Demo", layout="centered")
st.title("KWS Inference UI")
st.caption("Upload WAV and run single-shot or streaming-like KWS inference.")

api_url = os.getenv("API_URL", "http://inference:8000")
st.write(f"API endpoint: `{api_url}`")

uploaded = st.file_uploader("Upload .wav", type=["wav"])
if uploaded is not None:
    mode = st.radio(
        "Mode",
        ["Single prediction", "Sliding window stream simulation"],
        index=0,
    )

    if mode == "Single prediction":
        if st.button("Predict"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "audio/wav")}
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
                y, sr = librosa.load(io.BytesIO(uploaded.getvalue()), sr=16000, mono=True)
                window_sec = 1.0
                win_size = int(window_sec * sr)
                step = int(stride_sec * sr)

                if len(y) <= win_size:
                    st.info("Audio <= 1 sec, using single window.")
                    step = win_size

                rows = []
                detections = []
                refractory_until = -1.0

                for start in range(0, max(1, len(y) - win_size + 1), step):
                    end = start + win_size
                    chunk = y[start:end]
                    if len(chunk) < win_size:
                        chunk = np.pad(chunk, (0, win_size - len(chunk)))

                    # Convert chunk to WAV bytes, then send as base64 JSON
                    wav_buf = io.BytesIO()
                    sf.write(wav_buf, chunk, sr, format="WAV")
                    audio_b64 = base64.b64encode(wav_buf.getvalue()).decode("utf-8")

                    response = requests.post(
                        f"{api_url}/predict-base64",
                        json={"audio_base64": audio_b64},
                        timeout=30,
                    )
                    if not response.ok:
                        st.error(f"Request failed at window {start/sr:.2f}s: {response.status_code}")
                        st.code(response.text)
                        break

                    data = response.json()
                    t_sec = start / sr
                    pred = data["predicted_class"]
                    conf = float(data["confidence"])

                    rows.append(
                        {
                            "t_sec": round(t_sec, 3),
                            "predicted_class": pred,
                            "confidence": round(conf, 4),
                        }
                    )

                    is_trigger = (
                        pred in target_labels
                        and conf >= confidence_threshold
                        and t_sec >= refractory_until
                    )
                    if is_trigger:
                        detections.append(
                            {
                                "t_sec": round(t_sec, 3),
                                "label": pred,
                                "confidence": round(conf, 4),
                            }
                        )
                        refractory_until = t_sec + refractory_sec

                st.success("Stream simulation completed")
                st.write(f"Windows processed: {len(rows)}")
                st.write(f"Detections after refractory: {len(detections)}")

                if detections:
                    st.markdown("### Final detections")
                    st.dataframe(detections, use_container_width=True)
                else:
                    st.info("No detections passed target/confidence/refractory conditions.")

                st.markdown("### Window-by-window predictions")
                st.dataframe(rows, use_container_width=True, height=350)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error: {exc}")

