import os

import requests
import streamlit as st


st.set_page_config(page_title="KWS Demo", layout="centered")
st.title("KWS Inference UI")
st.caption("Upload WAV and get prediction from inference API.")

api_url = os.getenv("API_URL", "http://inference:8000")
st.write(f"API endpoint: `{api_url}`")

uploaded = st.file_uploader("Upload .wav", type=["wav"])
if uploaded is not None:
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

