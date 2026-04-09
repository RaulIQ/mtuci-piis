import streamlit as st

st.set_page_config(page_title="KWS", layout="centered")

demo = st.Page(
    "pages/kws_demo.py",
    title="KWS Demo",
    icon="📊",
    default=True,
)
record = st.Page(
    "pages/kws_record.py",
    title="KWS Record",
    icon="🎤",
)
realtime = st.Page(
    "pages/realtime_kws.py",
    title="Realtime KWS",
    icon="🎙️",
)

st.navigation([demo, record, realtime]).run()
