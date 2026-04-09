import streamlit as st

st.set_page_config(page_title="KWS", layout="centered")

demo = st.Page(
    "pages/kws_demo.py",
    title="KWS Demo",
    icon="📊",
    default=True,
)
realtime = st.Page(
    "pages/realtime_kws.py",
    title="Realtime KWS",
    icon="🎙️",
)

st.navigation([demo, realtime]).run()
