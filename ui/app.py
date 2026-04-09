import streamlit as st

st.set_page_config(page_title="KWS", layout="centered")

server_inference = st.Page(
    "pages/kws_inference_server.py",
    title="KWS · server (WAV)",
    icon="🖥️",
    default=True,
)
edge_inference = st.Page(
    "pages/kws_inference_edge.py",
    title="KWS · edge (log-mel)",
    icon="⚡",
)
realtime = st.Page(
    "pages/realtime_kws.py",
    title="Realtime KWS",
    icon="🎙️",
)
realtime_edge = st.Page(
    "pages/realtime_kws_edge.py",
    title="Realtime KWS · log-mel",
    icon="📡",
)

st.navigation([server_inference, edge_inference, realtime, realtime_edge]).run()
