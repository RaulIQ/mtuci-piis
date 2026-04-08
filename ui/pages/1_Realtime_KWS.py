import base64
import html
import io
import os
import time
from datetime import datetime
from queue import Empty

import numpy as np
import requests
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer


st.set_page_config(page_title="Realtime KWS", layout="wide")
st.title("Realtime KWS")
st.caption("Speak to microphone and watch detected digits in real time.")

api_url = os.getenv("API_URL", "http://localhost:8000")
st.write(f"API endpoint: `{api_url}`")

col1, col2 = st.columns(2)
with col1:
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.99, 0.55, 0.01)
with col2:
    refractory_sec = st.slider("Refractory (sec)", 0.2, 2.0, 0.8, 0.1)

# Fixed by training setup: model expects 1-second chunks.
chunk_sec = 1.0

target_labels_raw = st.text_input(
    "Target labels (comma-separated)",
    value="one,two,three,four,five,six,seven,eight,nine",
)
target_labels = {x.strip() for x in target_labels_raw.split(",") if x.strip()}

if "rt_buffer" not in st.session_state:
    st.session_state.rt_buffer = np.array([], dtype=np.float32)
if "rt_sr" not in st.session_state:
    st.session_state.rt_sr = 16000
if "rt_logs" not in st.session_state:
    st.session_state.rt_logs = []
if "rt_detections" not in st.session_state:
    st.session_state.rt_detections = []
if "rt_refractory_until" not in st.session_state:
    st.session_state.rt_refractory_until = 0.0
if "rt_frame_idx" not in st.session_state:
    st.session_state.rt_frame_idx = 0

if st.button("Clear logs"):
    st.session_state.rt_logs = []
    st.session_state.rt_detections = []
    st.session_state.rt_refractory_until = 0.0
    st.session_state.rt_frame_idx = 0

ctx = webrtc_streamer(
    key="realtime-kws",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=128,
    async_processing=True,
)

detection_section = st.container()
with detection_section:
    st.markdown("### Detected digits")
    detection_content = st.empty()

logs_section = st.container()
with logs_section:
    st.markdown("### Live logs")
    logs_content = st.empty()

# Fixed realtime view behavior: auto-update on each incoming chunk.
visible_log_lines = 20
compact_logs = False


def to_mono_float32(audio_frame):
    arr = audio_frame.to_ndarray()
    # Possible shapes: (channels, samples) or (samples, channels)
    if arr.ndim == 2:
        if arr.shape[0] <= 2:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)

    # Integer PCM -> [-1, 1]
    if np.max(np.abs(arr)) > 1.5:
        arr = arr / 32768.0
    return arr


def send_chunk_for_prediction(chunk, sr):
    wav_buf = io.BytesIO()
    sf.write(wav_buf, chunk, sr, format="WAV")
    payload = {"audio_base64": base64.b64encode(wav_buf.getvalue()).decode("utf-8")}
    r = requests.post(f"{api_url}/predict-base64", json=payload, timeout=15)
    if not r.ok:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    return r.json(), None


def format_log_line(item):
    if item.get("event") == "error":
        return (
            f"[{item.get('time')}] frame={item.get('frame_idx')} "
            f"ERROR: {item.get('message', '')}"
        )
    return (
        f"[{item.get('time')}] frame={item.get('frame_idx')} t={item.get('t_sec')}s "
        f"pred={item.get('predicted_class')} conf={item.get('confidence')} "
        f"req_ms={item.get('request_ms')} trigger={item.get('trigger')} "
        f"reason={item.get('reason')}"
    )


def render_live_views(target_labels_set, conf_threshold, compact, visible_lines):
    def render_scroll_box(lines, visible_count, box_id):
        dyn_id = f"{box_id}_{len(lines)}"
        if not lines:
            return f"""
<div id="{dyn_id}" style="
  height:{int(visible_count * 20)}px;
  overflow-y:auto;
  border:1px solid #444;
  border-radius:6px;
  padding:8px;
  background:#111;
  color:#ddd;
  white-space:pre;
  font-family:monospace;
  font-size:12px;
  line-height:1.35;">No data yet.</div>
<script>
  (function() {{
    const el = document.getElementById("{dyn_id}");
    if (el) el.scrollTop = el.scrollHeight;
  }})();
</script>
"""

        safe_text = html.escape("\n".join(lines))
        return f"""
<div id="{dyn_id}" style="
  height:{int(visible_count * 20)}px;
  overflow-y:auto;
  border:1px solid #444;
  border-radius:6px;
  padding:8px;
  background:#111;
  color:#ddd;
  white-space:pre;
  font-family:monospace;
  font-size:12px;
  line-height:1.35;">{safe_text}</div>
<script>
  (function() {{
    const el = document.getElementById("{dyn_id}");
    if (el) el.scrollTop = el.scrollHeight;
  }})();
</script>
"""

    # Detected digits as autoscroll log-like stream
    det_lines = [
        f"[{x.get('time')}] frame={x.get('frame_idx')} digit={x.get('digit')} conf={x.get('confidence')}"
        for x in st.session_state.rt_detections
    ]
    detection_html = render_scroll_box(det_lines, visible_lines, "detected_digits_box")
    with detection_content.container():
        components.html(detection_html, height=int(visible_lines * 20) + 18, scrolling=False)

    view = st.session_state.rt_logs
    if compact:
        view = [
            x
            for x in view
            if x.get("event") == "error"
            or (
                x.get("event") == "prediction"
                and x.get("predicted_class") in target_labels_set
                and float(x.get("confidence", 0.0)) >= conf_threshold
            )
        ]

    lines = [format_log_line(x) for x in view]
    logs_html = render_scroll_box(lines, visible_lines, "live_logs_box")
    with logs_content.container():
        components.html(logs_html, height=int(visible_lines * 20) + 18, scrolling=False)


if ctx.state.playing and ctx.audio_receiver:
    chunk_samples = int(chunk_sec * st.session_state.rt_sr)

    while ctx.state.playing:
        updated = False
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except Empty:
            continue
        if not frames:
            continue

        for frame in frames:
            st.session_state.rt_sr = frame.sample_rate or st.session_state.rt_sr
            y = to_mono_float32(frame)
            st.session_state.rt_buffer = np.concatenate([st.session_state.rt_buffer, y])

        sr = st.session_state.rt_sr
        chunk_samples = int(chunk_sec * sr)

        while len(st.session_state.rt_buffer) >= chunk_samples:
            chunk = st.session_state.rt_buffer[:chunk_samples]
            st.session_state.rt_buffer = st.session_state.rt_buffer[chunk_samples:]
            st.session_state.rt_frame_idx += 1

            started = time.perf_counter()
            data, err = send_chunk_for_prediction(chunk, sr)
            recv_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            t_sec = round(time.time(), 3)

            if err is not None:
                st.session_state.rt_logs.append(
                    {
                        "time": recv_time,
                        "frame_idx": st.session_state.rt_frame_idx,
                        "event": "error",
                        "message": err,
                    }
                )
                updated = True
                continue

            pred = data["predicted_class"]
            conf = float(data["confidence"])
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            st.session_state.rt_logs.append(
                {
                    "time": recv_time,
                    "frame_idx": st.session_state.rt_frame_idx,
                    "event": "prediction",
                    "t_sec": t_sec,
                    "predicted_class": pred,
                    "confidence": round(conf, 4),
                    "request_ms": elapsed_ms,
                    "trigger": False,
                    "reason": "",
                }
            )
            updated = True

            now_ts = time.time()
            trigger = False
            reason = "ok"
            if pred not in target_labels:
                reason = "non_target"
            elif conf < confidence_threshold:
                reason = "low_conf"
            elif now_ts < st.session_state.rt_refractory_until:
                reason = "refractory"
            else:
                trigger = True

            # enrich last prediction log with trigger decision
            st.session_state.rt_logs[-1]["trigger"] = trigger
            st.session_state.rt_logs[-1]["reason"] = reason

            if trigger:
                st.session_state.rt_detections.append(
                    {
                        "time": recv_time,
                        "frame_idx": st.session_state.rt_frame_idx,
                        "digit": pred,
                        "confidence": round(conf, 4),
                    }
                )
                st.session_state.rt_refractory_until = now_ts + refractory_sec
                updated = True

        if updated:
            render_live_views(
                target_labels_set=target_labels,
                conf_threshold=confidence_threshold,
                compact=compact_logs,
                visible_lines=visible_log_lines,
            )

else:
    st.info("Press START in the WebRTC widget to begin real-time capture.")
    render_live_views(
        target_labels_set=target_labels,
        conf_threshold=confidence_threshold,
        compact=compact_logs,
        visible_lines=visible_log_lines,
    )

