import html
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import streamlit as st

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from helpers.labels import DEFAULT_TARGET_LABELS, parse_target_labels
from helpers.realtime_logmel_runner import run_realtime_logmel_in_thread
from helpers.ws_traffic import WsTrafficCounter, format_bytes, format_rate
from services.api import get_api_url
from services.urls import build_ws_kws_logmel_url

st.set_page_config(page_title="Realtime KWS · edge", layout="centered")
st.title("Realtime KWS · Python log-mel")
st.caption(
    "Log-mel считается в процессе Streamlit (как на странице edge), на API уходят NPZ по WebSocket. "
    "Микрофон — у той машины, где запущен Streamlit (не в браузере)."
)
st.caption(
    "**Нагрузка на сеть (оценка на клиенте):** ниже — байты полезной нагрузки WS (JSON конфиг + сжатые NPZ и входящие JSON). "
    "**Мгновенно** — за интервал обновления фрагмента; **средняя за сессию** — всего / время с «Старт» (после «Стоп» показывается "
    "значение за последнюю сессию). Framing не считается; при том же шаге окна upload обычно ниже, чем у «Realtime KWS» (PCM)."
)

api_url = get_api_url()
ws_url = build_ws_kws_logmel_url(api_url)
st.write(f"REST: `{api_url}` · WebSocket: `{ws_url}`")

if "edge_rt_lines" not in st.session_state:
    st.session_state.edge_rt_lines = deque(maxlen=40)
if "edge_rt_center" not in st.session_state:
    st.session_state.edge_rt_center = None
if "edge_rt_bad_since" not in st.session_state:
    st.session_state.edge_rt_bad_since = None
if "edge_rt_running" not in st.session_state:
    st.session_state.edge_rt_running = False
if "_edge_traffic_prev" not in st.session_state:
    st.session_state._edge_traffic_prev = None

DIGIT_MAP = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "zero": "0",
}
BAD_MS = 2000.0


def _target_label_set(raw: str) -> set[str]:
    return {x.lower() for x in parse_target_labels(raw)}


def _apply_prediction_ui(pred_class: str, conf: float, conf_thr: float, targets: set[str]) -> None:
    low = (pred_class or "").lower()
    valid = low in targets and conf >= conf_thr
    if valid:
        st.session_state.edge_rt_center = DIGIT_MAP.get(low, low)
        st.session_state.edge_rt_bad_since = None
        return
    if low in ("silence", "unknown"):
        if st.session_state.edge_rt_center is not None:
            if st.session_state.edge_rt_bad_since is None:
                st.session_state.edge_rt_bad_since = time.monotonic() * 1000.0
            elif time.monotonic() * 1000.0 - st.session_state.edge_rt_bad_since >= BAD_MS:
                st.session_state.edge_rt_center = None
                st.session_state.edge_rt_bad_since = None
        else:
            st.session_state.edge_rt_bad_since = None
        return
    st.session_state.edge_rt_bad_since = None


def _append_line(text: str) -> None:
    st.session_state.edge_rt_lines.appendleft(text)


def _drain_queue_and_update(conf_thr: float, targets: set[str]) -> None:
    q = st.session_state.get("edge_rt_q")
    if q is None:
        return
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        mtype = msg.get("type")
        if mtype == "prediction":
            pred = msg.get("predicted_class", "")
            conf = float(msg.get("confidence") or 0.0)
            _apply_prediction_ui(pred, conf, conf_thr, targets)
            line = (
                f"t={msg.get('t_sec')} {pred} conf={conf:.3f} "
                f"inf_ms={msg.get('inference_ms', 0):.1f}"
            )
            if msg.get("trigger") and msg.get("detection"):
                line += f" TRIGGER {msg.get('detection')}"
            _append_line(line)
        elif mtype == "ready":
            _append_line(f"ready model={msg.get('model_version')}")
        elif mtype == "reconfigured":
            _append_line("reconfigured")
        elif mtype == "error":
            _append_line(f"ERR {msg.get('message')}")
        elif mtype == "runner_error":
            _append_line(f"runner: {msg.get('message')}")
            st.session_state.edge_rt_running = False
        elif mtype == "prep_warning":
            _append_line(f"prep: {msg.get('message')}")
        elif mtype == "audio_status":
            _append_line(f"audio: {msg.get('message')}")
        elif mtype == "runner_stopped":
            _append_line("остановлено")
            st.session_state.edge_rt_running = False


st.markdown("### Параметры")
col_a, col_b = st.columns(2)
with col_a:
    stride_sec = st.slider(
        "Шаг окна (с)",
        0.10,
        0.50,
        0.25,
        0.05,
        help="Как часто отправлять NPZ с последней 1 с аудио.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.99, 0.55, 0.01)
with col_b:
    refractory_sec = st.slider("Refractory (sec)", 0.2, 2.0, 0.8, 0.1)
    target_labels_raw = st.text_input(
        "Target labels (comma-separated)",
        value=DEFAULT_TARGET_LABELS,
    )
    blocksize = st.slider("sounddevice blocksize", 512, 8192, 4096, 256)
    buffer_sec = st.slider("Буфер аудио (с)", 1.0, 8.0, 4.0, 0.5)

targets = _target_label_set(target_labels_raw)
st.session_state["_edge_rt_conf_thr"] = confidence_threshold
st.session_state["_edge_rt_targets"] = targets

c1, c2, c3 = st.columns(3)
with c1:
    start = st.button("Старт", disabled=st.session_state.edge_rt_running, type="primary")
with c2:
    stop = st.button("Стоп", disabled=not st.session_state.edge_rt_running)
with c3:
    if st.button("Сбросить лог"):
        st.session_state.edge_rt_lines.clear()
        st.session_state.edge_rt_center = None
        st.session_state.edge_rt_bad_since = None

if start and not st.session_state.edge_rt_running:
    st.session_state.edge_rt_q = queue.Queue()
    st.session_state.edge_rt_stop = threading.Event()
    st.session_state.edge_rt_traffic = WsTrafficCounter()
    st.session_state.edge_rt_stream_t0 = time.monotonic()
    st.session_state.edge_rt_session_avg_up = None
    st.session_state.edge_rt_session_avg_down = None
    st.session_state._edge_traffic_prev = None
    st.session_state.edge_rt_lines.clear()
    st.session_state.edge_rt_center = None
    st.session_state.edge_rt_bad_since = None

    q: queue.Queue = st.session_state.edge_rt_q
    stop_ev: threading.Event = st.session_state.edge_rt_stop
    traf: WsTrafficCounter = st.session_state.edge_rt_traffic

    st.session_state.edge_rt_thread = run_realtime_logmel_in_thread(
        api_url=api_url,
        confidence=confidence_threshold,
        refractory=refractory_sec,
        target_labels=parse_target_labels(target_labels_raw),
        stride=stride_sec,
        buffer_sec=buffer_sec,
        blocksize=blocksize,
        stop_event=stop_ev,
        on_message=q.put,
        traffic=traf,
    )
    st.session_state.edge_rt_running = True
    st.rerun()

if stop and st.session_state.edge_rt_running:
    st.session_state.edge_rt_stop.set()
    th = st.session_state.get("edge_rt_thread")
    if th is not None:
        th.join(timeout=5.0)
    t0 = st.session_state.get("edge_rt_stream_t0")
    tc_stop = st.session_state.get("edge_rt_traffic")
    if t0 is not None and isinstance(tc_stop, WsTrafficCounter):
        elapsed = max(time.monotonic() - t0, 1e-9)
        u_stop, d_stop = tc_stop.snapshot()
        st.session_state.edge_rt_session_avg_up = u_stop / elapsed
        st.session_state.edge_rt_session_avg_down = d_stop / elapsed
    st.session_state.edge_rt_stream_t0 = None
    st.session_state.edge_rt_running = False
    st.rerun()


def _render_live_panel() -> None:
    conf_thr = float(st.session_state.get("_edge_rt_conf_thr", 0.55))
    tset = st.session_state.get("_edge_rt_targets") or set()
    if not isinstance(tset, set):
        tset = set(tset)
    _drain_queue_and_update(conf_thr, tset)

    tc = st.session_state.get("edge_rt_traffic")
    if tc is not None and isinstance(tc, WsTrafficCounter):
        up, down = tc.snapshot()
        now = time.monotonic()
        prev = st.session_state._edge_traffic_prev
        rate_up = 0.0
        rate_down = 0.0
        if prev is not None:
            pu, pd, pt = prev
            dt = now - pt
            if dt > 1e-6:
                rate_up = max(0.0, (up - pu) / dt)
                rate_down = max(0.0, (down - pd) / dt)
        st.session_state._edge_traffic_prev = (up, down, now)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Отправлено (WS)", format_bytes(up))
        with m2:
            st.metric("Принято (WS)", format_bytes(down))
        with m3:
            st.metric("Мгновенно ↑", format_rate(rate_up))
        with m4:
            st.metric("Мгновенно ↓", format_rate(rate_down))

        t0 = st.session_state.get("edge_rt_stream_t0")
        avg_up: float | None = None
        avg_down: float | None = None
        if st.session_state.edge_rt_running and t0 is not None:
            elapsed = now - t0
            if elapsed >= 0.05:
                avg_up = up / elapsed
                avg_down = down / elapsed
        else:
            avg_up = st.session_state.get("edge_rt_session_avg_up")
            avg_down = st.session_state.get("edge_rt_session_avg_down")

        a1, a2 = st.columns(2)
        with a1:
            st.metric("Средняя нагрузка ↑ (за сессию)", format_rate(avg_up) if avg_up is not None else "—")
        with a2:
            st.metric("Средняя нагрузка ↓ (за сессию)", format_rate(avg_down) if avg_down is not None else "—")

    center = st.session_state.edge_rt_center
    if center is not None:
        safe = html.escape(str(center))
        st.markdown(
            f'<p style="text-align:center;font-size:6rem;font-weight:700;margin:0.2em 0;">{safe}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="text-align:center;font-size:1.5rem;color:#888;margin:1em 0;">Скажите число</p>',
            unsafe_allow_html=True,
        )
    lines = list(st.session_state.edge_rt_lines)
    st.text_area(
        "События",
        value="\n".join(reversed(lines)) if lines else "(пусто)",
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )


if hasattr(st, "fragment"):
    st.markdown("### Поток")

    @st.fragment(run_every=0.25)
    def _live_fragment() -> None:
        _render_live_panel()

    _live_fragment()
else:
    st.warning(
        "Нужен Streamlit ≥ 1.33 (`st.fragment`). Обновите streamlit или запустите "
        "`scripts/realtime_kws_logmel_client.py`."
    )
    st.markdown("### Поток")
    _render_live_panel()
    if st.button("Обновить вид"):
        st.rerun()
