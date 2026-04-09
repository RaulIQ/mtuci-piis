import json
import io
import os
import sys
import threading
import time
import urllib.parse
from collections import deque
from pathlib import Path

import av
import librosa
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ui.services.kws_model import build_mel_transform, compute_canonical_log_mels
from ui.services.kws_transport import (
    audio_frame_to_mono_float32,
    render_center,
    run_realtime_bridge,
)

st.set_page_config(page_title="Realtime KWS", layout="centered")
st.title("Realtime KWS")

api_url = os.getenv("API_URL", "http://localhost:8000")
parsed = urllib.parse.urlparse(api_url)
_ws_scheme = "wss" if parsed.scheme == "https" else "ws"
ws_url = f"{_ws_scheme}://{parsed.netloc}/ws/kws"

st.write(f"REST: `{api_url}` · WebSocket: `{ws_url}`")

st.markdown("### Параметры потока (отправляются при старте)")
col_a, col_b = st.columns(2)
with col_a:
    poll_interval_sec = st.slider(
        "Интервал опроса (с)",
        0.10,
        0.50,
        0.25,
        0.05,
        help="Раз в столько секунд сервер берёт последнюю 1 с аудио и вызывает модель.",
    )
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.99, 0.55, 0.01)
with col_b:
    refractory_sec = st.slider("Refractory (sec)", 0.2, 2.0, 0.8, 0.1)
    ws_input_type_human = st.radio(
        "Формат потока в WS",
        ["PCM float32", "log-mel спектрограммы"],
        index=0,
        help="Во втором режиме браузер шлёт PCM в UI backend, а UI backend считает каноничные log-mel и отправляет их в API.",
    )
    target_labels_raw = st.text_input(
        "Target labels (comma-separated)",
        value="one,two,three,four,five,six,seven,eight,nine",
    )

target_labels = [x.strip() for x in target_labels_raw.split(",") if x.strip()]

ws_input_type = "log_mel" if ws_input_type_human == "log-mel спектрограммы" else "audio"
ws_config = {
    "poll_interval_sec": poll_interval_sec,
    "confidence_threshold": confidence_threshold,
    "refractory_sec": refractory_sec,
    "target_labels": target_labels,
    "input_type": ws_input_type,
}

st.markdown("### Поток с микрофона")
if ws_input_type == "audio":
    st.caption(
        "Режим PCM: браузер отправляет float32 PCM напрямую в API по WebSocket, как в исходной реализации."
    )
    _ws_cfg_json = json.dumps(
        {
            "poll_interval_sec": poll_interval_sec,
            "confidence_threshold": confidence_threshold,
            "refractory_sec": refractory_sec,
            "target_labels": target_labels,
        }
    )
    _ws_url_json = json.dumps(ws_url)
    html = f"""
<div id="kws-ws-ui" style="font-family:sans-serif;max-width:560px;margin:0 auto;">
  <p>
    <button id="kwsStart" type="button" style="padding:8px 16px;margin-right:8px;">Старт</button>
    <button id="kwsStop" type="button" style="padding:8px 16px;" disabled>Стоп</button>
  </p>
  <p id="kwsStatus" style="color:#666;font-size:14px;">Отключено</p>
  <div id="kwsCenter" style="min-height:200px;display:flex;align-items:center;justify-content:center;margin:16px 0;">
    <p id="kwsCenterText" style="margin:0;text-align:center;font-size:1.5rem;color:#888;">Скажите число</p>
  </div>
  <pre id="kwsLog" style="background:#111;color:#ddd;padding:10px;border-radius:6px;height:180px;overflow:auto;font-size:11px;"></pre>
</div>
<script>
(function() {{
  const WS_URL = {_ws_url_json};
  const CFG_BASE = {_ws_cfg_json};
  const DIGIT_MAP = {{
    one: "1", two: "2", three: "3", four: "4", five: "5",
    six: "6", seven: "7", eight: "8", nine: "9", zero: "0"
  }};
  const BAD_MS = 2000;

  const el = (id) => document.getElementById(id);
  const statusEl = el("kwsStatus");
  const centerTextEl = el("kwsCenterText");
  const logEl = el("kwsLog");
  const btnStart = el("kwsStart");
  const btnStop = el("kwsStop");

  let ws = null;
  let ctx = null;
  let proc = null;
  let src = null;
  let stream = null;
  let gain = null;
  let centerDigit = null;
  let badSince = null;

  function targetSet() {{
    const arr = CFG_BASE.target_labels || [];
    return new Set(arr.map(function(x) {{ return String(x).toLowerCase(); }}));
  }}

  function renderCenter() {{
    if (centerDigit !== null) {{
      centerTextEl.textContent = centerDigit;
      centerTextEl.style.fontSize = "6rem";
      centerTextEl.style.fontWeight = "700";
      centerTextEl.style.color = "#111";
    }} else {{
      centerTextEl.textContent = "Скажите число";
      centerTextEl.style.fontSize = "1.5rem";
      centerTextEl.style.fontWeight = "400";
      centerTextEl.style.color = "#888";
    }}
  }}

  function onPrediction(pred, conf) {{
    const low = String(pred || "").toLowerCase();
    const thr = Number(CFG_BASE.confidence_threshold);
    const targets = targetSet();
    const valid = targets.has(low) && conf >= thr;

    if (valid) {{
      centerDigit = DIGIT_MAP[low] || low;
      badSince = null;
      renderCenter();
      return;
    }}

    if (low === "silence" || low === "unknown") {{
      if (centerDigit !== null) {{
        if (badSince === null) badSince = performance.now();
        else if (performance.now() - badSince >= BAD_MS) {{
          centerDigit = null;
          badSince = null;
        }}
      }} else {{
        badSince = null;
      }}
      renderCenter();
      return;
    }}

    badSince = null;
    renderCenter();
  }}

  function logLine(s) {{
    const t = new Date().toLocaleTimeString();
    const line = "[" + t + "] " + s;
    const prev = logEl.textContent.split("\\n").filter(function(x) {{ return x.length; }});
    prev.unshift(line);
    logEl.textContent = prev.slice(0, 10).join("\\n");
  }}

  function stopAll() {{
    try {{
      if (proc) {{ proc.disconnect(); proc.onaudioprocess = null; proc = null; }}
      if (gain) {{ gain.disconnect(); gain = null; }}
      if (src) {{ src.disconnect(); src = null; }}
      if (ctx) {{ ctx.close(); ctx = null; }}
      if (stream) {{
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }}
      if (ws && ws.readyState <= 1) ws.close();
      ws = null;
    }} catch (e) {{}}
    centerDigit = null;
    badSince = null;
    renderCenter();
    btnStart.disabled = false;
    btnStop.disabled = true;
    statusEl.textContent = "Отключено";
  }}

  btnStop.onclick = stopAll;

  btnStart.onclick = async function() {{
    if (ws) return;
    btnStart.disabled = true;
    statusEl.textContent = "Подключение…";
    centerDigit = null;
    badSince = null;
    renderCenter();

    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onerror = function() {{
      logLine("WebSocket error");
      statusEl.textContent = "Ошибка WebSocket";
      stopAll();
    }};

    ws.onclose = function() {{
      logLine("WebSocket closed");
      stopAll();
    }};

    ws.onmessage = function(ev) {{
      try {{
        const d = JSON.parse(ev.data);
        if (d.type === "ready") {{
          statusEl.textContent = "Готово, модель: " + (d.model_version || "?");
          logLine("ready " + JSON.stringify(d.model_version));
          return;
        }}
        if (d.type === "prediction") {{
          const conf = Number(d.confidence || 0);
          onPrediction(d.predicted_class, conf);
          const p = d.predicted_class + " (" + conf.toFixed(3) + ")";
          if (d.trigger && d.detection) {{
            logLine("TRIGGER " + JSON.stringify(d.detection));
          }} else {{
            logLine("pred " + p + " inf_ms=" + (d.inference_ms || 0));
          }}
          return;
        }}
        if (d.type === "error") {{
          logLine("ERR " + (d.message || ""));
          return;
        }}
        logLine(ev.data);
      }} catch (e) {{
        logLine("parse err " + e);
      }}
    }};

    ws.onopen = async function() {{
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{ audio: true, video: false }});
        ctx = new (window.AudioContext || window.webkitAudioContext)();
        const sr = ctx.sampleRate;
        const cfg = Object.assign({{}}, CFG_BASE, {{ sample_rate: sr }});
        ws.send(JSON.stringify(cfg));

        src = ctx.createMediaStreamSource(stream);
        const bufferSize = 4096;
        proc = ctx.createScriptProcessor(bufferSize, 1, 1);
        gain = ctx.createGain();
        gain.gain.value = 0;

        proc.onaudioprocess = function(e) {{
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          const ch = e.inputBuffer.getChannelData(0);
          const copy = new Float32Array(ch.length);
          copy.set(ch);
          ws.send(copy.buffer);
        }};

        src.connect(proc);
        proc.connect(gain);
        gain.connect(ctx.destination);

        btnStop.disabled = false;
        statusEl.textContent = "Стрим " + sr + " Hz…";
      }} catch (e) {{
        logLine("mic/audio: " + e);
        statusEl.textContent = "Нет доступа к микрофону или AudioContext";
        if (ws) ws.close();
        stopAll();
      }}
    }};
  }};
}})();
</script>
"""
    st.components.v1.html(html, height=420, scrolling=False)
else:
    st.caption(
        "Режим log-mel: браузер шлёт PCM в UI backend через WebRTC без DSP-улучшателей; "
        "UI backend повторяет серверную логику формирования log-mel и отправляет в API уже спектрограммы."
    )

    status_placeholder = st.empty()
    traffic_placeholder = st.empty()
    center_placeholder = st.empty()
    log_placeholder = st.empty()
    render_center(center_placeholder, None)
    traffic_placeholder.info("UI <-> API: tx 0 B, rx 0 B, up 0.0 kbps, down 0.0 kbps")

    ready_resp = requests.get(f"{api_url}/ready", timeout=10)
    if not ready_resp.ok:
        status_placeholder.error(f"/ready ошибка: {ready_resp.status_code}")
        st.code(ready_resp.text)
    else:
        spec_cfg = ready_resp.json().get("spec", {})
        model_sr = int(spec_cfg.get("sample_rate", 16000))
        n_fft = int(spec_cfg.get("n_fft", 512))
        hop_length = int(spec_cfg.get("hop_length", 160))
        n_mels = int(spec_cfg.get("n_mels", 128))
        frames = int(spec_cfg.get("frames", 101))
        mel_tf = build_mel_transform(model_sr, n_fft, hop_length, n_mels)

        audio_lock = threading.Lock()
        audio_chunks: deque[np.ndarray] = deque()
        audio_meta = {"sample_rate": None}

        def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            mono = audio_frame_to_mono_float32(frame)
            with audio_lock:
                audio_chunks.append(mono)
                audio_meta["sample_rate"] = int(frame.sample_rate)
            return frame

        webrtc_ctx = webrtc_streamer(
            key="kws-edge-bridge",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "echoCancellation": False,
                    "noiseSuppression": False,
                    "autoGainControl": False,
                    "channelCount": 1,
                },
            },
            audio_frame_callback=_audio_frame_callback,
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            status_placeholder.info("Жду первые аудио-фреймы от браузера...")

            while webrtc_ctx.state.playing and audio_meta["sample_rate"] is None:
                time.sleep(0.05)

            try:
                run_realtime_bridge(
                    webrtc_ctx=webrtc_ctx,
                    audio_lock=audio_lock,
                    audio_chunks=audio_chunks,
                    audio_meta=audio_meta,
                    ws_url=ws_url,
                    ws_input_type=ws_input_type,
                    ws_config=ws_config,
                    model_sr=model_sr,
                    poll_interval_sec=poll_interval_sec,
                    confidence_threshold=confidence_threshold,
                    target_labels=target_labels,
                    mel_tf=mel_tf,
                    frames=frames,
                    status_placeholder=status_placeholder,
                    traffic_placeholder=traffic_placeholder,
                    center_placeholder=center_placeholder,
                    log_placeholder=log_placeholder,
                )
            except Exception as exc:  # pylint: disable=broad-except
                status_placeholder.error(f"Ошибка realtime bridge: {exc}")
                log_placeholder.code(f"bridge error {exc}")
        else:
            status_placeholder.info("Нажмите Start в WebRTC-блоке ниже.")
            log_placeholder.code("Ожидание запуска микрофона...")

st.markdown("---")
st.markdown("### Запись файлом (как раньше)")
st.caption("Одиночный Predict и predict-stream по завершённой записи — тот же API.")

recorded = st.audio_input("Записать для офлайн-режима")

audio_bytes = None
audio_name = "recorded.wav"

if recorded is not None:
    audio_bytes = recorded.getvalue()
    st.audio(audio_bytes, format="audio/wav")

if audio_bytes is not None:
    mode = st.radio(
        "Режим",
        ["Один запрос (Predict)", "Скользящее окно на сервере"],
        index=0,
    )

    if mode == "Один запрос (Predict)":
        if st.button("Predict"):
            try:
                files = {"file": (audio_name, audio_bytes, "audio/wav")}
                response = requests.post(f"{api_url}/predict", files=files, timeout=30)
                if response.ok:
                    st.success("Готово")
                    st.json(response.json())
                else:
                    st.error(f"Ошибка: {response.status_code}")
                    st.code(response.text)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Ошибка: {exc}")
    else:
        st.markdown("### Параметры окна")
        stride_sec = st.slider("Stride (seconds)", 0.10, 0.50, 0.25, 0.05, key="off_stride")
        refractory_off = st.slider(
            "Refractory period (seconds)", 0.20, 2.00, 0.80, 0.10, key="off_refr"
        )
        confidence_off = st.slider(
            "Confidence threshold", 0.10, 0.99, 0.55, 0.01, key="off_conf"
        )
        targets_off = st.text_input(
            "Target labels (comma-separated)",
            value="one,two,three,four,five,six,seven,eight,nine",
            key="off_targets",
        )
        stream_input_mode = st.radio(
            "Что отправлять на API",
            ["Сырые аудио-чанки (как сейчас)", "Секундные log-mel спектрограммы (edge)"],
            index=0,
            key="off_stream_input_mode",
        )
        target_set = {x.strip() for x in targets_off.split(",") if x.strip()}

        if st.button("Запустить predict-stream"):
            try:
                if stream_input_mode == "Сырые аудио-чанки (как сейчас)":
                    files = {"file": (audio_name, audio_bytes, "audio/wav")}
                    data = {
                        "stride_sec": str(stride_sec),
                        "refractory_sec": str(refractory_off),
                        "confidence_threshold": str(confidence_off),
                        "target_labels": ",".join(sorted(target_set)),
                    }
                    response = requests.post(
                        f"{api_url}/predict-stream",
                        files=files,
                        data=data,
                        timeout=60,
                    )
                    if not response.ok:
                        st.error(f"Ошибка: {response.status_code}")
                        st.code(response.text)
                        st.stop()
                    payload = response.json()
                    detections = payload.get("detections", [])
                    rows = payload.get("window_predictions", [])

                    st.success("Готово")
                    st.write(f"Окон: {payload.get('windows_processed', len(rows))}")
                    st.write(f"Детекций: {payload.get('detections_count', len(detections))}")

                    if detections:
                        st.dataframe(detections, use_container_width=True)
                    else:
                        st.info("Нет детекций.")

                    st.markdown("### По окнам")
                    st.dataframe(rows, use_container_width=True, height=350)
                else:
                    ready_resp = requests.get(f"{api_url}/ready", timeout=10)
                    if not ready_resp.ok:
                        st.error(f"/ready ошибка: {ready_resp.status_code}")
                        st.code(ready_resp.text)
                        st.stop()
                    spec_cfg = ready_resp.json().get("spec", {})
                    model_sr = int(spec_cfg.get("sample_rate", 16000))
                    n_fft = int(spec_cfg.get("n_fft", 512))
                    hop_length = int(spec_cfg.get("hop_length", 160))
                    n_mels = int(spec_cfg.get("n_mels", 128))
                    frames = int(spec_cfg.get("frames", 101))
                    mel_tf = build_mel_transform(model_sr, n_fft, hop_length, n_mels)

                    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                    if sr != model_sr:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=model_sr)
                        sr = model_sr

                    win_size = int(1.0 * sr)
                    step = max(1, int(stride_sec * sr))
                    starts = list(range(0, max(1, len(audio) - win_size + 1), step))
                    if len(audio) <= win_size:
                        starts = [0]

                    windows = []
                    detections = []
                    refractory_until = -1.0
                    t0 = time.perf_counter()
                    ui_api_tx_bytes = 0
                    ui_api_rx_bytes = 0

                    for start in starts:
                        end = start + win_size
                        chunk = audio[start:end]
                        if len(chunk) < win_size:
                            chunk = np.pad(chunk, (0, win_size - len(chunk)))
                        log_mels = compute_canonical_log_mels(
                            chunk,
                            sr=sr,
                            mel_tf=mel_tf,
                            frames=frames,
                        )
                        if log_mels is None:
                            pred_label = "silence"
                            pred_conf = 1.0
                        else:
                            payload = {"log_mels": log_mels.astype(np.float32).tolist()}
                            payload_bytes = json.dumps(payload).encode("utf-8")
                            ui_api_tx_bytes += len(payload_bytes)
                            resp = requests.post(
                                f"{api_url}/predict-spectrogram",
                                data=payload_bytes,
                                headers={"Content-Type": "application/json"},
                                timeout=30,
                            )
                            ui_api_rx_bytes += len(resp.content or b"")
                            if not resp.ok:
                                st.error(f"/predict-spectrogram ошибка: {resp.status_code}")
                                st.code(resp.text)
                                st.stop()
                            pred = resp.json()
                            pred_label = pred["predicted_class"]
                            pred_conf = float(pred["confidence"])

                        t_sec = round(start / sr, 3)
                        windows.append(
                            {
                                "t_sec": t_sec,
                                "predicted_class": pred_label,
                                "confidence": round(pred_conf, 4),
                            }
                        )
                        is_trigger = (
                            pred_label in target_set
                            and pred_conf >= confidence_off
                            and t_sec >= refractory_until
                        )
                        if is_trigger:
                            detections.append(
                                {
                                    "t_sec": t_sec,
                                    "label": pred_label,
                                    "confidence": round(pred_conf, 4),
                                }
                            )
                            refractory_until = t_sec + refractory_off

                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    windows_processed = len(windows)
                    payload = {
                        "windows_processed": windows_processed,
                        "detections_count": len(detections),
                        "detections": detections,
                        "window_predictions": windows,
                    }
                    raw_window_bytes = win_size * 4
                    spec_window_bytes = n_mels * frames * 4

                    st.success("Готово (edge-режим: отправка спектрограмм)")
                    st.write(f"Окон: {payload.get('windows_processed', len(windows))}")
                    st.write(f"Детекций: {payload.get('detections_count', len(detections))}")
                    st.write(f"Суммарное время клиента (мс): {elapsed_ms:.1f}")
                    st.markdown("### Сравнение сетевой нагрузки (float32)")
                    st.write(f"Raw audio на окно (1с): {raw_window_bytes / 1024:.1f} KiB")
                    st.write(f"log-mel на окно: {spec_window_bytes / 1024:.1f} KiB")
                    st.write(f"Raw audio суммарно: {(raw_window_bytes * windows_processed) / 1024:.1f} KiB")
                    st.write(f"log-mel суммарно: {(spec_window_bytes * windows_processed) / 1024:.1f} KiB")
                    st.markdown("### Фактическая нагрузка UI ↔ API (edge режим)")
                    st.write(f"UI -> API (JSON payload): {ui_api_tx_bytes / 1024:.1f} KiB")
                    st.write(f"API -> UI (responses): {ui_api_rx_bytes / 1024:.1f} KiB")

                    if detections:
                        st.dataframe(detections, use_container_width=True)
                    else:
                        st.info("Нет детекций.")

                    st.markdown("### По окнам")
                    st.dataframe(windows, use_container_width=True, height=350)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Ошибка: {exc}")
