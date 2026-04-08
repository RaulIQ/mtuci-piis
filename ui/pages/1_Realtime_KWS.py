import json
import os
import urllib.parse

import requests
import streamlit as st
import streamlit.components.v1 as components

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
    target_labels_raw = st.text_input(
        "Target labels (comma-separated)",
        value="one,two,three,four,five,six,seven,eight,nine",
    )

target_labels = [x.strip() for x in target_labels_raw.split(",") if x.strip()]

ws_config = {
    "poll_interval_sec": poll_interval_sec,
    "confidence_threshold": confidence_threshold,
    "refractory_sec": refractory_sec,
    "target_labels": target_labels,
}

st.markdown("### Поток с микрофона (WebSocket)")
st.caption(
    "Нажмите «Старт» в блоке ниже, разрешите микрофон. "
    "Аудио уходит на сервер; ответы — в реальном времени. "
    "Нужен запущенный API (uvicorn) на том же хосте, что в REST URL."
)

_ws_cfg_json = json.dumps(ws_config)
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

components.html(html, height=420, scrolling=False)

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
        target_set = {x.strip() for x in targets_off.split(",") if x.strip()}

        if st.button("Запустить predict-stream"):
            try:
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
                else:
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
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Ошибка: {exc}")
