import json

import streamlit.components.v1 as components


def render_realtime_widget(ws_url: str, ws_config: dict) -> None:
    ws_cfg_json = json.dumps(ws_config)
    ws_url_json = json.dumps(ws_url)

    html = f"""
<div id="kws-ws-ui" style="font-family:sans-serif;max-width:560px;margin:0 auto;">
  <p>
    <button id="kwsStart" type="button" style="padding:8px 16px;margin-right:8px;">Старт</button>
    <button id="kwsStop" type="button" style="padding:8px 16px;" disabled>Стоп</button>
  </p>
  <p id="kwsStatus" style="color:#666;font-size:14px;">Отключено</p>
  <div id="kwsMetrics" style="font-size:12px;line-height:1.5;color:#222;background:#f4f4f4;padding:8px 10px;border-radius:6px;margin:8px 0;">
    <strong>Трафик (клиент → WebSocket)</strong><br/>
    <span id="kwsMetUpTot">↑ 0 B</span> · <span id="kwsMetDownTot">↓ 0 B</span><br/>
    <span id="kwsMetRates">скорость: ↑ — · ↓ —</span><br/>
    <span style="color:#666;">Без учёта TCP/WS framing; PCM float32 непрерывно.</span>
  </div>
  <div id="kwsCenter" style="min-height:200px;display:flex;align-items:center;justify-content:center;margin:16px 0;">
    <p id="kwsCenterText" style="margin:0;text-align:center;font-size:1.5rem;color:#888;">Скажите число</p>
  </div>
  <pre id="kwsLog" style="background:#111;color:#ddd;padding:10px;border-radius:6px;height:180px;overflow:auto;font-size:11px;"></pre>
</div>
<script>
(function() {{
  const WS_URL = {ws_url_json};
  const CFG_BASE = {ws_cfg_json};
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
  let bytesUp = 0;
  let bytesDown = 0;
  let metPrevUp = 0;
  let metPrevDown = 0;
  let metPrevT = performance.now();
  let metricsTimer = null;
  const te = new TextEncoder();

  function fmtBytes(n) {{
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KiB";
    return (n / 1048576).toFixed(2) + " MiB";
  }}

  function updateMetricsDisplay() {{
    const elU = el("kwsMetUpTot");
    const elD = el("kwsMetDownTot");
    const elR = el("kwsMetRates");
    if (!elU) return;
    elU.textContent = "↑ " + fmtBytes(bytesUp);
    elD.textContent = "↓ " + fmtBytes(bytesDown);
    const now = performance.now();
    const dt = (now - metPrevT) / 1000;
    if (dt > 0.15) {{
      const ru = (bytesUp - metPrevUp) / dt;
      const rd = (bytesDown - metPrevDown) / dt;
      metPrevUp = bytesUp;
      metPrevDown = bytesDown;
      metPrevT = now;
      elR.textContent = "скорость: ↑ " + fmtBytes(Math.max(0, ru)) + "/s · ↓ " + fmtBytes(Math.max(0, rd)) + "/s";
    }}
  }}

  function startMetricsTimer() {{
    stopMetricsTimer();
    metricsTimer = setInterval(updateMetricsDisplay, 250);
  }}

  function stopMetricsTimer() {{
    if (metricsTimer) {{
      clearInterval(metricsTimer);
      metricsTimer = null;
    }}
  }}

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
    stopMetricsTimer();
    bytesUp = 0;
    bytesDown = 0;
    metPrevUp = 0;
    metPrevDown = 0;
    metPrevT = performance.now();
    updateMetricsDisplay();
    const elRate = el("kwsMetRates");
    if (elRate) elRate.textContent = "скорость: ↑ — · ↓ —";
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
        if (typeof ev.data === "string") {{
          bytesDown += te.encode(ev.data).length;
        }}
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
        const cfgStr = JSON.stringify(cfg);
        bytesUp += te.encode(cfgStr).length;
        ws.send(cfgStr);
        startMetricsTimer();

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
          const buf = copy.buffer;
          bytesUp += buf.byteLength;
          ws.send(buf);
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

    components.html(html, height=520, scrolling=False)
