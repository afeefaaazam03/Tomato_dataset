"use strict";

const DEFAULT_SPACE_URL = "https://afeefaaazam03-tomatogui-external-v2.hf.space";
const SETTINGS_KEY = "aerialyield-pwa-settings-v1";
const LIVE_INTERVAL_MS = 1000;

// ---------- Gradio REST client ----------
// Talks directly to a Gradio Space's public /gradio_api/ REST API:
//   POST  <base>/gradio_api/upload            (large files: video)   -> [serverPath, ...]
//   POST  <base>/gradio_api/call/<api_name>    {data: [...]}          -> {event_id}
//   GET   <base>/gradio_api/call/<api_name>/<event_id>  (SSE)         -> event: complete\ndata: [...]

async function gradioUpload(baseUrl, file) {
  const form = new FormData();
  form.append("files", file, file.name || "upload");
  const resp = await fetch(`${baseUrl}/gradio_api/upload`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`Upload failed (HTTP ${resp.status})`);
  const paths = await resp.json();
  return paths[0];
}

async function gradioCall(baseUrl, apiName, data) {
  const postResp = await fetch(`${baseUrl}/gradio_api/call/${apiName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  if (!postResp.ok) throw new Error(`Request failed (HTTP ${postResp.status})`);
  const { event_id: eventId } = await postResp.json();

  const streamResp = await fetch(`${baseUrl}/gradio_api/call/${apiName}/${eventId}`);
  if (!streamResp.ok || !streamResp.body) throw new Error(`Stream failed (HTTP ${streamResp.status})`);

  const reader = streamResp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sepIdx;
    while ((sepIdx = buf.indexOf("\n\n")) !== -1) {
      const rawEvent = buf.slice(0, sepIdx);
      buf = buf.slice(sepIdx + 2);
      let eventType = "message";
      let dataLine = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (eventType === "complete") return JSON.parse(dataLine);
      if (eventType === "error") {
        let message = "The Space returned an error";
        try { message = JSON.parse(dataLine).error || message; } catch (_e) { /* ignore */ }
        throw new Error(message);
      }
      // heartbeat / generating -> keep waiting
    }
  }
  throw new Error("Stream ended before a result arrived");
}

function imageFileData(dataUrl, name) {
  return {
    path: null,
    url: dataUrl,
    size: null,
    orig_name: name,
    mime_type: "image/jpeg",
    is_stream: false,
    meta: { _type: "gradio.FileData" },
  };
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

function captureFrameDataURL(videoEl, canvasEl, maxDim) {
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (!vw || !vh) return null;
  let w = vw;
  let h = vh;
  if (Math.max(w, h) > maxDim) {
    const scale = maxDim / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  canvasEl.width = w;
  canvasEl.height = h;
  canvasEl.getContext("2d").drawImage(videoEl, 0, 0, w, h);
  return canvasEl.toDataURL("image/jpeg", 0.85);
}

// ---------- Rendering helpers ----------

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderTable(container, df) {
  if (!df || !df.data) {
    container.innerHTML = "";
    return;
  }
  let html = "<table><thead><tr>";
  for (const heading of df.headers) html += `<th>${escapeHtml(heading)}</th>`;
  html += "</tr></thead><tbody>";
  for (const row of df.data) {
    html += "<tr>" + row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  container.innerHTML = html;
}

function renderMarkdownish(container, text) {
  if (!text) {
    container.innerHTML = "";
    return;
  }
  const html = text
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n{2,}/g, "<br><br>")
    .replace(/\n(?!<)/g, "<br>");
  container.innerHTML = html;
}

// ---------- Settings ----------

function loadSettings() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  } catch (_e) {
    stored = {};
  }
  return {
    spaceUrl: stored.spaceUrl || DEFAULT_SPACE_URL,
    conf: stored.conf ?? 0.25,
    iou: stored.iou ?? 0.45,
    weight: stored.weight ?? 120,
    imgSize: stored.imgSize ?? 1024,
  };
}

function getSettings() {
  return {
    spaceUrl: document.getElementById("cfg-space-url").value.trim().replace(/\/$/, ""),
    conf: Number(document.getElementById("cfg-conf").value),
    iou: Number(document.getElementById("cfg-iou").value),
    weight: Number(document.getElementById("cfg-weight").value),
    imgSize: Number(document.getElementById("cfg-imgsize").value),
  };
}

function persistSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(getSettings()));
}

function initSettingsUI() {
  const s = loadSettings();
  document.getElementById("cfg-space-url").value = s.spaceUrl;
  document.getElementById("cfg-conf").value = s.conf;
  document.getElementById("cfg-conf-out").textContent = s.conf;
  document.getElementById("cfg-iou").value = s.iou;
  document.getElementById("cfg-iou-out").textContent = s.iou;
  document.getElementById("cfg-weight").value = s.weight;
  document.getElementById("cfg-weight-out").textContent = s.weight;
  document.getElementById("cfg-imgsize").value = s.imgSize;

  document.getElementById("cfg-space-url").addEventListener("change", persistSettings);
  document.getElementById("cfg-conf").addEventListener("input", (e) => {
    document.getElementById("cfg-conf-out").textContent = e.target.value;
    persistSettings();
  });
  document.getElementById("cfg-iou").addEventListener("input", (e) => {
    document.getElementById("cfg-iou-out").textContent = e.target.value;
    persistSettings();
  });
  document.getElementById("cfg-weight").addEventListener("input", (e) => {
    document.getElementById("cfg-weight-out").textContent = e.target.value;
    persistSettings();
  });
  document.getElementById("cfg-imgsize").addEventListener("change", persistSettings);

  document.getElementById("settings-toggle").addEventListener("click", () => {
    const body = document.getElementById("settings-body");
    const btn = document.getElementById("settings-toggle");
    const wasHidden = body.hidden;
    body.hidden = !wasHidden;
    btn.setAttribute("aria-expanded", String(wasHidden));
  });
}

// ---------- Tabs ----------

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab !== "live" && liveActive) stopLive();
    });
  });
}

// ---------- Live tab ----------

let liveStream = null;
let liveActive = false;
let liveBusy = false;
let liveTimer = null;

function setLiveStatus(text, kind) {
  const el = document.getElementById("live-status");
  el.textContent = text;
  el.className = "status-text" + (kind ? ` ${kind}` : "");
}

async function startLive() {
  const videoEl = document.getElementById("live-video");
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
  } catch (err) {
    setLiveStatus(`Camera access denied: ${err.message}`, "error");
    return;
  }
  videoEl.srcObject = liveStream;
  try {
    await videoEl.play();
  } catch (_e) {
    /* autoplay can reject before first user gesture settles; stream still attaches */
  }
  liveActive = true;
  document.getElementById("live-start").disabled = true;
  document.getElementById("live-stop").disabled = false;
  setLiveStatus("Starting camera...", "ok");
  liveLoop();
}

function stopLive() {
  liveActive = false;
  if (liveTimer) clearTimeout(liveTimer);
  liveTimer = null;
  if (liveStream) {
    liveStream.getTracks().forEach((track) => track.stop());
    liveStream = null;
  }
  document.getElementById("live-video").srcObject = null;
  document.getElementById("live-start").disabled = false;
  document.getElementById("live-stop").disabled = true;
  setLiveStatus("Camera off");
}

async function liveLoop() {
  if (!liveActive) return;
  if (!liveBusy) {
    liveBusy = true;
    try {
      const settings = getSettings();
      const dataUrl = captureFrameDataURL(
        document.getElementById("live-video"),
        document.getElementById("live-canvas"),
        settings.imgSize
      );
      if (dataUrl) {
        const [imgOut, df, yieldMd] = await gradioCall(settings.spaceUrl, "run_image", [
          imageFileData(dataUrl, "live.jpg"),
          settings.conf,
          settings.iou,
          settings.weight,
          settings.imgSize,
        ]);
        document.getElementById("live-result-img").src = imgOut.url;
        renderTable(document.getElementById("live-table"), df);
        renderMarkdownish(document.getElementById("live-yield"), yieldMd);
        setLiveStatus(`Live — updated ${new Date().toLocaleTimeString()}`, "ok");
      }
    } catch (err) {
      setLiveStatus(`Detection error: ${err.message}`, "error");
    } finally {
      liveBusy = false;
    }
  }
  if (liveActive) liveTimer = setTimeout(liveLoop, LIVE_INTERVAL_MS);
}

function initLiveTab() {
  document.getElementById("live-start").addEventListener("click", startLive);
  document.getElementById("live-stop").addEventListener("click", stopLive);
}

// ---------- Video tab ----------

let selectedVideoFile = null;

function initVideoTab() {
  const fileInput = document.getElementById("video-file");
  const runBtn = document.getElementById("video-run");
  const statusEl = document.getElementById("video-status");

  fileInput.addEventListener("change", (e) => {
    selectedVideoFile = e.target.files[0] || null;
    runBtn.disabled = !selectedVideoFile;
    statusEl.textContent = selectedVideoFile ? selectedVideoFile.name : "No video selected";
    statusEl.className = "status-text";
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedVideoFile) return;
    runBtn.disabled = true;
    statusEl.className = "status-text";
    const settings = getSettings();
    try {
      statusEl.textContent = "Uploading...";
      const serverPath = await gradioUpload(settings.spaceUrl, selectedVideoFile);

      statusEl.textContent = "Detecting — this can take a minute...";
      const sampleEvery = Number(document.getElementById("video-sample-every").value);
      const maxFrames = Number(document.getElementById("video-max-frames").value);
      const result = await gradioCall(settings.spaceUrl, "run_video", [
        { path: serverPath, meta: { _type: "gradio.FileData" } },
        settings.conf,
        settings.iou,
        settings.weight,
        settings.imgSize,
        sampleEvery,
        maxFrames,
      ]);
      const [videoOut, df, overlapMd, yieldMd, csvOut] = result;

      const resultVideoEl = document.getElementById("video-result");
      resultVideoEl.src = videoOut.url;
      resultVideoEl.hidden = false;
      renderTable(document.getElementById("video-table"), df);
      renderMarkdownish(document.getElementById("video-overlap"), overlapMd);
      renderMarkdownish(document.getElementById("video-yield"), yieldMd);
      const csvLink = document.getElementById("video-csv");
      csvLink.href = csvOut.url;
      csvLink.hidden = false;

      statusEl.textContent = "Done";
      statusEl.className = "status-text ok";
    } catch (err) {
      statusEl.textContent = `Error: ${err.message}`;
      statusEl.className = "status-text error";
    } finally {
      runBtn.disabled = false;
    }
  });
}

// ---------- Image tab ----------

let selectedImageFile = null;

function initImageTab() {
  const fileInput = document.getElementById("image-file");
  const runBtn = document.getElementById("image-run");
  const statusEl = document.getElementById("image-status");

  fileInput.addEventListener("change", async () => {
    selectedImageFile = fileInput.files[0] || null;
    runBtn.disabled = !selectedImageFile;
    if (selectedImageFile) {
      statusEl.textContent = selectedImageFile.name;
      statusEl.className = "status-text";
      document.getElementById("image-preview").src = await fileToDataURL(selectedImageFile);
    } else {
      statusEl.textContent = "No photo selected";
      document.getElementById("image-preview").src = "";
    }
  });

  runBtn.addEventListener("click", async () => {
    if (!selectedImageFile) return;
    runBtn.disabled = true;
    try {
      statusEl.textContent = "Detecting...";
      statusEl.className = "status-text";
      const settings = getSettings();
      const dataUrl = await fileToDataURL(selectedImageFile);
      const [imgOut, df, yieldMd] = await gradioCall(settings.spaceUrl, "run_image", [
        imageFileData(dataUrl, selectedImageFile.name || "photo.jpg"),
        settings.conf,
        settings.iou,
        settings.weight,
        settings.imgSize,
      ]);
      document.getElementById("image-result-img").src = imgOut.url;
      renderTable(document.getElementById("image-table"), df);
      renderMarkdownish(document.getElementById("image-yield"), yieldMd);
      statusEl.textContent = "Done";
      statusEl.className = "status-text ok";
    } catch (err) {
      statusEl.textContent = `Error: ${err.message}`;
      statusEl.className = "status-text error";
    } finally {
      runBtn.disabled = false;
    }
  });
}

// ---------- Boot ----------

document.addEventListener("DOMContentLoaded", () => {
  initSettingsUI();
  initTabs();
  initLiveTab();
  initVideoTab();
  initImageTab();
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {
      /* offline app-shell caching is a nice-to-have, not required for the app to function */
    });
  });
}
