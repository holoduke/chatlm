/* ui.js
 * UI chrome: theme cycling, toggle buttons (THINK/TOOLS/AUTO-APPROVE),
 * model-selection dropdowns, mode switcher (CHAT/VIDEO), PURGE MEM
 * button, stats polling, textarea autosize, keyboard submit. */

import {
  $, logEl, inputEl, formEl, modelEl, setStatus, setBar,
} from "./core.js";
import { toggles, history, pending, STORAGE_KEYS } from "./state.js";
import { addMessage } from "./messages.js";
import { Sessions } from "./sessions.js";
import {
  appendFeed, clearFeed, startCam, stopCam, camStream, camVideo,
} from "./camera.js";
import { enhanceSelect } from "./customSelect.js";
import { startLive, startScanAuto } from "./video.js";
import { renderAttachStrip } from "./attach.js";

// ---------- theme ----------
const THEMES = ["cyberpunk", "light", "dark", "ice", "matrix", "classic"];
const THEME_LABEL = {
  cyberpunk: "◐", light: "☀", dark: "☾", ice: "❄", matrix: "▣", classic: "▭",
};

function applyTheme(name) {
  if (!THEMES.includes(name)) name = "cyberpunk";
  const root = document.documentElement;
  for (const t of THEMES) if (t !== "cyberpunk") root.classList.remove("theme-" + t);
  if (name !== "cyberpunk") root.classList.add("theme-" + name);
  const btn = document.getElementById("theme-btn");
  if (btn) {
    btn.textContent = THEME_LABEL[name] || "◐";
    btn.title = `Theme: ${name} (click to cycle)`;
  }
  console.info("[theme] applied", name);
}

function cycleTheme() {
  const cur = localStorage.getItem("chatlm.theme") || "cyberpunk";
  const next = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
  localStorage.setItem("chatlm.theme", next);
  applyTheme(next);
}

applyTheme(localStorage.getItem("chatlm.theme") || "cyberpunk");
// Delegated so theme button works even if wiring runs before mount.
document.addEventListener("click", (e) => {
  if (e.target && e.target.closest && e.target.closest("#theme-btn")) cycleTheme();
});

// ---------- THINK toggle ----------
const thinkEl = document.getElementById("think-toggle");
function applyThink() {
  thinkEl.setAttribute("aria-pressed", toggles.think ? "true" : "false");
  const s = document.getElementById("s-think");
  if (s) s.textContent = toggles.think ? "ON" : "OFF";
}
thinkEl.addEventListener("click", () => {
  toggles.think = !toggles.think;
  localStorage.setItem("chatlm.think", toggles.think ? "1" : "0");
  applyThink();
});
applyThink();

// ---------- TOOLS + AUTO-APPROVE ----------
const toolsEl = document.getElementById("tools-toggle");
const autoApproveEl = document.getElementById("auto-approve");
const autoApproveWrap = document.getElementById("auto-approve-wrap");

function applyTools() {
  toolsEl.setAttribute("aria-pressed", toggles.tools ? "true" : "false");
  document.body.classList.toggle("tools-on", toggles.tools);
  // When tools are switched off, disarm auto-approve so re-enabling
  // tools later starts safe (avoids surprise auto-execution).
  if (!toggles.tools && toggles.autoApprove) {
    toggles.autoApprove = false;
    localStorage.setItem("chatlm.autoApprove", "0");
    autoApproveEl.checked = false;
    autoApproveWrap.classList.remove("armed");
  }
}
toolsEl.addEventListener("click", () => {
  toggles.tools = !toggles.tools;
  localStorage.setItem("chatlm.tools", toggles.tools ? "1" : "0");
  applyTools();
});
autoApproveEl.checked = toggles.autoApprove;
autoApproveWrap.classList.toggle("armed", toggles.autoApprove);
autoApproveEl.addEventListener("change", () => {
  toggles.autoApprove = autoApproveEl.checked;
  localStorage.setItem("chatlm.autoApprove", toggles.autoApprove ? "1" : "0");
  autoApproveWrap.classList.toggle("armed", toggles.autoApprove);
});
applyTools();

// ---------- GUARD toggle ----------
// Server holds the source-of-truth (so the model is actually loaded /
// unloaded inside Ollama). The button just reflects + flips it. We
// pull the initial state at boot and trust the response.
const guardEl = document.getElementById("guard-toggle");
function paintGuard(on) {
  guardEl.setAttribute("aria-pressed", on ? "true" : "false");
}
async function fetchGuardStatus() {
  try {
    const r = await fetch("/guard/status");
    if (r.ok) paintGuard(((await r.json()) || {}).enabled);
  } catch { /* server may be older — leave button off */ }
}
guardEl.addEventListener("click", async () => {
  const next = guardEl.getAttribute("aria-pressed") !== "true";
  // Optimistic paint so the click feels snappy; revert if the server
  // disagrees (e.g. ollama unreachable when trying to enable).
  paintGuard(next);
  try {
    const r = await fetch("/guard/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: next }),
    });
    const data = await r.json();
    paintGuard(!!data.enabled);
  } catch (err) {
    paintGuard(!next);
    console.warn("[guard] toggle failed", err);
  }
});
fetchGuardStatus();

// ---------- composer ----------
export function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
}

inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

inputEl.addEventListener("paste", async (e) => {
  const items = e.clipboardData?.items || [];
  for (const it of items) {
    if (it.type.startsWith("image/")) {
      e.preventDefault();
      const { fileToResizedB64 } = await import("./attach.js");
      pending.images.push(await fileToResizedB64(it.getAsFile()));
      renderAttachStrip();
    }
  }
});

// ---------- health + stats polling ----------
export async function loadHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    modelEl.textContent = data.default_model ?? "--";
    setStatus("ok", "LINK // READY");
  } catch {
    setStatus("err", "LINK // OFFLINE");
  }
}

async function pollStats() {
  try {
    const res = await fetch("/stats");
    if (!res.ok) return;
    const s = await res.json();
    if (s.model) {
      const rawName = s.model.name ?? "--";
      $("s-model").textContent = rawName.replace(/^mlx:/, "").slice(0, 28);
      $("s-backend").textContent = (s.model.backend ?? "--").toUpperCase();
      $("s-params").textContent = s.model.parameter_size ?? "--";
      $("s-quant").textContent = s.model.quantization ?? "--";
    }
    if (s.ollama) {
      const rss = `${(s.ollama.rss_mb / 1024).toFixed(2)} GB`;
      $("s-rss").textContent = rss;
      $("s-rss-inline").textContent = rss;
    } else {
      $("s-rss").textContent = "--";
      $("s-rss-inline").textContent = "--";
    }
    if (s.system) {
      const cpu = s.system.cpu_percent ?? 0;
      const cpuStr = `${cpu.toFixed(0)}%`;
      $("s-cpu").textContent = cpuStr;
      $("s-cpu-inline").textContent = cpuStr;
      setBar("s-cpu-bar", cpu);
      const mem = s.system.memory_percent ?? 0;
      const memStr = `${s.system.memory_used_gb}/${s.system.memory_total_gb}G`;
      $("s-mem").textContent = memStr;
      $("s-mem-inline").textContent = memStr;
      setBar("s-mem-bar", mem);
    }
  } catch { /* ignore */ }
}
setInterval(pollStats, 2000);
pollStats();

// ---------- mode switcher ----------
export async function setMode(mode) {
  const m = mode === "video" ? "video" : "chat";
  document.body.classList.remove("mode-chat", "mode-video");
  document.body.classList.add(`mode-${m}`);
  document.querySelectorAll(".mode-btn").forEach((b) => {
    b.setAttribute("aria-pressed", b.dataset.mode === m ? "true" : "false");
  });
  localStorage.setItem("chatlm.mode", m);
  if (m === "video") {
    await startCam();
    if (!camStream) return;
    // Wait for videoWidth/height to land before downstream captures.
    if (!camVideo.videoWidth) {
      await new Promise((r) => camVideo.addEventListener("loadedmetadata", r, { once: true }));
    }
    startLive();
    startScanAuto();
    // TRACK auto-starts inside renderChips once the first scan returns chips.
  } else {
    stopCam();
  }
}
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// ---------- clear / purge ----------
async function clearCurrent() {
  if (document.body.classList.contains("mode-video")) {
    clearFeed();
    return;
  }
  history.length = 0;
  pending.images.length = 0;
  renderAttachStrip();
  // Drop the active session server-side + create a fresh one so the wipe
  // survives refresh — Sessions.deleteSession auto-creates a new session
  // when the last one is removed.
  if (Sessions.activeId) {
    await Sessions.deleteSession(Sessions.activeId);
  } else {
    logEl.innerHTML = "";
    addMessage("sys", "Neural link purged. Memory wiped. Ready for new transmission.");
  }
  const last = document.getElementById("s-last");
  if (last) last.textContent = "--";
  inputEl.focus();
}
document.getElementById("clear").addEventListener("click", clearCurrent);

const memPurgeBtn = document.getElementById("mem-purge");
memPurgeBtn.addEventListener("click", async () => {
  memPurgeBtn.disabled = true;
  const originalLabel = memPurgeBtn.textContent;
  memPurgeBtn.textContent = "PURGING…";
  setStatus("busy", "LINK // PURGING MEMORY");
  try {
    const res = await fetch("/memory/flush", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    const msg = `[MEM PURGED] ollama=${d.ollama_unloaded} mlx=${d.mlx_unloaded}`;
    if (document.body.classList.contains("mode-video")) appendFeed(msg, "scan");
    else addMessage("sys", msg);
    setStatus("ok", "LINK // READY");
  } catch (err) {
    addMessage("sys", `[MEM PURGE FAILED] ${err.message}`);
    setStatus("err", "LINK // ERROR");
  } finally {
    memPurgeBtn.disabled = false;
    memPurgeBtn.textContent = originalLabel;
  }
});

// ---------- model dropdowns ----------
const selEmma = document.getElementById("select-emma");
const selScan = document.getElementById("select-scan");
const selDetector = document.getElementById("select-detector");
const selSegmenter = document.getElementById("select-segmenter");
const selInpaint = document.getElementById("select-inpaint");
const selTxt2img = document.getElementById("select-txt2img");

const allModelSelects = [selEmma, selScan, selDetector, selSegmenter, selInpaint, selTxt2img];
allModelSelects.forEach((s) => s && enhanceSelect(s));

// ---- option metadata: grouping + capability badges (client-side) ----
// Derived from the name/family/size the /models endpoint already returns,
// so no extra backend round-trips. Keeps the dropdown organized by family
// with backend + capability + "heavy" flags surfaced as chips.
const UNCENSORED_RE = /uncensored|abliterated|heretic|uncen\b/i;
const HEAVY_GB = 14; // models at/above this lock up smaller Macs

function backendOf(name) {
  if (name.startsWith("mlx:")) return "MLX";
  if (name.startsWith("llama:")) return "llama-srv";
  if (name.startsWith("talkie:")) return "talkie";
  if (name.startsWith("hf.co/")) return "Ollama·HF";
  return "Ollama";
}

// The model NAME is authoritative for family. The /models `family` field
// can't be trusted: the MLX backend reports "gemma4-mlx" for every MLX
// model, Qwen included. So classify by name tokens, family only as last
// resort.
function familyOf(name, family) {
  const n = name.toLowerCase();
  if (n.includes("guard")) return "guard";
  if (n.includes("qwen")) return "qwen";
  if (n.includes("gemma")) return "gemma";
  if (n.includes("llama")) return "llama";
  const f = (family || "").toLowerCase();
  if (f.includes("qwen")) return "qwen";
  if (f.includes("gemma")) return "gemma";
  if (f.includes("llama")) return "llama";
  return "other";
}

const FAMILY_GROUP = {
  gemma: { key: "Gemma 4", rank: 1 },
  qwen: { key: "Qwen", rank: 2 },
  llama: { key: "Llama", rank: 3 },
  other: { key: "Other", rank: 4 },
  guard: { key: "Safety / Guard", rank: 5 },
};

function familyGroup(name, family) {
  return FAMILY_GROUP[familyOf(name, family)] || FAMILY_GROUP.other;
}

// Capability badges. Prefer the model's REAL Ollama capabilities (sent by
// /models); only fall back to name-based guessing for backends with no
// capability metadata (MLX / llama-server / talkie).
function capBadges(m) {
  const caps = m.capabilities;
  if (Array.isArray(caps)) {
    const out = [];
    if (caps.includes("vision")) out.push("VISION");
    if (caps.includes("audio")) out.push("AUDIO");
    if (caps.includes("tools")) out.push("TOOLS");
    if (caps.includes("thinking")) out.push("THINK");
    return out;
  }
  // No server caps → MLX/llama-server/talkie. The app drops tool-calling
  // for these backends, so never claim TOOLS. Badge vision for families
  // that are multimodal (Gemma, *-VL).
  const n = m.name.toLowerCase();
  const out = [];
  if (n.includes("gemma") || n.includes("-vl") || n.includes("vision")) out.push("VISION");
  return out;
}

function shortName(name) {
  if (name.startsWith("mlx:")) return name.slice(4);
  if (name.startsWith("hf.co/")) {
    const parts = name.slice(6).split("/"); // user/repo:tag
    return parts[parts.length - 1] || name;
  }
  return name;
}

function ollamaMeta(m) {
  const name = m.name;
  const grp = familyGroup(name, m.family);
  const backend = backendOf(name);
  const uncensored = UNCENSORED_RE.test(name);
  const heavy = (m.size_gb || 0) >= HEAVY_GB;
  const badges = [];
  if (uncensored) badges.push("UNCENSORED");
  badges.push(...capBadges(m));
  badges.push(backend);
  if (heavy) badges.push("⚠ HEAVY");
  const params = m.parameter_size || "?";
  const size = m.size_gb != null ? `${m.size_gb} GB` : "";
  return {
    group: uncensored ? `${grp.key} · uncensored` : grp.key,
    groupRank: grp.rank * 10 + (uncensored ? 1 : 0),
    title: shortName(name),
    sub: [backend, params, size].filter(Boolean).join(" · "),
    badges: badges.join(","),
  };
}

function fillOllamaSelect(sel, models, current) {
  sel.innerHTML = "";
  if (!models.length) {
    const o = document.createElement("option");
    o.textContent = "(none installed)";
    o.disabled = true;
    sel.appendChild(o);
    sel._cs && sel._cs.refresh();
    return;
  }
  // Sort by group rank, then size ascending (smallest/safest first, the
  // heavy 26B variants sink to the bottom of their group).
  const decorated = models
    .map((m) => ({ m, meta: ollamaMeta(m) }))
    .sort((a, b) => a.meta.groupRank - b.meta.groupRank || (a.m.size_gb || 0) - (b.m.size_gb || 0));
  for (const { m, meta } of decorated) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = `${m.name} · ${m.parameter_size || "?"}`; // native fallback text
    o.dataset.group = meta.group;
    o.dataset.title = meta.title;
    o.dataset.sub = meta.sub;
    o.dataset.badges = meta.badges;
    o.title = m.name;
    if (m.name === current) o.selected = true;
    sel.appendChild(o);
  }
  sel._cs && sel._cs.refresh();
}

function fillPresetSelect(sel, presets, current) {
  sel.innerHTML = "";
  for (const p of presets) {
    const o = document.createElement("option");
    o.value = p.name;
    const label = p.label || p.name;
    o.textContent = label;
    o.dataset.group = p.kind || "Presets";
    o.dataset.title = label;
    const badges = [];
    if (p.fits === false) {
      o.disabled = true;
      badges.push("⚠ TOO LARGE");
    }
    o.dataset.badges = badges.join(",");
    if (p.name === current) o.selected = true;
    sel.appendChild(o);
  }
  sel._cs && sel._cs.refresh();
}

async function postModelChange(path, name, sel, storageKey) {
  sel.disabled = true;
  console.log(`[model] ${path} -> ${name}`);
  if (document.body.classList.contains("mode-video")) {
    appendFeed(`[MODEL] ${path.split("/").pop()} → ${name}`, "scan");
  }
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err);
    }
    if (storageKey) localStorage.setItem(storageKey, name);
    setStatus("ok", `SWITCHED // ${name.toUpperCase().slice(0, 24)}`);
  } catch (err) {
    setStatus("err", `SWITCH FAIL`);
    console.warn(`[model] ${path} failed:`, err);
    if (document.body.classList.contains("mode-video")) {
      appendFeed(`[MODEL ERR] ${err.message}`, "err");
    } else {
      addMessage("sys", `[MODEL ERR ${path}] ${err.message}`);
    }
  } finally {
    sel.disabled = false;
    setTimeout(() => setStatus("ok", "LINK // READY"), 1200);
  }
}

export async function refreshModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) return;
    const d = await res.json();
    fillOllamaSelect(selEmma, d.emma.available, d.emma.current);
    fillOllamaSelect(selScan, d.scan.available, d.scan.current);
    fillPresetSelect(selDetector, d.detector.presets, d.detector.current);
    fillPresetSelect(selSegmenter, d.segmenter.presets, d.segmenter.current);
    if (d.inpaint) fillPresetSelect(selInpaint, d.inpaint.presets, d.inpaint.current);
    if (d.txt2img) fillPresetSelect(selTxt2img, d.txt2img.presets, d.txt2img.current);

    const maybeRestore = async (key, currentValue, available, url, sel) => {
      const saved = localStorage.getItem(key);
      if (!saved || saved === currentValue) return;
      const match = available.some((m) => m.name === saved);
      if (!match) return;
      sel.value = saved;
      await postModelChange(url, saved, sel, key);
    };
    await maybeRestore(STORAGE_KEYS.emma, d.emma.current, d.emma.available, "/models/emma", selEmma);
    await maybeRestore(STORAGE_KEYS.scan, d.scan.current, d.scan.available, "/models/scan", selScan);
    await maybeRestore(STORAGE_KEYS.detector, d.detector.current, d.detector.presets, "/models/detector", selDetector);
    await maybeRestore(STORAGE_KEYS.segmenter, d.segmenter.current, d.segmenter.presets, "/models/segmenter", selSegmenter);
    if (d.inpaint) await maybeRestore(STORAGE_KEYS.inpaint, d.inpaint.current, d.inpaint.presets, "/models/inpaint", selInpaint);
    if (d.txt2img) await maybeRestore(STORAGE_KEYS.txt2img, d.txt2img.current, d.txt2img.presets, "/models/txt2img", selTxt2img);
    allModelSelects.forEach((s) => s && s._cs && s._cs.refresh());
  } catch { /* ignore */ }
}

selEmma.addEventListener("change", (e) => postModelChange("/models/emma", e.target.value, selEmma, STORAGE_KEYS.emma));
selScan.addEventListener("change", (e) => postModelChange("/models/scan", e.target.value, selScan, STORAGE_KEYS.scan));
selDetector.addEventListener("change", (e) => postModelChange("/models/detector", e.target.value, selDetector, STORAGE_KEYS.detector));
selSegmenter.addEventListener("change", (e) => postModelChange("/models/segmenter", e.target.value, selSegmenter, STORAGE_KEYS.segmenter));
selInpaint.addEventListener("change", (e) => postModelChange("/models/inpaint", e.target.value, selInpaint, STORAGE_KEYS.inpaint));
selTxt2img.addEventListener("change", (e) => postModelChange("/models/txt2img", e.target.value, selTxt2img, STORAGE_KEYS.txt2img));
