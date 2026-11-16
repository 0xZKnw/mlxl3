import { pickLanguage, translator } from "./i18n.js";
import { engineMessages, escapeHtml, formatBytes, formatStats, renderMarkdown, splitThinking } from "./render.js";
import {
  abortTurn, applyEngineEvent, engineExited, engineStarted, initialSession, newConversation,
  restoreTurns, serializeTurns, startTurn, unloaded,
} from "./session.js";

const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const listen = tauri.event.listen;
const t = translator(pickLanguage(navigator.languages));
const $ = (id) => document.getElementById(id);
const STORE_KEY = "mlxl3.conversation.v1";
const PREFS_KEY = "mlxl3.prefs.v1";

let session = initialSession();
let system = null;
let models = [];
let loadedId = null;
let busy = false;

function storage(action) {
  try { return action(window.localStorage); } catch { return null; }
}

function log(line) {
  const el = $("log");
  el.hidden = false;
  el.textContent = (el.textContent + line + "\n").slice(-20000);
  el.scrollTop = el.scrollHeight;
}

function fail(error) {
  const message = typeof error === "string" ? error : error?.message ?? String(error);
  log(`error: ${message}`);
  $("engine-state").textContent = message;
}

function backend() { return $("backend").value; }

function sampling() {
  const num = (id) => Number($(id).value);
  return {
    max_tokens: Math.trunc(num("max-tokens")) || -1,
    temperature: num("temperature"),
    top_k: Math.max(0, Math.trunc(num("top-k"))),
    top_p: num("top-p"),
    min_p: num("min-p"),
    repetition_penalty: num("rep"),
    enable_thinking: $("thinking").checked,
  };
}

function savePrefs() {
  const prefs = { backend: backend(), distro: $("distro").value, context: $("context").value, system: $("system").value };
  for (const id of ["temperature", "top-p", "top-k", "min-p", "rep", "max-tokens"]) prefs[id] = $(id).value;
  prefs.thinking = $("thinking").checked;
  storage((s) => s.setItem(PREFS_KEY, JSON.stringify(prefs)));
}

function loadPrefs() {
  let prefs = {};
  try { prefs = JSON.parse(storage((s) => s.getItem(PREFS_KEY)) || "{}") || {}; } catch { prefs = {}; }
  for (const [id, key] of [["context", "context"], ["system", "system"], ["temperature", "temperature"], ["top-p", "top-p"],
    ["top-k", "top-k"], ["min-p", "min-p"], ["rep", "rep"], ["max-tokens", "max-tokens"]]) {
    if (typeof prefs[key] === "string") $(id).value = prefs[key];
  }
  if (typeof prefs.thinking === "boolean") $("thinking").checked = prefs.thinking;
  return prefs;
}

function setSession(next) {
  const before = session;
  session = next;
  if (before.turns !== next.turns && !next.active) storage((s) => s.setItem(STORE_KEY, serializeTurns(next.turns)));
  render();
}

// ---------- rendering ----------

function renderTurn(turn) {
  if (turn.role === "user") return `<div class="msg user">${escapeHtml(turn.content).replace(/\n/g, "<br>")}</div>`;
  const { thinking, answer, open } = splitThinking(turn.content);
  let html = "";
  if (thinking !== null) {
    html += `<details class="think"${open ? " open" : ""}><summary>${escapeHtml(t("reasoning"))}</summary>${renderMarkdown(thinking)}</details>`;
  }
  html += renderMarkdown(answer);
  if (!turn.complete && !turn.failed && !turn.cancelled && !answer && thinking === null) {
    html += `<p class="muted">${escapeHtml(session.phase === "prefill" ? t("prefill") : "…")}</p>`;
  }
  if (turn.failed) html += `<p class="error">${escapeHtml(turn.failed)}</p>`;
  if (turn.cancelled) html += `<p class="muted">■</p>`;
  if (turn.stats) html += `<p class="stats">${escapeHtml(formatStats(turn.stats))}</p>`;
  return `<div class="msg assistant">${html}</div>`;
}

function render() {
  const box = $("messages");
  const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.innerHTML = session.turns.map(renderTurn).join("");
  if (pinned) box.scrollTop = box.scrollHeight;

  const info = session.info;
  const state = {
    idle: t("noModel"),
    loading: `${t("loading")}…`,
    ready: info ? `${t("ready")} · ${info.device} · ${info.backend} · ctx ${info.context_limit}` : t("ready"),
    failed: session.error || t("engineStopped"),
    stopped: session.error || t("engineStopped"),
  }[session.engine];
  $("engine-state").textContent = state;
  $("stats").textContent = formatStats(session.lastStats);
  $("send").hidden = Boolean(session.active);
  $("stop").hidden = !session.active;
  $("send").disabled = session.engine !== "ready";
  $("unload").disabled = session.engine === "idle";
  $("new-chat").disabled = Boolean(session.active);
  renderModels();
}

function renderModels() {
  const list = $("models");
  list.innerHTML = "";
  for (const model of models) {
    const li = document.createElement("li");
    li.className = model.id === loadedId && session.engine !== "idle" ? "model loaded" : "model";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = model.name;
    const meta = document.createElement("div");
    meta.className = "muted";
    meta.textContent = [model.architecture, model.bits ? `${model.bits} bpw` : "", formatBytes(model.size_bytes)].filter(Boolean).join(" · ");
    meta.title = model.path;
    const actions = document.createElement("div");
    actions.className = "row";
    const load = document.createElement("button");
    load.textContent = t("load");
    load.disabled = busy || Boolean(session.active);
    load.onclick = () => loadModel(model.id);
    const remove = document.createElement("button");
    remove.textContent = t("remove");
    remove.className = "secondary";
    remove.disabled = model.id === loadedId && session.engine !== "idle";
    remove.onclick = async () => {
      if (!window.confirm(t("confirmRemove"))) return;
      try { await invoke("remove_model", { id: model.id }); await refreshModels(); } catch (e) { fail(e); }
    };
    actions.append(load, remove);
    li.append(title, meta, actions);
    list.append(li);
  }
}

function renderRuntime() {
  if (!system) return;
  const key = backend();
  const marker = system.runtimes[key];
  $("distro-row").hidden = key !== "rocm_wsl";
  $("runtime-status").textContent = marker
    ? `${t("runtimeReady")} · ${marker.probe?.devices?.[0]?.name ?? ""} · ${marker.id}`
    : t("runtimeMissing");
  $("install-runtime").textContent = marker ? t("reinstall") : t("install");
  $("install-runtime").disabled = busy || (key === "rocm_wsl" && !$("distro").value);
}

function setBusy(value) {
  busy = value;
  for (const id of ["download", "add-model"]) $(id).disabled = value;
  $("runtime-progress").hidden = !value;
  renderRuntime();
  renderModels();
}

// ---------- actions ----------

async function refreshSystem() {
  system = await invoke("system_info");
  $("app-version").textContent = system.app_version;
  const gpus = [
    ...system.nvidia.map((g) => `${g.name} (${Math.round(g.memory_mib / 1024)} GB, driver ${g.driver}${g.driver_ok ? "" : " ⚠"})`),
    ...system.amd,
  ];
  $("gpu-summary").textContent = `${gpus.join(" · ") || t("noGpu")} — ${system.recommendation.reason}`;
  const distro = $("distro");
  distro.innerHTML = "";
  for (const name of system.wsl_distros) distro.append(new Option(name, name));
  if (!system.wsl_distros.length) distro.append(new Option(t("noDistro"), ""));
  renderRuntime();
}

async function refreshModels() {
  models = await invoke("list_models");
  renderModels();
}

async function installRuntime() {
  setBusy(true);
  try {
    await invoke("install_runtime", { backend: backend(), distro: backend() === "rocm_wsl" ? $("distro").value : null });
    await refreshSystem();
  } catch (e) { fail(e); } finally { setBusy(false); }
}

async function addModel() {
  try {
    const path = await tauri.dialog.open({ directory: true, multiple: false });
    if (!path) return;
    await invoke("add_model", { path });
    await refreshModels();
  } catch (e) { fail(e); }
}

async function downloadModel() {
  const repo = $("repo").value.trim();
  if (!repo) return;
  setBusy(true);
  try {
    await invoke("download_model", { backend: backend(), repo, folder: $("folder").value.trim() });
    await refreshModels();
  } catch (e) { fail(e); } finally { setBusy(false); }
}

async function loadModel(id) {
  savePrefs();
  loadedId = id;
  const context = Math.max(0, Math.trunc(Number($("context").value) || 0));
  try {
    await invoke("load_model", { id, backend: backend(), contextLength: context });
  } catch (e) {
    loadedId = null;
    fail(e);
  }
  render();
}

async function unloadModel() {
  try { await invoke("unload_model"); } catch (e) { fail(e); }
  loadedId = null;
  setSession(unloaded(session));
}

async function send(event) {
  event.preventDefault();
  const prompt = $("prompt").value;
  if (!prompt.trim() || session.engine !== "ready" || session.active) return;
  savePrefs();
  const requestId = `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  let next;
  try { next = startTurn(session, requestId, prompt); } catch (e) { fail(e); return; }
  const messages = engineMessages($("system").value, next.turns.slice(0, -1));
  setSession(next);
  $("prompt").value = "";
  try {
    await invoke("generate", { requestId, messages, sampling: sampling() });
  } catch (e) {
    setSession(abortTurn(session, requestId, String(e)));
    $("prompt").value = prompt;
  }
}

async function stop() {
  if (!session.active) return;
  try { await invoke("cancel", { requestId: session.active }); } catch (e) { fail(e); }
}

// ---------- wiring ----------

function translate() {
  document.documentElement.lang = pickLanguage(navigator.languages);
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
}

async function main() {
  translate();
  const prefs = loadPrefs();
  session = { ...session, turns: restoreTurns(storage((s) => s.getItem(STORE_KEY)) || "[]") };

  await listen("engine-started", ({ payload }) => setSession(engineStarted(session, payload.epoch)));
  await listen("engine-event", ({ payload }) => setSession(applyEngineEvent(session, payload.epoch, payload.event)));
  await listen("engine-exited", ({ payload }) => {
    const tail = (payload.stderr_tail || "").trim().split("\n").slice(-3).join("\n");
    if (payload.stderr_tail) log(payload.stderr_tail);
    setSession(engineExited(session, payload.epoch, `${t("engineStopped")} (code ${payload.code ?? "?"})${tail ? `: ${tail}` : ""}`));
  });
  await listen("engine-log", ({ payload }) => log(payload.line));
  await listen("runtime-log", ({ payload }) => log(payload.line));
  await listen("download-log", ({ payload }) => log(payload.line));
  await listen("runtime-progress", ({ payload }) => {
    const bar = $("runtime-progress");
    bar.max = payload.total;
    bar.value = payload.step;
    log(`[${payload.step}/${payload.total}] ${payload.label}`);
  });

  $("backend").onchange = () => { savePrefs(); renderRuntime(); };
  $("distro").onchange = () => { savePrefs(); renderRuntime(); };
  $("install-runtime").onclick = installRuntime;
  $("add-model").onclick = addModel;
  $("download").onclick = downloadModel;
  $("unload").onclick = unloadModel;
  $("composer").onsubmit = send;
  $("stop").onclick = stop;
  $("new-chat").onclick = () => { try { setSession(newConversation(session)); } catch (e) { fail(e); } };
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });

  render();
  await refreshModels().catch(fail);
  await refreshSystem().catch(fail);
  const preferred = prefs.backend || (system?.recommendation?.backend ?? "cuda");
  if (["cuda", "rocm_wsl"].includes(preferred)) $("backend").value = preferred;
  if (prefs.distro && system?.wsl_distros.includes(prefs.distro)) $("distro").value = prefs.distro;
  renderRuntime();
}

main().catch(fail);
