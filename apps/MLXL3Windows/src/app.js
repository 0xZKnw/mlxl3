// MLXL3 Desktop for Windows: the Mac 2.0.3 layout on top of the Tauri shell commands.
import { applyAppearance, DEFAULT_APPEARANCE, idleSeconds, IDLE_DELAYS, normalizeAppearance, normalizeHex } from "./appearance.js";
import { paint as paintAurora } from "./aurora.js";
import { pickLanguage, translator } from "./i18n.js";
import { icon, monogram } from "./icons.js";
import { addSample, chartPoints, exportJson, history, labSample, latestByModel, METRICS, models as labModels, restoreSamples } from "./lab.js";
import { engineMessages, escapeHtml, formatBytes, formatStats, renderMarkdown, splitThinking } from "./render.js";
import {
  abortTurn, applyEngineEvent, engineExited, engineStarted, initialSession, newConversation, restoreTurns, rewind,
  startTurn, unloaded,
} from "./session.js";
import {
  backupJson, branchOf, filterConversations, formatCount, markdownExport, mergeBackup, greeting, modelSubtitle, newId, percent, pickVariant, recommendedChoices,
  RECOMMENDED, removeConversation, restoreConversations, revisionLabel, sections, serializeConversations, splitRepo,
  titleFrom, togglePinned, updateBadge, upsertConversation,
} from "./store.js";
import {
  admitAttachments, attachmentContext, composeSystem, documentContext, MAX_EXCERPTS, normalizeProfile, normalizeProject, normalizeWorkspace,
} from "./workspace.js";

const tauri = window.__TAURI__;
const invoke = tauri.core.invoke;
const listen = tauri.event.listen;
const $ = (id) => document.getElementById(id);
const CONVERSATIONS_KEY = "mlxl3.conversations.v2";
const LEGACY_KEY = "mlxl3.conversation.v1";
const PREFS_KEY = "mlxl3.prefs.v2";
const SAMPLING_DEFAULTS = { temperature: "0.6", "top-p": "0.95", "top-k": "20", "min-p": "0", rep: "1", "max-tokens": "-1" };
const WORKSPACE_KEY = "mlxl3.workspace.v1";
const LAB_KEY = "mlxl3.lab.v1";
const EMPTY_PROMPT = "Analyse les fichiers joints."; // Mac: sent to the model in both UI languages
const IMAGE_LIMIT = { count: 8, bytes: 8 * 1024 * 1024 };
const AGENT_MAX_TOKENS = 4096;
const AGENT_TOTAL_MS = 600_000;
const AGENT_SILENT_MS = 120_000;
const HUB_PAGE = 60;
const HUB_MAX = 200;

let prefs = {};
let language = "en";
let t = translator("en");
let session = initialSession();
let conversations = [];
let currentId = null;
let system = null;
let models = [];
let loadedId = null;
let busy = false;
let updateStatus = null;
let download = null; // { id, name, repo, done, total }
let pending = [];
let hub = { query: "", results: [], limit: HUB_PAGE, searching: false, error: null, seq: 0, detail: null, variantId: null, loading: false };
let modelsTab = "library";
let inspectorOpen = false;
let appearance = normalizeAppearance(null);
let lastModel = null; // Mac "Last selected model": the id most recently loaded
let startupModel = ""; // "" = last selected model
let idleTimer = null;
let idleDeadline = 0;
let autoLoaded = false;
let account = null;
let mode = "chat"; // chat | agent | server | lab (the main view)
let conversationMode = "chat"; // chat | agent: which conversations the sidebar lists
let workspace = { projects: [], profiles: [] };
let selectedProject = null;
let attachments = []; // composer files waiting to be sent
let importing = false;
let searchingDocs = false;
let mcpEnabled = false;
let mcpStatus = null;
let mcpBusy = false;
let mcpCatalog = []; // tool names from the last connection or test
let mcpEditing = null; // name of the server in the editor, "" for a new one
let agentInfo = null; // workspace of the current Agent task
let agentTimers = null;
let serverStatus = null;
let serverKey = "";
let serverBusy = false;
let labSamples = [];
let wsTab = "projects";
let docEntries = [];
let docResults = null;
const MENUS = ["model-list", "tools-list", "display-pop", "agent-pop"];
const greetingSeed = Math.floor(Math.random() * 1e6);
const openThinking = new Map(); // "conversation:index" → user-chosen open state of a reasoning block
const renderedTurns = new WeakMap();

// ---------- small utilities ----------

function storage(action) {
  try { return action(window.localStorage); } catch { return null; }
}

function message(error) {
  return typeof error === "string" ? error : error?.message ?? String(error);
}

function log(line) {
  const el = $("log");
  el.textContent = (el.textContent + line + "\n").slice(-30000);
  el.scrollTop = el.scrollHeight;
}

let toastTimer = null;
function toast(text, ms = 4200) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function fail(error) {
  const text = message(error);
  log(`error: ${text}`);
  toast(text, 7000);
}

function fillIcons(root = document) {
  for (const el of root.querySelectorAll("[data-icon]")) {
    if (el.dataset.filled === el.dataset.icon) continue;
    const svg = icon(el.dataset.icon);
    if (el.tagName === "SPAN") el.outerHTML = svg;
    else { el.insertAdjacentHTML("afterbegin", svg); el.dataset.filled = el.dataset.icon; }
  }
  for (const el of root.querySelectorAll("[data-logo]")) {
    el.innerHTML = monogram(Number(el.dataset.logo) || 20);
  }
}

function setIcon(button, name) {
  button.querySelector("svg")?.remove();
  button.insertAdjacentHTML("afterbegin", icon(name));
  button.dataset.filled = name;
}

function translate() {
  document.documentElement.lang = language;
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-placeholder]")) el.placeholder = t(el.dataset.i18nPlaceholder);
  for (const el of document.querySelectorAll("[data-i18n-title]")) {
    el.title = t(el.dataset.i18nTitle);
    el.setAttribute("aria-label", el.title);
  }
}

// ---------- preferences ----------

function backend() { return $("backend").value; }

function savePrefs() {
  const next = {
    backend: backend(), distro: $("distro").value, context: $("context").value, system: $("system").value,
    thinking: $("thinking").checked, language: $("language").value, theme: $("theme").value,
    sidebar: !$("chat-view").classList.contains("collapsed"),
    appearance, notifications: $("notifications").checked, loadOnStartup: $("load-on-startup").checked,
    startupModel, idleUnload: $("idle-unload").value, lastModel,
    schema: $("schema").value, mcpEnabled, conversationMode, project: selectedProject,
  };
  for (const id of Object.keys(SAMPLING_DEFAULTS)) next[id] = $(id).value;
  prefs = next;
  storage((s) => s.setItem(PREFS_KEY, JSON.stringify(next)));
}

function loadPrefs() {
  try { prefs = JSON.parse(storage((s) => s.getItem(PREFS_KEY)) || "{}") || {}; } catch { prefs = {}; }
  for (const id of [...Object.keys(SAMPLING_DEFAULTS), "context", "system"]) {
    if (typeof prefs[id] === "string") $(id).value = prefs[id];
  }
  if (typeof prefs.thinking === "boolean") $("thinking").checked = prefs.thinking;
  if (["", "en", "fr"].includes(prefs.language)) $("language").value = prefs.language;
  if (["", "dark", "light"].includes(prefs.theme)) $("theme").value = prefs.theme;
  if (prefs.sidebar === false) $("chat-view").classList.add("collapsed");
  appearance = normalizeAppearance(prefs.appearance);
  $("notifications").checked = prefs.notifications === true;
  $("load-on-startup").checked = prefs.loadOnStartup !== false; // Mac default: on
  startupModel = typeof prefs.startupModel === "string" ? prefs.startupModel : "";
  lastModel = typeof prefs.lastModel === "string" ? prefs.lastModel : null;
  $("idle-unload").value = Object.hasOwn(IDLE_DELAYS, prefs.idleUnload) ? prefs.idleUnload : "15m";
  if (typeof prefs.schema === "string") $("schema").value = prefs.schema.slice(0, 64 * 1024);
  mcpEnabled = prefs.mcpEnabled === true; // Mac default: off
  conversationMode = prefs.conversationMode === "agent" ? "agent" : "chat";
  mode = conversationMode;
  selectedProject = typeof prefs.project === "string" ? prefs.project : null;
}

function loadWorkspace() {
  try { workspace = normalizeWorkspace(JSON.parse(storage((s) => s.getItem(WORKSPACE_KEY)) || "{}")); } catch { workspace = normalizeWorkspace(null); }
  if (!workspace.projects.some((p) => p.id === selectedProject)) selectedProject = null;
  labSamples = restoreSamples(storage((s) => s.getItem(LAB_KEY)) || "[]");
}

function persistWorkspace() {
  workspace = normalizeWorkspace(workspace);
  storage((s) => s.setItem(WORKSPACE_KEY, JSON.stringify(workspace)));
}

function currentProject() {
  return workspace.projects.find((p) => p.id === selectedProject) ?? null;
}

// ---------- appearance (Mac Settings → Appearance and the header display control) ----------

function renderAppearance() {
  appearance = applyAppearance(document.documentElement, appearance);
  for (const group of document.querySelectorAll(".segmented[data-set], .swatches[data-set]")) {
    for (const button of group.querySelectorAll("[data-value]")) {
      button.classList.toggle("selected", appearance[group.dataset.set] === button.dataset.value);
    }
  }
  for (const select of document.querySelectorAll("select[data-set]")) select.value = appearance[select.dataset.set];
  for (const range of document.querySelectorAll('input[data-set="textSize"]')) range.value = String(appearance.textSize);
  for (const out of document.querySelectorAll(".text-size-value")) out.textContent = `${appearance.textSize} pt`;
  const custom = appearance.accent === "custom";
  if (document.activeElement !== $("custom-accent")) $("custom-accent").value = `#${appearance.custom.toLowerCase()}`;
  $("custom-selected").hidden = !custom;
  $("use-custom-accent").hidden = custom && normalizeHex($("custom-accent").value) === appearance.custom;
  document.querySelector(".custom-accent").classList.toggle("selected", custom);
}

function setAppearance(change) {
  appearance = normalizeAppearance({ ...appearance, ...change });
  renderAppearance();
  savePrefs();
  if (inspectorOpen) setInspector(true);
}

function onAppearanceClick(event) {
  const button = event.target.closest("[data-set] [data-value]");
  if (button) {
    event.stopPropagation();
    setAppearance({ [button.closest("[data-set]").dataset.set]: button.dataset.value });
  } else if (event.target.closest("[data-reset-text-size]")) {
    event.stopPropagation();
    setAppearance({ textSize: DEFAULT_APPEARANCE.textSize });
  }
}

// ---------- models & memory (Mac startup model and ModelIdleUnloader) ----------

function renderStartupModels() {
  const select = $("startup-model");
  select.innerHTML = "";
  select.append(new Option(t("lastSelectedModel"), ""));
  for (const m of models) select.append(new Option(m.name, m.id));
  if (startupModel && !models.some((m) => m.id === startupModel)) {
    select.append(new Option(`${startupModel} · ${t("modelUnavailable")}`, startupModel));
  }
  select.value = startupModel;
  select.disabled = !$("load-on-startup").checked;
}

async function autoLoadOnStartup() {
  if (autoLoaded) return;
  autoLoaded = true;
  if (!$("load-on-startup").checked || session.engine !== "idle" || !system?.runtimes?.[backend()]) return;
  const id = startupModel || lastModel;
  if (id && models.some((m) => m.id === id)) await loadModel(id, { quiet: true });
}

function idleBlocked() {
  return session.engine !== "ready" || Boolean(session.active) || busy;
}

// The deadline starts when the model becomes idle; re-rendering does not extend it.
function updateIdleUnload() {
  const delay = idleSeconds($("idle-unload").value);
  if (!delay || idleBlocked()) {
    clearTimeout(idleTimer);
    idleTimer = null;
    idleDeadline = 0;
    return;
  }
  if (idleDeadline) return;
  idleDeadline = Date.now() + delay * 1000;
  idleTimer = setTimeout(async () => {
    idleTimer = null;
    idleDeadline = 0;
    if (idleBlocked()) { updateIdleUnload(); return; }
    await unloadModel();
    toast(t("autoUnloaded"));
  }, delay * 1000);
}

function resetIdleUnload() {
  clearTimeout(idleTimer);
  idleTimer = null;
  idleDeadline = 0;
  updateIdleUnload();
}

// ---------- conversations export / import (Mac Settings → Conversations, + menu export) ----------

async function exportCurrentConversation() {
  closeMenus();
  const turns = session.turns.length ? session.turns : conversations.find((c) => c.id === currentId)?.turns ?? [];
  if (!turns.length) { toast(t("noConversation")); return; }
  const current = conversations.find((c) => c.id === currentId);
  const markdown = markdownExport({ title: current?.title ?? titleFrom(turns, t("newChat")), turns },
    (content) => splitThinking(content).answer.trim());
  try {
    if (await invoke("save_text_file", { name: "MLXL3-conversation.md", contents: markdown })) toast(t("exported"));
  } catch (e) { fail(e); }
}

async function exportHistory() {
  if (!conversations.length) { toast(t("noConversation")); return; }
  try {
    if (await invoke("save_text_file", { name: "MLXL3-conversations.json", contents: backupJson(conversations) })) toast(t("exported"));
  } catch (e) { fail(e); }
}

async function importHistory() {
  try {
    const raw = await invoke("open_backup_file");
    if (raw == null) return;
    const { list, added } = mergeBackup(conversations, raw);
    conversations = list;
    persistConversations();
    render();
    toast(added === 1 ? t("imported1") : t("imported").replace("{n}", String(added)));
  } catch (e) { fail(`${t("importFailed")}: ${message(e)}`); }
}

// ---------- Hugging Face account (Mac Settings → Connections) ----------

function renderAccount() {
  const source = account?.source ?? null;
  $("hf-status").textContent = source === "environment" ? t("signedInEnv")
    : source === "file" ? (account.name ? t("signedInAs").replace("{name}", account.name) : t("signedIn"))
      : t("signedOut");
  $("hf-sign-out").disabled = source !== "file";
}

async function refreshAccount() {
  account = await invoke("hf_account").catch(() => null);
  renderAccount();
}

async function signIn() {
  const token = $("hf-token").value.trim();
  if (!token.startsWith("hf_") || token.length > 4000) { toast(t("hfInvalid")); return; }
  $("hf-sign-in").disabled = true;
  try {
    account = await invoke("hf_sign_in", { token });
    $("hf-token").value = "";
    renderAccount();
  } catch (e) { fail(e); } finally { $("hf-sign-in").disabled = false; }
}

async function signOut() {
  try { account = await invoke("hf_sign_out"); renderAccount(); } catch (e) { fail(e); }
}

function applyLanguage() {
  language = $("language").value || pickLanguage(navigator.languages);
  t = translator(language);
  translate();
}

function applyTheme() {
  const theme = $("theme").value;
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  requestAnimationFrame(() => paintAurora($("aurora")));
}

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

function renderSliders() {
  for (const id of ["temperature", "top-p", "top-k", "min-p", "rep"]) $(`${id}-value`).textContent = $(id).value;
  $("thinking-quick").checked = $("thinking").checked;
}

// ---------- conversations ----------

function persistConversations() {
  storage((s) => s.setItem(CONVERSATIONS_KEY, serializeConversations(conversations)));
}

function restoreState() {
  const raw = storage((s) => s.getItem(CONVERSATIONS_KEY));
  conversations = raw ? restoreConversations(raw) : [];
  if (!raw) {
    // One-time migration of the single conversation kept by the first Windows build.
    const legacy = restoreTurns(storage((s) => s.getItem(LEGACY_KEY)) || "[]");
    if (legacy.length) {
      conversations = [{ id: newId(), title: titleFrom(legacy, t("newChat")), updated: Date.now(), pinned: false, turns: legacy }];
      persistConversations();
      storage((s) => s.removeItem(LEGACY_KEY));
    }
  }
}

function saveCurrent(turns) {
  const kept = turns.filter((x) => x.complete);
  if (!kept.length) return;
  if (!currentId) currentId = newId();
  const existing = conversations.find((c) => c.id === currentId);
  const folder = agentInfo?.task === currentId && agentInfo.source ? agentInfo.source.split(/[\\/]/).filter(Boolean).pop() : null;
  conversations = upsertConversation(conversations, {
    id: currentId, title: existing?.title ?? folder ?? titleFrom(kept, t("newChat")), updated: Date.now(), pinned: Boolean(existing?.pinned), turns: kept,
    projectId: existing ? existing.projectId ?? null : selectedProject, agent: existing ? Boolean(existing.agent) : conversationMode === "agent",
  });
  persistConversations();
}

function setSession(next) {
  const before = session;
  session = next;
  if (before.turns !== next.turns && !next.active) saveCurrent(next.turns);
  render();
}

function openConversation(id) {
  if (session.active) { toast(t("stop")); return; }
  const conversation = conversations.find((c) => c.id === id);
  if (!conversation) return;
  stopSpeech();
  currentId = id;
  session = { ...session, turns: conversation.turns, lastStats: null, error: null };
  showPage("chat");
  render();
  scrollToBottom();
  if (conversationMode === "agent") refreshAgent();
}

function startNewConversation() {
  try {
    session = newConversation(session);
    stopSpeech();
    currentId = null;
    agentInfo = null;
    if (mode !== "chat" && mode !== "agent") mode = conversationMode;
    showPage("chat");
    render();
    $("prompt").focus();
  } catch (e) { fail(e); }
}

/** Mac sidebar filter: the conversations of the current mode (Chat or Agent) and project. */
function visibleConversations() {
  return conversations.filter((c) => Boolean(c.agent) === (conversationMode === "agent") && (c.projectId ?? null) === selectedProject);
}

function renderChats() {
  const query = $("chat-search").value;
  const list = filterConversations(visibleConversations(), query);
  const { pinned, recent } = sections(list);
  const shown = visibleConversations().length;
  $("chat-count").textContent = shown ? String(shown) : "";
  const row = (c) => `<div class="chat-row${c.id === currentId ? " selected" : ""}" data-chat="${escapeHtml(c.id)}" role="button" tabindex="0">
      <span class="title">${escapeHtml(c.title)}</span>
      <span class="row-tools${c.pinned ? " on" : ""}">
        <button class="mini${c.pinned ? " on" : ""}" data-pin="${escapeHtml(c.id)}" title="${escapeHtml(c.pinned ? t("unpin") : t("pin"))}">${icon("pin", 14)}</button>
        <button class="mini" data-delete="${escapeHtml(c.id)}" title="${escapeHtml(t("deleteChat"))}">${icon("trash", 14)}</button>
      </span></div>`;
  let html = "";
  if (pinned.length) html += `<div class="chat-group">${escapeHtml(t("pinned"))}</div>${pinned.map(row).join("")}`;
  if (recent.length) html += `${pinned.length ? `<div class="chat-group">${escapeHtml(t("recent"))}</div>` : ""}${recent.map(row).join("")}`;
  $("chats").innerHTML = html;
}

// ---------- messages ----------

function codeCards(html) {
  return html
    .replace(/<pre(?: data-lang="([^"]*)")?>/g, (_, lang) => `<div class="code-card"><div class="code-head"><span>${lang || "code"}</span><span class="spacer"></span><button class="mini" data-action="copy-code" title="${escapeHtml(t("copy"))}">${icon("copy", 14)}</button></div><pre>`)
    .replace(/<\/pre>/g, "</pre></div>");
}

function stripFrontMatter(text) {
  return String(text).replace(/^---\n[\s\S]*?\n---\n/, "");
}

/** Chips of the files attached to a user message; images keep a thumbnail while their bytes are in memory. */
function attachmentChips(turn, index) {
  const images = turn.images ?? [];
  const chips = (turn.attachments ?? []).map((a, i) => {
    const image = a.kind === "image" ? images.find((x) => x.name === a.name) : null;
    if (image && /^[A-Za-z0-9+/=]+$/.test(image.data) && /^image\/(png|jpeg|webp)$/.test(image.mime)) {
      return `<button class="thumb large" data-action="preview-image" data-attachment="${i}" title="${escapeHtml(t("previewImage"))}"><img alt="${escapeHtml(a.name)}" src="data:${escapeHtml(image.mime)};base64,${image.data}"></button>`;
    }
    return `<span class="file-chip">${icon(a.kind === "image" ? "photo" : "doc", 13)}<span>${escapeHtml(a.name)}</span></span>`;
  }).join("");
  return chips ? `<div class="chip-row" data-turn="${index}">${chips}</div>` : "";
}

function toolRows(tools) {
  return tools.map((tool, i) => {
    const badge = { running: ["toolRunning", "busy"], complete: ["toolDone", "ready"], failed: ["toolError", "failed"] }[tool.status];
    const server = tool.server ? `MCP · ${tool.server}` : "MCP";
    const output = typeof tool.result === "string" && tool.result
      ? `<pre class="tool-output clamp" data-tool="${i}">${escapeHtml(tool.result)}</pre>
        <div class="tool-actions"><button class="text-button" data-action="tool-more">${escapeHtml(t("showFullOutput"))}</button>
        <button class="mini" data-action="tool-copy" data-tool="${i}" title="${escapeHtml(t("copy"))}">${icon("copy", 13)}</button></div>` : "";
    return `<details class="tool-row"${tool.status === "failed" ? " open" : ""}><summary>${icon("tool", 13)}<span class="tool-name">${escapeHtml(tool.name)}</span>
      <span class="quiet small">${escapeHtml(server)}</span><span class="spacer"></span><span class="badge ${badge[1]}">${escapeHtml(t(badge[0]))}</span></summary>${output}</details>`;
  }).join("");
}

/** Engine progress labels, localized like the Mac (§0.4). */
function statusText(text) {
  const keys = { "Preparing tool call": "preparingToolCall", "Reading MCP results": "readingMcpResults", "Retrying JSON": "retryingJson" };
  return keys[text] ? t(keys[text]) : null;
}

function turnHtml(turn, index, last) {
  if (turn.role === "user") {
    return `<div class="turn user" data-index="${index}">
      <div class="who">${escapeHtml(t("you"))}</div>
      ${attachmentChips(turn, index)}
      ${turn.content ? `<div class="bubble">${escapeHtml(turn.content)}</div>` : ""}
      <div class="actions">
        <button class="mini" data-action="copy" title="${escapeHtml(t("copy"))}">${icon("copy", 14)}</button>
        <button class="mini" data-action="edit" title="${escapeHtml(t("edit"))}">${icon("pencil", 14)}</button>
        ${conversationMode === "agent" ? "" : `<button class="mini" data-action="branch" title="${escapeHtml(t("branchFromHere"))}">${icon("branch", 14)}</button>`}
      </div></div>`;
  }
  const streaming = !turn.complete && !turn.failed && !turn.cancelled;
  const { thinking, answer, open } = splitThinking(turn.content);
  let html = `<div class="who">${monogram(15)}<span>MLXL3</span>${streaming ? '<span class="streaming"><i></i><i></i><i></i></span>' : ""}</div>`;
  if (thinking !== null) {
    const chosen = openThinking.get(thinkingKey(index));
    const isOpen = chosen ?? (streaming && open);
    const label = streaming && open ? `<span class="shimmer">${escapeHtml(t("thinkingLive"))}</span>` : escapeHtml(t("reasoning"));
    html += `<details class="think"${isOpen ? " open" : ""}><summary>${icon("sparkles", 13)}${label}</summary><div class="think-body">${escapeHtml(thinking.trim())}</div></details>`;
  }
  if (turn.tools?.length) html += `<div class="tools-block">${toolRows(turn.tools)}</div>`;
  if (answer.trim()) html += `<div class="answer">${codeCards(renderMarkdown(answer, turn.sources ?? []))}</div>`;
  if (streaming && !answer.trim() && thinking === null) {
    const text = (last && statusText(session.status)) || (session.phase === "prefill" ? t("prefill") : t("thinkingLive"));
    html += `<div class="placeholder shimmer">${escapeHtml(text)}</div>`;
  }
  if (turn.schema === "valid") html += `<div class="badge-line"><span class="badge ready">${icon("check", 12)}${escapeHtml(t("validatedJson"))}</span></div>`;
  if (turn.sources?.length) {
    html += `<div class="sources"><span class="caps quiet">${escapeHtml(t("sources"))}</span>${turn.sources.map((source, i) => `<button class="source-chip" data-source="${escapeHtml(source.id)}" title="${escapeHtml(source.label)}"><b>${i + 1}</b><span>${escapeHtml(source.label)}</span></button>`).join("")}</div>`;
  }
  if (turn.failed) html += `<div class="error-line">${escapeHtml(turn.failed)}</div>`;
  if (turn.cancelled) html += `<div class="placeholder">■</div>`;
  if (turn.stats) html += `<div class="stat-chips">${statChips(turn.stats)}</div>`;
  if (turn.complete && turn.content) {
    html += `<div class="actions">
      <button class="mini" data-action="copy" title="${escapeHtml(t("copy"))}">${icon("copy", 14)}</button>
      ${last ? `<button class="mini" data-action="regenerate" title="${escapeHtml(t("regenerate"))}">${icon("retry", 14)}</button>` : ""}
      ${speechAvailable() ? `<button class="mini${speaking === index ? " on" : ""}" data-action="speak" title="${escapeHtml(t(speaking === index ? "stopReading" : "readAloud"))}">${icon("speaker", 14)}</button>` : ""}
      ${conversationMode === "agent" ? "" : `<button class="mini" data-action="branch" title="${escapeHtml(t("branchFromHere"))}">${icon("branch", 14)}</button>`}
      </div>`;
  }
  return `<div class="turn assistant" data-index="${index}">${html}</div>`;
}

function thinkingKey(index) {
  return `${currentId ?? "new"}:${index}`;
}

function statChips(stats) {
  const text = formatStats(stats);
  return text ? text.split(" · ").map((part) => `<span class="chip">${escapeHtml(part)}</span>`).join("") : "";
}

function renderMessages() {
  const box = $("messages");
  const scroller = $("scroll");
  const pinned = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 60;
  const turns = session.turns;
  const lastAssistant = turns.findLastIndex((x) => x.role === "assistant");
  while (box.children.length > turns.length) box.lastElementChild.remove();
  turns.forEach((turn, index) => {
    const key = `${index}|${index === lastAssistant}|${session.phase}|${session.status}|${language}|${speaking === index}|${conversationMode}`;
    const existing = box.children[index];
    if (existing && renderedTurns.get(existing)?.turn === turn && renderedTurns.get(existing)?.key === key) return;
    const holder = document.createElement("div");
    holder.innerHTML = turnHtml(turn, index, index === lastAssistant);
    const element = holder.firstElementChild;
    renderedTurns.set(element, { turn, key });
    if (existing) existing.replaceWith(element); else box.append(element);
  });
  if (pinned) scroller.scrollTop = scroller.scrollHeight;
}

function scrollToBottom() {
  requestAnimationFrame(() => { $("scroll").scrollTop = $("scroll").scrollHeight; });
}

async function onMessageAction(event) {
  const link = event.target.closest(".link[data-url]");
  if (link) {
    event.preventDefault();
    invoke("open_url", { url: link.dataset.url }).catch(fail);
    return;
  }
  const cited = event.target.closest("[data-source]");
  if (cited) {
    const turn = session.turns[Number(cited.closest(".turn")?.dataset.index)];
    const source = turn?.sources?.find((x) => x.id === cited.dataset.source);
    if (source) openSheet({ title: source.label, text: source.text, note: t("excerptNote") });
    return;
  }
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const element = button.closest(".turn");
  const index = Number(element?.dataset.index);
  const turn = session.turns[index];
  const action = button.dataset.action;
  if (action === "tool-more") {
    const output = button.closest(".tool-row")?.querySelector(".tool-output");
    if (output) {
      output.classList.toggle("clamp");
      button.textContent = t(output.classList.contains("clamp") ? "showFullOutput" : "showLess");
    }
  } else if (action === "tool-copy" && turn) {
    await copy(turn.tools?.[Number(button.dataset.tool)]?.result ?? "", button);
  } else if (action === "preview-image" && turn) {
    const a = turn.attachments?.[Number(button.dataset.attachment)];
    const image = turn.images?.find((x) => x.name === a?.name);
    if (image) openSheet({ title: a.name, image });
  } else if (action === "copy-code") {
    const code = button.closest(".code-card")?.querySelector("code")?.textContent ?? "";
    await copy(code, button);
  } else if (action === "copy" && turn) {
    await copy(turn.role === "assistant" ? splitThinking(turn.content).answer.trim() : turn.content, button);
  } else if (action === "edit" && turn) {
    try {
      const { state, prompt } = rewind(session, index);
      setSession(state);
      $("prompt").value = prompt;
      growPrompt();
      $("prompt").focus();
    } catch (e) { fail(e); }
  } else if (action === "branch" && turn) {
    branchFrom(index);
  } else if (action === "speak" && turn) {
    toggleSpeech(index, splitThinking(turn.content).answer.trim());
  } else if (action === "regenerate" && turn) {
    if (session.engine !== "ready" || session.active) { toast(t("noEngine")); return; }
    const userIndex = session.turns.slice(0, index).findLastIndex((x) => x.role === "user");
    if (userIndex < 0) return;
    const previous = { user: session.turns[userIndex], sources: turn.sources ?? [] };
    const before = session;
    try {
      const { state, prompt } = rewind(session, userIndex);
      session = state;
      if (!(await sendPrompt(prompt, previous))) { session = before; render(); }
    } catch (e) { fail(e); }
  }
}

// Mac "Read aloud" (AVSpeechSynthesizer) → the WebView's speechSynthesis, which uses the Windows voices.
let speaking = null;
function speechAvailable() {
  return typeof window.speechSynthesis !== "undefined" && typeof window.SpeechSynthesisUtterance === "function";
}

function toggleSpeech(index, text) {
  if (!speechAvailable()) return;
  window.speechSynthesis.cancel();
  if (speaking === index || !text) { speaking = null; renderMessages(); return; }
  const utterance = new SpeechSynthesisUtterance(text.replace(/```[\s\S]*?```/g, " ").replace(/[*_#>`]/g, ""));
  utterance.lang = language === "fr" ? "fr-FR" : "en-US";
  utterance.onend = utterance.onerror = () => { if (speaking === index) { speaking = null; renderMessages(); } };
  speaking = index;
  window.speechSynthesis.speak(utterance);
  renderMessages();
}

function stopSpeech() {
  if (speaking === null) return;
  speaking = null;
  if (speechAvailable()) window.speechSynthesis.cancel();
}

function branchFrom(index) {
  if (session.active) { toast(t("stop")); return; }
  let branch;
  try { branch = branchOf(session.turns, index); } catch (e) { fail(e); return; }
  stopSpeech();
  const source = conversations.find((c) => c.id === currentId);
  const base = source?.title ?? titleFrom(session.turns, t("newChat"));
  currentId = null;
  if (branch.turns.length) {
    currentId = newId();
    conversations = upsertConversation(conversations, {
      id: currentId, title: `${base} · ${t("branch")}`, updated: Date.now(), pinned: false, turns: branch.turns,
      projectId: source?.projectId ?? selectedProject, agent: false,
    });
    persistConversations();
  }
  session = { ...session, turns: branch.turns, lastStats: null, error: null };
  showPage("chat");
  render();
  $("prompt").value = branch.draft;
  growPrompt();
  $("prompt").focus();
}

async function copy(text, button, original = "copy") {
  try {
    await navigator.clipboard.writeText(text);
    setIcon(button, "check");
    setTimeout(() => setIcon(button, original), 1200);
  } catch (e) { fail(e); }
}

// ---------- chat chrome ----------

function engineState() {
  if (session.active) return "generating";
  return { idle: "idle", loading: "loading", ready: "ready", failed: "failed", stopped: "failed" }[session.engine] || "idle";
}

function loadedModel() {
  return session.engine !== "idle" ? models.find((m) => m.id === loadedId) : null;
}

function render() {
  const empty = session.turns.length === 0;
  const conversation = $("conversation");
  conversation.classList.toggle("empty", empty);
  const composer = $("composer");
  const slot = empty ? $("welcome-slot") : $("dock");
  if (composer.parentElement !== slot) slot.append(composer);
  if (empty) $("greeting").textContent = greeting(new Date().getHours(), greetingSeed, language);
  renderMessages();

  const current = conversations.find((c) => c.id === currentId);
  const modeTitle = { server: "Server", lab: "Lab" }[mode] ?? (conversationMode === "agent" ? "Agent" : t("chat"));
  $("chat-title").textContent = empty || mode === "server" || mode === "lab" ? modeTitle : current?.title ?? titleFrom(session.turns, t("newChat"));
  const model = loadedModel();
  const dot = $("status-dot");
  dot.className = `dot ${engineState()}`;
  $("model-name").textContent = model ? model.name : t("chooseModel");
  $("eject").hidden = session.engine === "idle";
  $("welcome-models-label").textContent = models.length ? t("manageModels") : t("addAModel");
  $("welcome-meta").textContent = model ? [modelSubtitle(model), formatBytes(model.size_bytes)].filter(Boolean).join(" · ") : "";

  const note = {
    idle: system && !system.runtimes?.[backend()] ? t("installFirst") : t("noEngine"),
    loading: `${t("loading")}…`,
    ready: "",
    failed: session.error || t("engineStopped"),
    stopped: session.error || t("engineStopped"),
  }[session.engine];
  $("engine-note").textContent = importing ? t("readingFiles") : session.active && session.phase === "prefill" ? t("prefill") : note;
  $("engine-note").title = $("engine-note").textContent;
  $("send").hidden = Boolean(session.active);
  $("stop").hidden = !session.active;
  const blocked = sendBlocked();
  $("send").disabled = Boolean(blocked);
  $("send").title = blocked ?? t("send");
  $("new-chat").disabled = Boolean(session.active);
  renderModes();
  renderAttachments();
  renderMcpCard();
  renderAgent();
  renderServer();
  renderWorkspacePage();
  renderMcpPage();
  renderLab();
  $("stats").innerHTML = session.lastStats ? statChips(session.lastStats) : `<span class="quiet small">${escapeHtml(t("noStats"))}</span>`;
  renderChats();
  renderModelMenu();
  renderLibrary();
  renderDownloads();
  updateIdleUnload();
}

function renderModelMenu() {
  const list = $("model-list");
  const info = session.info;
  let html = models.map((m) => {
    const current = m.id === loadedId && session.engine !== "idle";
    return `<button class="menu-item${current ? " current" : ""}" data-load="${escapeHtml(m.id)}">${current ? icon("check", 14) : icon("hex", 14)}<span>${escapeHtml(m.name)}</span><span class="sub">${escapeHtml(Number.isFinite(m.bits) ? `${m.bits.toFixed(2)} bpw` : "")}</span></button>`;
  }).join("");
  if (info && session.engine === "ready") {
    html += `<div class="menu-sep"></div><div class="menu-item quiet small">${escapeHtml([info.device, info.backend, info.context_limit ? `ctx ${info.context_limit}` : ""].filter(Boolean).join(" · "))}</div>`;
  }
  html += `<div class="menu-sep"></div><button class="menu-item" data-menu="add">${icon("download", 14)}<span>${escapeHtml(t("addModel"))}</span></button>`;
  if (session.engine !== "idle") html += `<button class="menu-item" data-menu="eject">${icon("eject", 14)}<span>${escapeHtml(t("eject"))}</span></button>`;
  list.innerHTML = html;
}

function growPrompt() {
  const el = $("prompt");
  el.style.height = "auto";
  el.style.height = `${Math.min(240, el.scrollHeight)}px`;
}

function setInspector(open) {
  inspectorOpen = open;
  const overlay = window.innerWidth < 1280;
  $("inspector").hidden = !open;
  $("inspector").classList.toggle("overlay", overlay);
  $("scrim").hidden = !(open && overlay);
  $("toggle-inspector").classList.toggle("on", open);
}

function closeMenus(except) {
  for (const id of MENUS) if (id !== except) $(id).hidden = true;
}

// ---------- pages ----------

function showPage(page) {
  $("chat-view").hidden = page !== "chat";
  $("models-page").hidden = page !== "models";
  $("settings-page").hidden = page !== "settings";
  $("workspace-page").hidden = page !== "workspace";
  $("mcp-page").hidden = page !== "mcp";
  closeMenus();
  if (page === "workspace") { renderWorkspacePage(); if (wsTab === "documents") refreshDocs(); }
  if (page === "mcp") { if (mcpEditing === null && mcpServers().length) editServer(mcpServers()[0].name); renderMcpPage(); refreshMcp(); }
  if (page === "models") {
    renderLibrary();
    refreshPending();
    if (modelsTab === "hub" && !hub.results.length && !hub.searching) searchHub();
  }
}

function showModelsTab(tab) {
  modelsTab = tab;
  $("nav-library").classList.toggle("selected", tab === "library");
  $("nav-hub").classList.toggle("selected", tab === "hub");
  $("library-view").hidden = tab !== "library";
  $("hub-view").hidden = tab !== "hub" || Boolean(hub.detail);
  $("detail-view").hidden = tab !== "hub" || !hub.detail;
  if (tab === "hub" && !hub.results.length && !hub.searching) searchHub();
}

function showSettingsPane(name) {
  for (const button of document.querySelectorAll("[data-settings]")) button.classList.toggle("selected", button.dataset.settings === name);
  for (const pane of document.querySelectorAll("[data-pane]")) pane.hidden = pane.dataset.pane !== name;
}

// ---------- library ----------

function renderLibrary() {
  const total = models.reduce((sum, m) => sum + (m.size_bytes || 0), 0);
  $("storage-size").textContent = formatBytes(total) || "0 B";
  $("storage-count").textContent = t("registered").replace("{n}", String(models.length));
  if (!models.length) {
    $("library-list").innerHTML = `<p class="quiet empty">${escapeHtml(t("emptyLibrary"))}</p>`;
    return;
  }
  $("library-list").innerHTML = models.map((m) => {
    const loaded = m.id === loadedId && session.engine !== "idle";
    const disabled = busy || Boolean(session.active) ? " disabled" : "";
    return `<div class="model-card${loaded ? " loaded" : ""}">
      <div class="card-top"><span class="hexagon">${icon("hex", 20)}</span>
        <div><div class="card-name">${escapeHtml(m.name)}</div><div class="card-sub">${escapeHtml(modelSubtitle(m))}</div></div>
        <span class="card-size">${escapeHtml(formatBytes(m.size_bytes))}</span></div>
      <div class="card-actions">
        ${loaded ? `<span class="loaded-label">${icon("check", 14)}${escapeHtml(t("loaded"))}</span>` : `<button class="load" data-load="${escapeHtml(m.id)}"${disabled}>${escapeHtml(t("load"))}</button>`}
        <span class="spacer"></span>
        <button class="round-icon" data-reveal="${escapeHtml(m.id)}" title="${escapeHtml(t("locate"))}">${icon("folder", 15)}</button>
        <button class="round-icon" data-remove="${escapeHtml(m.id)}" title="${escapeHtml(t("remove"))}"${loaded ? " disabled" : ""}>${icon("trash", 15)}</button>
      </div></div>`;
  }).join("");
}

async function refreshModels() {
  models = await invoke("list_models");
  renderStartupModels();
  render();
}

async function importFolder() {
  try {
    const path = await tauri.dialog.open({ directory: true, multiple: false });
    if (!path) return;
    await invoke("add_model", { path });
    await refreshModels();
  } catch (e) { fail(e); }
}

async function loadModel(id, { quiet = false } = {}) {
  if (session.active) { toast(t("stop")); return; }
  if (system && !system.runtimes?.[backend()]) {
    if (quiet) return;
    toast(t("installFirst"));
    showPage("settings");
    showSettingsPane("engine");
    return;
  }
  lastModel = id;
  savePrefs();
  loadedId = id;
  if (!quiet) showPage("chat");
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

async function removeModel(id) {
  if (!window.confirm(t("confirmRemove"))) return;
  try { await invoke("delete_model", { id }); await refreshModels(); } catch (e) { fail(e); }
}

// ---------- Hugging Face ----------

let searchTimer = null;
function scheduleSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { hub.limit = HUB_PAGE; searchHub(); }, 350);
}

async function searchHub() {
  const seq = ++hub.seq;
  hub.query = $("hub-query").value.trim();
  hub.searching = true;
  hub.error = null;
  renderHub();
  try {
    const results = await invoke("hub_search", { query: hub.query, limit: hub.limit });
    if (seq !== hub.seq) return;
    hub.results = results;
  } catch (e) {
    if (seq !== hub.seq) return;
    hub.error = message(e);
  }
  hub.searching = false;
  renderHub();
}

function renderHub() {
  $("hub-spinner").hidden = !hub.searching;
  $("hub-error").hidden = !hub.error;
  $("hub-error").textContent = hub.error || "";
  const showRecommended = !hub.query;
  $("recommended").hidden = !showRecommended;
  if (showRecommended && !$("recommended").childElementCount) {
    $("recommended").innerHTML = `<div class="caps quiet">${escapeHtml(t("recommended"))}</div>` + RECOMMENDED.map((family) => `
      <div class="family-name">${escapeHtml(family.family)}</div>
      <div class="rec-grid">${family.models.map((m) => `
        <button class="rec-card" data-rec="${escapeHtml(m.id)}" title="${escapeHtml(m.id)}">
          <div class="rec-top">${icon(m.icon, 21)}<span class="tag">EXL3</span></div>
          <div class="rec-title">${escapeHtml(m.title)}</div>
          <div class="rec-params">${escapeHtml(m.parameters)}</div>
          <div class="rec-quant">${escapeHtml(m.quantization)}</div>
          <div class="rec-view"><span>${escapeHtml(t("viewModel"))}</span>${icon("arrowUpRight", 13)}</div>
        </button>`).join("")}</div>`).join("");
  }
  $("hub-results").innerHTML = hub.results.map((model) => {
    const { owner, name } = splitRepo(model.id);
    const gated = model.gated && model.gated !== false ? icon("lock", 13) : "";
    return `<button class="hub-row" data-repo="${escapeHtml(model.id)}">
      <div class="names"><div class="name">${escapeHtml(name)}</div><div class="owner">${escapeHtml(owner)}</div></div>
      ${gated}<span class="dl">${icon("arrowDown", 12)}${escapeHtml(formatCount(model.downloads))}</span>${icon("chevronRight", 13)}</button>`;
  }).join("");
  $("hub-empty").hidden = hub.searching || Boolean(hub.error) || hub.results.length > 0;
  $("hub-more").hidden = hub.results.length < hub.limit || hub.limit >= HUB_MAX;
  $("hub-more").disabled = hub.searching;
}

function recommendedEntry(repo) {
  for (const family of RECOMMENDED) {
    const model = family.models.find((m) => m.id === repo);
    if (model) return model;
  }
  return null;
}

async function openDetail(repo, revision = null, preferredVariant = undefined) {
  hub.loading = true;
  $("detail-loading").hidden = false;
  if (!hub.detail) {
    hub.detail = { id: repo, revision: revision ?? "main", branches: [], variants: [], readme: "", commit: "" };
    $("detail-title").textContent = repo;
    $("detail-readme").innerHTML = "";
    $("detail-notes").hidden = true;
    $("detail-size").textContent = "—";
    $("detail-download").disabled = true;
  }
  showModelsTab("hub");
  try {
    const detail = await invoke("hub_details", { repo, revision });
    hub.detail = detail;
    hub.variantId = pickVariant(detail.variants, preferredVariant)?.id ?? null;
  } catch (e) {
    hub.detail = null;
    hub.error = message(e);
    showModelsTab("hub");
    renderHub();
  }
  hub.loading = false;
  $("detail-loading").hidden = true;
  if (hub.detail) renderDetail();
}

function renderDetail() {
  const detail = hub.detail;
  $("detail-title").textContent = detail.id;
  const revisions = [detail.revision, ...detail.branches.filter((b) => b !== detail.revision)];
  $("detail-revision").innerHTML = revisions.map((r) => `<option value="${escapeHtml(r)}">${escapeHtml(revisionLabel(r, t))}</option>`).join("");
  $("detail-revision").value = detail.revision;
  const choices = recommendedChoices(detail.id);
  $("detail-quant-row").hidden = choices.length < 2;
  $("detail-quant").innerHTML = choices.map((c) => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.label)}</option>`).join("");
  if (choices.length > 1) $("detail-quant").value = detail.id;
  $("detail-variant-row").hidden = detail.variants.length < 2;
  $("detail-variant").innerHTML = detail.variants.map((v) => `<option value="${escapeHtml(v.id)}">${escapeHtml(v.label)}</option>`).join("");
  if (hub.variantId !== null) $("detail-variant").value = hub.variantId;
  const variant = detail.variants.find((v) => v.id === hub.variantId);
  $("detail-size").textContent = variant ? formatBytes(variant.size_bytes) : "—";
  $("detail-download").disabled = !variant || Boolean(download) || hub.loading;
  const notes = [];
  if (/flash-next/i.test(detail.id)) notes.push(escapeHtml(t("flashNote")));
  if (!detail.variants.length) notes.push(`<span class="warn">${escapeHtml(t("noVariant"))}</span>`);
  if (detail.gated) notes.push(`<span class="warn">${escapeHtml(t("gated"))}</span>`);
  $("detail-notes").innerHTML = notes.join("");
  $("detail-notes").hidden = !notes.length;
  $("detail-readme").innerHTML = codeCards(renderMarkdown(stripFrontMatter(detail.readme) || t("noReadme")));
}

async function startDownload() {
  const detail = hub.detail;
  const variant = detail?.variants.find((v) => v.id === hub.variantId);
  if (!variant || download) return;
  $("detail-download").disabled = true;
  try {
    const request = await invoke("hub_download", { repo: detail.id, commit: detail.commit, folder: variant.id });
    download = { id: request.id, name: request.name, repo: request.repo, done: request.retained_bytes || 0, total: request.size_bytes };
    renderDownloads();
  } catch (e) { fail(e); }
  if (hub.detail) renderDetail();
}

async function refreshPending() {
  try { pending = await invoke("hub_pending"); } catch (e) { log(`error: ${message(e)}`); pending = []; }
  renderDownloads();
}

function renderDownloads() {
  const cards = [];
  if (download) {
    const pct = percent(download.done, download.total);
    cards.push(`<div class="download-card"><div class="name">${escapeHtml(download.repo)}</div>
      <div class="bar"><i data-pct="${pct}"></i></div>
      <div class="line"><span>${escapeHtml(`${t("downloading")} · ${formatBytes(download.done) || "0 B"} / ${formatBytes(download.total)} · ${pct}%`)}</span><span class="spacer"></span>
      <button class="mini" data-pause title="${escapeHtml(t("pause"))}">${icon("pause", 13)}</button></div></div>`);
  }
  for (const job of pending.filter((p) => p.id !== download?.id)) {
    cards.push(`<div class="download-card"><div class="name">${escapeHtml(job.repo)}</div>
      <div class="line"><span>${escapeHtml(`${formatBytes(job.retained_bytes) || "0 B"} · ${t("interrupted")}`)}</span><span class="spacer"></span>
      <button class="mini" data-resume="${escapeHtml(job.id)}" title="${escapeHtml(t("resume"))}"${download ? " disabled" : ""}>${icon("play", 13)}</button>
      <button class="mini" data-discard="${escapeHtml(job.id)}" title="${escapeHtml(t("discardPartial"))}"${download ? " disabled" : ""}>${icon("trash", 13)}</button></div></div>`);
  }
  const html = cards.length ? `<div class="caps quiet">${escapeHtml(t("downloadsTitle"))}</div>${cards.join("")}` : "";
  const side = $("side-downloads");
  side.innerHTML = html;
  for (const bar of side.querySelectorAll("[data-pct]")) bar.style.width = `${bar.dataset.pct}%`;
  const indicator = $("downloads-indicator");
  indicator.hidden = !cards.length;
  indicator.title = download ? `${t("downloading")} ${percent(download.done, download.total)}%` : t("downloadsIndicator");
}

async function onDownloadsClick(event) {
  const target = event.target.closest("button");
  if (!target) return;
  try {
    if (target.hasAttribute("data-pause")) await invoke("hub_pause");
    else if (target.dataset.resume) {
      const request = await invoke("hub_resume", { id: target.dataset.resume });
      download = { id: request.id, name: request.name, repo: request.repo, done: request.retained_bytes || 0, total: request.size_bytes };
      renderDownloads();
    } else if (target.dataset.discard) {
      if (!window.confirm(t("confirmDiscard"))) return;
      await invoke("hub_discard", { id: target.dataset.discard });
      await refreshPending();
    }
  } catch (e) { fail(e); }
}

// ---------- engine runtime ----------

function renderRuntime() {
  if (!system) return;
  const key = backend();
  const marker = system.runtimes[key];
  $("distro-row").hidden = key !== "rocm_wsl";
  $("amd-help").hidden = key !== "rocm_wsl";
  $("runtime-status").textContent = marker
    ? `${t("runtimeReady")} · ${marker.probe?.devices?.[0]?.name ?? ""} · ${marker.id}`
    : t("runtimeMissing");
  $("install-runtime").textContent = marker ? t("reinstall") : t("install");
  $("install-runtime").disabled = busy || (key === "rocm_wsl" && !$("distro").value);
}

async function refreshSystem() {
  system = await invoke("system_info");
  $("app-version").textContent = system.app_version;
  const gpus = [
    ...system.nvidia.map((g) => `${g.name} (${Math.round(g.memory_mib / 1024)} GB, driver ${g.driver}${g.driver_ok ? "" : " ⚠"})`),
    ...system.amd,
  ];
  $("gpu-summary").textContent = `${gpus.join(" · ") || t("noGpu")} — ${system.recommendation.reason}`;
  const distro = $("distro");
  const previous = distro.value || prefs.distro;
  distro.innerHTML = "";
  for (const name of system.wsl_distros) distro.append(new Option(name, name));
  if (!system.wsl_distros.length) distro.append(new Option(t("noDistro"), ""));
  if (previous && system.wsl_distros.includes(previous)) distro.value = previous;
  renderRuntime();
  render();
}

async function installRuntime() {
  busy = true;
  $("runtime-progress").hidden = false;
  renderRuntime();
  try {
    await invoke("install_runtime", { backend: backend(), distro: backend() === "rocm_wsl" ? $("distro").value : null });
    await refreshSystem();
  } catch (e) {
    fail(e);
    $("runtime-status").textContent = message(e);
    document.querySelector("details.log").open = true;
  } finally {
    busy = false;
    $("runtime-progress").hidden = true;
    renderRuntime();
    renderLibrary();
  }
}

// ---------- updates ----------

function renderUpdate() {
  const status = updateStatus;
  const badge = updateBadge(status, t);
  const button = $("update-badge");
  button.hidden = !badge;
  if (badge) {
    button.className = `update-badge ${badge.kind}`;
    button.title = badge.label;
    button.setAttribute("aria-label", badge.label);
  }
  if (!status) return;
  if (status.channel && document.activeElement !== $("update-channel")) $("update-channel").value = status.channel;
  const version = status.update?.version ?? "";
  const text = {
    idle: t("updateIdle"),
    checking: t("checking"),
    current: t("upToDate"),
    downloading: `${t("updateDownloading").replace("{v}", version)} · ${percent(status.done, status.total)}%`,
    ready: t("updateReady").replace("{v}", version),
    failed: `${t("updateFailed")}${status.error ? `: ${status.error}` : ""}`,
  }[status.phase] ?? "";
  $("update-text").textContent = text;
  $("update-install").hidden = status.phase !== "ready";
  $("update-check").disabled = status.phase === "checking" || status.phase === "downloading";
  const bar = $("update-progress");
  bar.hidden = status.phase !== "downloading";
  bar.value = percent(status.done, status.total);
}

async function installUpdate() {
  try { await invoke("update_install"); } catch (e) { fail(e); }
}

// ---------- sending ----------

const base64Bytes = (data) => Math.floor((data.length * 3) / 4) - (data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0);

/** Mac send gate: the reason sending is refused, or null. */
function sendBlocked() {
  if (mode !== "chat" && mode !== "agent") return t("chooseChatOrAgent");
  if (session.engine !== "ready") return t("loadModelReady");
  if (serverStatus?.running) return t("serverShares");
  if (importing) return t("importingFiles");
  if (mcpBusy && conversationMode === "chat" && mcpEnabled) return t("connectingTools");
  if (searchingDocs) return t("preparingProject");
  const pendingImages = attachments.filter((a) => a.kind === "image");
  if (pendingImages.length) {
    if (!session.info?.vision) return t("loadVisionModel");
    const kept = session.turns.flatMap((turn) => (turn.images ?? []).filter((x) => typeof x.data === "string"));
    const count = kept.length + pendingImages.length;
    const bytes = kept.reduce((sum, x) => sum + base64Bytes(x.data), 0) + pendingImages.reduce((sum, a) => sum + a.file_bytes, 0);
    if (count > IMAGE_LIMIT.count || bytes > IMAGE_LIMIT.bytes) return t("imageBudget");
  }
  return null;
}

/** Validated JSON (Chat only): the schema text, "" for free text, or null when it is not a JSON object. */
function responseSchema() {
  const text = $("schema").value;
  if (conversationMode === "agent" || !text.trim()) return "";
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? text : null;
  } catch { return null; }
}

/**
 * One turn. `previous` (regenerate) reuses the earlier user turn's files, images and excerpts
 * instead of the composer's.
 */
async function sendPrompt(prompt, previous = null) {
  if (session.active) return false;
  const files = previous ? [] : attachments;
  const text = prompt.trim();
  if (!text && !files.length && !previous) return false;
  const blocked = sendBlocked();
  if (blocked) { toast(blocked); return false; }
  const schema = responseSchema();
  if (schema === null) { toast(t("schemaInvalid")); return false; }
  savePrefs();
  const agent = conversationMode === "agent";
  const project = currentProject();
  let sources = previous?.sources ?? [];
  if (project && !previous) {
    searchingDocs = true;
    render();
    try {
      sources = await invoke("documents_search", { projectId: project.id, query: text || EMPTY_PROMPT, limit: MAX_EXCERPTS });
    } catch (e) { log(`documents: ${message(e)}`); sources = []; }
    searchingDocs = false;
    render();
  }
  let agentTask = null;
  let agentSystem = "";
  if (agent) {
    if (!currentId) currentId = newId();
    try {
      if (agentInfo?.task !== currentId) agentInfo = await invoke("agent_open", { task: currentId });
    } catch (e) { fail(e); return false; }
    agentTask = currentId;
    agentSystem = agentInfo.systemPrompt ?? "";
  }
  if (session.active || session.engine !== "ready") return false;
  const requestId = `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  const extra = previous
    ? { context: previous.user.context ?? previous.user.content, attachments: previous.user.attachments ?? [], images: previous.user.images ?? [], sources }
    : {
      context: attachmentContext(text || EMPTY_PROMPT, files) + documentContext(sources),
      attachments: files.map((f) => ({ name: f.file_name, kind: f.kind })),
      images: files.filter((f) => f.kind === "image").map((f) => ({ name: f.file_name, mime: f.info.mime, data: f.data })),
      sources,
    };
  let next;
  try { next = startTurn(session, requestId, previous ? previous.user.content : text, extra); } catch (e) { fail(e); return false; }
  const instructions = [composeSystem({ base: $("system").value, project, profile: null, hasDocuments: sources.length > 0 }), agentSystem]
    .filter((part) => part.trim()).join("\n\n");
  const messages = engineMessages(instructions, next.turns.slice(0, -1));
  if (!previous) attachments = [];
  setSession(next);
  scrollToBottom();
  const params = sampling();
  if (agent) params.max_tokens = AGENT_MAX_TOKENS;
  const options = {
    schema: schema || null,
    mcp: !agent && mcpEnabled && !schema,
    allowedTools: agent ? null : project ? project.allowedTools : null,
    agentTask,
  };
  try {
    await invoke("generate", { requestId, messages, sampling: params, options });
  } catch (e) {
    setSession(abortTurn(session, requestId, message(e)));
    return false;
  }
  if (agent) startAgentWatch(requestId);
  return true;
}

async function send(event) {
  event.preventDefault();
  const prompt = $("prompt").value;
  if ((!prompt.trim() && !attachments.length) || session.active) return;
  const blocked = sendBlocked();
  if (blocked) { toast(blocked); return; }
  $("prompt").value = "";
  growPrompt();
  if (!(await sendPrompt(prompt))) { $("prompt").value = prompt; growPrompt(); }
}

// ---------- sheet (excerpts, file previews, Agent review) ----------

let sheetApply = null;
const validImage = (mime, data) => /^image\/(png|jpeg|webp)$/.test(mime ?? "") && typeof data === "string" && /^[A-Za-z0-9+/=]+$/.test(data);

function openSheet({ title, text = null, note = null, image = null, html = null, apply = null }) {
  $("sheet-title").textContent = title ?? "";
  const body = $("sheet-body");
  if (html !== null) body.innerHTML = html;
  else if (image) body.innerHTML = validImage(image.mime, image.data) ? `<img class="sheet-image" alt="" src="data:${image.mime};base64,${image.data}">` : "";
  else {
    body.innerHTML = '<pre class="sheet-text"></pre>';
    body.firstElementChild.textContent = text ?? "";
  }
  $("sheet-note").textContent = note ?? "";
  $("sheet-note").hidden = !note;
  sheetApply = apply;
  $("sheet-apply").hidden = !apply;
  $("sheet").hidden = false;
}

function closeSheet() {
  $("sheet").hidden = true;
  $("sheet-body").innerHTML = "";
  sheetApply = null;
}

// ---------- modes and projects (Mac sidebar) ----------

/** Opens the most recent conversation of the current mode and project, or a new one. */
function openFirstOrNew() {
  const { pinned, recent } = sections(visibleConversations());
  const first = [...pinned, ...recent][0];
  stopSpeech();
  if (first) {
    currentId = first.id;
    session = { ...session, turns: first.turns, lastStats: null, error: null };
  } else {
    session = newConversation(session);
    currentId = null;
  }
  agentInfo = null;
  if (conversationMode === "agent") refreshAgent();
}

async function selectMode(next) {
  if (next === mode) return;
  if (session.active) { toast(t("stop")); return; }
  if (mode === "server" && serverStatus?.running) await stopServer();
  mode = next;
  if (next === "chat" || next === "agent") {
    if (conversationMode !== next) {
      conversationMode = next;
      openFirstOrNew();
    }
    savePrefs();
  } else if (next === "server") {
    refreshServer();
  }
  showPage("chat");
  render();
  if (next === "lab") renderLab();
}

function renderModes() {
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.classList.toggle("selected", button.dataset.mode === mode);
    button.disabled = Boolean(session.active) && button.dataset.mode !== mode;
  }
  const conversing = mode === "chat" || mode === "agent";
  $("conversation").hidden = !conversing;
  $("server-view").hidden = mode !== "server";
  $("lab-view").hidden = mode !== "lab";
  const agent = conversationMode === "agent";
  $("agent-bar").hidden = mode !== "agent";
  $("agent-card").hidden = !agent;
  $("json-card").hidden = agent;
  $("mcp-card").hidden = agent;
  $("tools-mcp-row").hidden = agent;
  $("tools-configure").hidden = agent;
  $("tools-mcp").checked = mcpEnabled;
  $("tools-mcp").disabled = mcpBusy;
  const newLabel = t(agent ? "newTask" : "newChat");
  $("new-chat-label").textContent = newLabel;
  $("new-chat").title = `${newLabel} · Ctrl+N`;
  $("prompt").placeholder = t(agent ? "taskPlaceholder" : "placeholder");
  const select = $("project-select");
  if (document.activeElement !== select) select.innerHTML = [["", t("noProject")], ...workspace.projects.map((p) => [p.id, p.name])]
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
  select.value = selectedProject ?? "";
  select.disabled = Boolean(session.active);
}

function chooseProject(id) {
  if (session.active) { toast(t("stop")); render(); return; }
  selectedProject = workspace.projects.some((p) => p.id === id) ? id : null;
  docResults = null;
  docEntries = [];
  savePrefs();
  openFirstOrNew();
  const project = currentProject();
  const profile = project?.profileId ? workspace.profiles.find((p) => p.id === project.profileId) : null;
  render();
  if (profile) applyProfile(profile);
}

// ---------- workspace page: projects, documents, profiles ----------

const optionsHtml = (items) => items.map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");

function renderWorkspacePage() {
  if ($("workspace-page").hidden) return;
  for (const button of document.querySelectorAll("[data-ws]")) button.classList.toggle("selected", button.dataset.ws === wsTab);
  for (const pane of document.querySelectorAll("[data-ws-pane]")) pane.hidden = pane.dataset.wsPane !== wsTab;
  const locked = Boolean(session.active);
  $("ws-busy").hidden = !locked;
  const project = currentProject();
  if (document.activeElement !== $("ws-project")) {
    $("ws-project").innerHTML = optionsHtml([["", t("noProject")], ...workspace.projects.map((p) => [p.id, p.name])]);
    $("ws-project").value = selectedProject ?? "";
  }
  $("ws-project").disabled = locked;
  $("ws-create").disabled = locked;
  $("ws-project-card").hidden = !project;
  if (project) {
    if (document.activeElement !== $("ws-name")) $("ws-name").value = project.name;
    if (document.activeElement !== $("ws-instructions")) $("ws-instructions").value = project.instructions;
    if (document.activeElement !== $("ws-profile")) {
      $("ws-profile").innerHTML = optionsHtml([["", t("currentSettings")], ...workspace.profiles.map((p) => [p.id, p.name])]);
      $("ws-profile").value = project.profileId ?? "";
    }
    $("ws-remove-project").disabled = locked;
  }
  $("ws-docs-none").hidden = Boolean(project);
  $("ws-docs").hidden = !project;
  $("ws-add-files").disabled = locked;
  $("ws-add-folder").disabled = locked;
  renderDocs();
  renderProfiles();
}

function updateProject(change) {
  const project = currentProject();
  if (!project) return;
  const next = normalizeProject({ ...project, ...change });
  if (!next) return;
  workspace = { ...workspace, projects: workspace.projects.map((p) => (p.id === project.id ? next : p)) };
  persistWorkspace();
  renderModes();
}

function createProject() {
  const name = $("ws-new-name").value.trim();
  if (!name || session.active) return;
  const project = normalizeProject({ id: newId(), name, instructions: "", profileId: null, allowedTools: [] });
  if (!project) return;
  workspace = { ...workspace, projects: [...workspace.projects, project] };
  persistWorkspace();
  $("ws-new-name").value = "";
  chooseProject(project.id);
  renderWorkspacePage();
}

async function removeProject() {
  const project = currentProject();
  if (!project || session.active || !window.confirm(t("confirmRemoveProject"))) return;
  try { await invoke("documents_remove_project", { projectId: project.id }); } catch (e) { fail(e); return; }
  workspace = { ...workspace, projects: workspace.projects.filter((p) => p.id !== project.id) };
  persistWorkspace();
  // Mac: the conversations stay, without a project.
  conversations = conversations.map((c) => (c.projectId === project.id ? { ...c, projectId: null } : c));
  persistConversations();
  chooseProject(null);
  renderWorkspacePage();
}

function profileName(profile) {
  const model = profile.model ? models.find((m) => m.id === profile.model)?.name ?? profile.model : t("currentModel");
  return [model, `T ${profile.temperature}`, `top-k ${profile.topK}`, profile.contextLength ? `ctx ${profile.contextLength}` : ""].filter(Boolean).join(" · ");
}

function renderProfiles() {
  const list = $("ws-profiles");
  $("ws-save-profile").disabled = Boolean(session.active);
  if (!workspace.profiles.length) {
    list.innerHTML = `<div class="empty-state"><div class="strong">${escapeHtml(t("noProfiles"))}</div><div class="quiet small">${escapeHtml(t("noProfilesDetail"))}</div></div>`;
    return;
  }
  const disabled = session.active ? " disabled" : "";
  list.innerHTML = workspace.profiles.map((p) => `<div class="row-field">
      <div class="grow"><div class="strong">${escapeHtml(p.name)}</div><div class="quiet small">${escapeHtml(profileName(p))}</div></div>
      <button class="glass-button" data-apply-profile="${escapeHtml(p.id)}"${disabled}>${escapeHtml(t("apply"))}</button>
      <button class="round-icon" data-delete-profile="${escapeHtml(p.id)}" title="${escapeHtml(t("deleteProfile"))}"${disabled}>${icon("trash", 15)}</button></div>`).join("");
}

function currentContext() {
  return Math.max(0, Math.trunc(Number($("context").value) || 0));
}

function saveProfile() {
  const name = $("ws-profile-name").value.trim();
  if (!name || session.active) return;
  const s = sampling();
  const profile = normalizeProfile({
    id: newId(), name, model: loadedModel()?.id ?? null, systemPrompt: $("system").value, temperature: s.temperature,
    topK: s.top_k, topP: s.top_p, repetitionPenalty: s.repetition_penalty, contextLength: currentContext(), responseSchema: $("schema").value,
  });
  if (!profile) { toast(t("invalidProfile")); return; }
  workspace = { ...workspace, profiles: [...workspace.profiles, profile] };
  persistWorkspace();
  $("ws-profile-name").value = "";
  renderWorkspacePage();
}

function deleteProfile(id) {
  workspace = {
    projects: workspace.projects.map((p) => (p.profileId === id ? { ...p, profileId: null } : p)),
    profiles: workspace.profiles.filter((p) => p.id !== id),
  };
  persistWorkspace();
  renderWorkspacePage();
}

/** Writes a profile into the inspector; reloads the model when it or the context differs. */
async function applyProfile(profile) {
  if (session.active) { toast(t("stop")); return; }
  const valid = profile.temperature >= 0 && profile.temperature <= 2 && profile.topK >= 0 && profile.topK <= 10000
    && profile.repetitionPenalty >= 0.1 && profile.repetitionPenalty <= 4;
  if (!valid) { toast(t("invalidProfile")); return; }
  $("temperature").value = String(profile.temperature);
  $("top-k").value = String(profile.topK);
  if (profile.topP !== null) $("top-p").value = String(profile.topP);
  $("rep").value = String(profile.repetitionPenalty);
  $("system").value = profile.systemPrompt;
  $("schema").value = profile.responseSchema;
  const contextChanged = profile.contextLength !== currentContext();
  $("context").value = String(profile.contextLength);
  renderSliders();
  savePrefs();
  const loaded = loadedModel()?.id ?? null;
  const target = profile.model && models.some((m) => m.id === profile.model) ? profile.model : loaded;
  if (target && (target !== loaded || (contextChanged && session.engine !== "idle"))) await loadModel(target, { quiet: true });
}

async function refreshDocs() {
  const project = currentProject();
  if (!project) { docEntries = []; renderDocs(); return; }
  try { docEntries = await invoke("documents_list", { projectId: project.id }); } catch (e) { fail(e); docEntries = []; }
  renderDocs();
}

function renderDocs() {
  $("ws-docs-count").textContent = docEntries.length ? String(docEntries.length) : "";
  const list = $("ws-doc-list");
  if (docResults) {
    list.innerHTML = docResults.length
      ? docResults.map((x, i) => `<button class="doc-row" data-result="${i}">${icon("doc", 15)}<div class="grow"><div class="strong">${escapeHtml(x.label)}</div><div class="quiet small clamp2">${escapeHtml(x.text.slice(0, 280))}</div></div></button>`).join("")
      : `<p class="quiet small padded">${escapeHtml(t("noPassage"))}</p>`;
    return;
  }
  const disabled = session.active ? " disabled" : "";
  list.innerHTML = docEntries.length
    ? docEntries.map((d) => `<div class="doc-row">${icon("doc", 15)}<div class="grow"><div class="strong">${escapeHtml(d.file_name)}</div><div class="quiet small">${escapeHtml(formatBytes(d.bytes))}</div></div>
        <button class="round-icon" data-remove-doc="${escapeHtml(d.id)}" title="${escapeHtml(t("removeFromLibrary"))}"${disabled}>${icon("trash", 15)}</button></div>`).join("")
    : `<p class="quiet small padded">${escapeHtml(t("noDocuments"))}</p>`;
}

async function addDocs(folder) {
  const project = currentProject();
  if (!project || session.active) return;
  $("ws-docs-spinner").hidden = false;
  try {
    const report = await invoke("documents_add", { projectId: project.id, folder });
    if (report?.skipped?.length) {
      const list = report.skipped.slice(0, 10).map(([name, reason]) => `${name} (${reason})`).join(", ");
      toast(t("skippedFiles").replace("{list}", list), 9000);
    }
  } catch (e) { fail(e); }
  $("ws-docs-spinner").hidden = true;
  await refreshDocs();
}

let docTimer = null;
let docSeq = 0;
function scheduleDocSearch() {
  clearTimeout(docTimer);
  docTimer = setTimeout(searchDocs, 300);
}

async function searchDocs() {
  const project = currentProject();
  const query = $("ws-doc-query").value.trim();
  const seq = ++docSeq;
  if (!project || !query) { docResults = null; renderDocs(); return; }
  try {
    const results = await invoke("documents_search", { projectId: project.id, query, limit: 8 });
    if (seq === docSeq) docResults = results;
  } catch (e) {
    if (seq === docSeq) docResults = [];
    fail(e);
  }
  renderDocs();
}

// ---------- composer attachments (Mac ChatAttachment) ----------

function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = attachments.map((a, i) => {
    const remove = `<button type="button" class="chip-x" data-remove-attachment="${i}" title="${escapeHtml(t("removeFile"))}">${icon("close", 11)}</button>`;
    if (a.kind === "image" && validImage(a.info?.mime, a.data)) {
      return `<span class="thumb"><button type="button" class="thumb-img" data-preview-attachment="${i}" title="${escapeHtml(t("previewImage"))}"><img alt="${escapeHtml(a.file_name)}" src="data:${a.info.mime};base64,${a.data}"></button>${remove}</span>`;
    }
    return `<span class="file-chip">${icon("doc", 13)}<button type="button" class="chip-name" data-preview-attachment="${i}" title="${escapeHtml(t("previewText"))}">${escapeHtml(a.file_name)}</button>${remove}</span>`;
  }).join("");
}

function addAttachments(results) {
  // Images need a vision model; without one loaded yet, the send gate explains it.
  const refuseImages = session.engine === "ready" && !session.info?.vision;
  const incoming = results.map((r) => (r.kind === "image" && refuseImages ? { kind: "error", file_name: r.file_name, error: "loadVisionModel" } : r));
  const { accepted, rejected } = admitAttachments(attachments, incoming);
  attachments = accepted;
  if (rejected.length) toast(rejected.map((r) => `${r.name}: ${t(r.reason)}`).join("\n"), 8000);
  render();
}

async function importAttachments(load) {
  if (importing) return;
  importing = true;
  render();
  try { addAttachments((await load()) ?? []); } catch (e) { fail(e); }
  importing = false;
  render();
}

function readImageFile(file) {
  const name = file.name || `image.${file.type.split("/")[1] ?? "png"}`;
  return new Promise((resolve) => {
    if (file.size > IMAGE_LIMIT.bytes) { resolve({ kind: "error", file_name: name, error: "attachImagesTooLarge" }); return; }
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      resolve({ kind: "image", file_name: name, path: null, info: { mime: file.type, width: 0, height: 0 }, data: url.slice(url.indexOf(",") + 1), file_bytes: file.size });
    };
    reader.onerror = () => resolve({ kind: "error", file_name: name, error: message(reader.error ?? "read failed") });
    reader.readAsDataURL(file);
  });
}

function onPaste(event) {
  const files = [...(event.clipboardData?.files ?? [])].filter((f) => /^image\/(png|jpeg|webp)$/.test(f.type));
  if (!files.length) return;
  event.preventDefault();
  importAttachments(() => Promise.all(files.slice(0, 8).map(readImageFile)));
}

function onAttachmentsClick(event) {
  const remove = event.target.closest("[data-remove-attachment]");
  const preview = event.target.closest("[data-preview-attachment]");
  if (remove) {
    attachments = attachments.filter((_, i) => i !== Number(remove.dataset.removeAttachment));
    render();
  } else if (preview) {
    const a = attachments[Number(preview.dataset.previewAttachment)];
    if (!a) return;
    if (a.kind === "image") openSheet({ title: a.file_name, image: { mime: a.info.mime, data: a.data } });
    else openSheet({ title: a.file_name, text: a.text });
  }
}

// ---------- MCP (Mac Settings → MCP servers) ----------

let mcpTransport = "https";
let mcpTestErrors = [];

function mcpServers() { return mcpStatus?.servers ?? []; }

function mcpErrors() {
  return [mcpStatus?.error, ...mcpServers().filter((s) => s.error).map((s) => `${s.name}: ${s.error}`)].filter(Boolean);
}

function mcpLine() {
  if (mcpBusy) return t("mcpConnecting");
  if (!mcpEnabled) return t("mcpDisabled");
  const connected = mcpServers().filter((s) => s.connected);
  return t("mcpConnected").replace("{s}", String(connected.length)).replace("{n}", String(connected.reduce((n, s) => n + s.tools.length, 0)));
}

function collectCatalog() {
  const names = mcpServers().flatMap((s) => s.tools);
  if (names.length) mcpCatalog = [...new Set([...mcpCatalog, ...names])].sort();
}

async function refreshMcp() {
  try { mcpStatus = await invoke("mcp_status"); collectCatalog(); } catch (e) { log(`mcp: ${message(e)}`); }
  render();
}

async function setMcp(enabled, refresh = false) {
  if (mcpBusy) return;
  mcpBusy = true;
  render();
  try {
    mcpStatus = await invoke("mcp_set_enabled", { enabled, refresh });
    mcpEnabled = enabled;
    savePrefs();
    collectCatalog();
  } catch (e) { fail(e); }
  mcpBusy = false;
  render();
}

async function testMcp() {
  if (mcpBusy) return;
  const was = mcpEnabled;
  mcpBusy = true;
  render();
  try {
    const tested = await invoke("mcp_set_enabled", { enabled: true, refresh: true });
    mcpStatus = tested;
    collectCatalog();
    mcpTestErrors = mcpErrors();
    // Testing does not turn MCP on for chats.
    mcpStatus = was ? tested : await invoke("mcp_set_enabled", { enabled: false, refresh: false });
  } catch (e) { fail(e); }
  mcpBusy = false;
  render();
}

function renderMcpCard() {
  const errors = mcpErrors();
  $("mcp-card-dot").className = `dot ${mcpBusy ? "loading" : !mcpEnabled ? "idle" : errors.length ? "failed" : "ready"}`;
  $("mcp-card-status").textContent = mcpLine();
  $("mcp-card-error").hidden = !(mcpEnabled && errors.length);
  $("mcp-card-error").textContent = errors.join("\n");
  $("mcp-card-reload").disabled = mcpBusy || !mcpEnabled;
  const indicator = $("mcp-indicator");
  const shown = conversationMode === "chat" && mcpEnabled;
  indicator.hidden = !shown;
  if (shown) {
    const tools = mcpServers().filter((s) => s.connected).reduce((n, s) => n + s.tools.length, 0);
    indicator.className = `mcp-indicator${errors.length && !mcpBusy ? " warn" : ""}`;
    indicator.innerHTML = mcpBusy ? '<span class="spinner"></span>' : errors.length ? icon("warning", 14) : `${icon("tool", 13)}<span>${tools}</span>`;
    indicator.title = mcpBusy ? t("connectingTools") : errors.length ? errors.join("\n") : mcpLine();
  }
}

function compactConfig(config) {
  return Object.fromEntries(Object.entries(config).filter(([, v]) => v !== null && v !== undefined && v !== ""
    && !(Array.isArray(v) && !v.length) && !(typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length)));
}

function editorConfig() {
  try {
    const parsed = JSON.parse($("mcp-json").value || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch { return null; }
}

function setTransport(kind) {
  mcpTransport = kind;
  for (const button of document.querySelectorAll("[data-transport]")) button.classList.toggle("selected", button.dataset.transport === kind);
  $("mcp-url-row").hidden = kind !== "https";
  $("mcp-local-rows").hidden = kind !== "local";
}

function fieldsFromConfig(config) {
  setTransport(config.command ? "local" : "https");
  $("mcp-url").value = config.url ?? "";
  $("mcp-command").value = config.command ?? "";
  $("mcp-args").value = Array.isArray(config.args) ? config.args.join("\n") : "";
  $("mcp-enabled").checked = config.enabled !== false;
}

/** Form → JSON: the transport decides which fields stay. */
function configFromFields() {
  const config = { ...(editorConfig() ?? {}) };
  if (mcpTransport === "https") {
    config.url = $("mcp-url").value.trim();
    delete config.command; delete config.args; delete config.cwd; delete config.env;
  } else {
    config.command = $("mcp-command").value.trim();
    config.args = $("mcp-args").value.split("\n").map((line) => line.trim()).filter(Boolean);
    delete config.url; delete config.headers;
  }
  config.enabled = $("mcp-enabled").checked;
  $("mcp-json").value = JSON.stringify(compactConfig(config), null, 2);
}

function editServer(name) {
  mcpEditing = name;
  const config = mcpServers().find((s) => s.name === name)?.config ?? { url: "", enabled: true };
  $("mcp-name").value = name;
  $("mcp-name").disabled = name !== "";
  $("mcp-json").value = JSON.stringify(compactConfig(config), null, 2);
  fieldsFromConfig(config);
  renderMcpPage();
}

async function saveServer() {
  const name = $("mcp-name").value.trim();
  const config = editorConfig();
  if (!name || !config) { toast(t("mcpInvalid"), 7000); return; }
  try {
    mcpStatus = await invoke("mcp_save", { name, config });
    mcpEditing = name;
    $("mcp-name").disabled = true;
  } catch (e) { toast(`${t("mcpInvalid")}\n${message(e)}`, 8000); }
  render();
}

async function removeServer() {
  if (!mcpEditing || !window.confirm(t("confirmRemoveServer"))) return;
  try { mcpStatus = await invoke("mcp_remove", { name: mcpEditing }); mcpEditing = null; } catch (e) { fail(e); }
  render();
}

function renderMcpPage() {
  if ($("mcp-page").hidden) return;
  $("mcp-servers").innerHTML = mcpServers().map((s) => `<button class="side-row${s.name === mcpEditing ? " selected" : ""}" data-server="${escapeHtml(s.name)}">
      <span class="dot ${s.connected ? "ready" : s.error ? "failed" : "idle"}"></span><span class="grow">${escapeHtml(s.name)}</span></button>`).join("");
  $("mcp-empty").hidden = mcpServers().length > 0;
  $("mcp-global").checked = mcpEnabled;
  $("mcp-global").disabled = mcpBusy;
  $("mcp-status-line").textContent = mcpLine();
  $("mcp-reload").disabled = mcpBusy || !mcpEnabled;
  $("mcp-test").disabled = mcpBusy;
  $("mcp-editor").hidden = mcpEditing === null;
  $("mcp-remove").hidden = !mcpEditing;
  const errors = [...new Set([...mcpErrors(), ...mcpTestErrors])];
  $("mcp-errors").hidden = !errors.length;
  $("mcp-errors").textContent = errors.join("\n");
  const project = currentProject();
  $("mcp-perms-title").textContent = project ? t("allowedToolsFor").replace("{name}", project.name) : t("choosePermissionsProject");
  $("mcp-perms").innerHTML = !project ? "" : mcpCatalog.length
    ? mcpCatalog.map((name) => `<label class="row-field bare switch"><span class="mono">${escapeHtml(name)}</span><input type="checkbox" data-tool-perm="${escapeHtml(name)}"${project.allowedTools.includes(name) ? " checked" : ""}></label>`).join("")
    : `<p class="quiet small">${escapeHtml(t("testToList"))}</p>`;
}

// ---------- Agent (Mac AgentWorkspace; commands are not sandboxed on Windows) ----------

async function refreshAgent() {
  agentInfo = null;
  const task = currentId;
  if (conversationMode === "agent" && task) {
    try {
      const info = await invoke("agent_open", { task });
      if (task === currentId) agentInfo = info;
    } catch (e) { log(`agent: ${message(e)}`); }
  }
  render();
}

function renderAgent() {
  const info = agentInfo && agentInfo.task === currentId ? agentInfo : null;
  $("agent-folder").textContent = info?.source ? info.source.split(/[\\/]/).filter(Boolean).pop() : t("taskWorkspace");
  const parts = [info?.files ? t("filesCount").replace("{n}", String(info.files)) : t("noFilesYet")];
  if (info?.skippedFiles) parts.push(t("excludedOnImport").replace("{n}", String(info.skippedFiles)));
  $("agent-files").textContent = parts.join(" · ");
  $("agent-folder").title = info?.source ?? "";
  const active = Boolean(session.active);
  $("agent-open").disabled = active;
  $("agent-review").disabled = active || !info?.source;
  $("agent-perms").disabled = active;
  const perms = info?.permissions ?? { read: true, write: true, run: false, commandSeconds: 5 };
  if (document.activeElement !== $("perm-seconds")) $("perm-seconds").value = String(perms.commandSeconds);
  $("perm-read").checked = perms.read;
  $("perm-write").checked = perms.write;
  $("perm-run").checked = perms.run;
  $("agent-card-run").hidden = !perms.run;
}

async function agentTask() {
  if (!currentId) currentId = newId();
  if (agentInfo?.task !== currentId) agentInfo = await invoke("agent_open", { task: currentId });
  return currentId;
}

async function importAgentFolder() {
  if (session.active) return;
  try {
    const task = await agentTask();
    const info = await invoke("agent_import", { task });
    if (info && task === currentId) agentInfo = info;
  } catch (e) { fail(e); }
  render();
}

async function setAgentPermissions() {
  try {
    const task = await agentTask();
    const permissions = {
      read: $("perm-read").checked, write: $("perm-write").checked, run: $("perm-run").checked,
      commandSeconds: Math.min(30, Math.max(1, Math.trunc(Number($("perm-seconds").value) || 5))),
    };
    agentInfo = await invoke("agent_set_permissions", { task, permissions });
  } catch (e) { fail(e); }
  render();
}

async function reviewAgentChanges() {
  if (!agentInfo?.source || session.active) return;
  const task = agentInfo.task;
  let changes;
  try { changes = await invoke("agent_changes", { task }); } catch (e) { fail(e); return; }
  if (!changes.length) { toast(t("noDifferences")); return; }
  const labels = { added: "changeAdded", modified: "changeModified", deleted: "changeDeleted" };
  const pre = (text, kind) => (typeof text === "string" ? `<pre class="diff-${kind}">${escapeHtml(text)}</pre>` : "");
  const html = changes.map((c) => `<details class="change"><summary><span class="badge ${c.kind}">${escapeHtml(t(labels[c.kind] ?? "changeModified"))}</span><span class="mono">${escapeHtml(c.path)}</span></summary>
      ${c.binary ? `<p class="quiet small">${escapeHtml(t("binaryFile"))}</p>` : `<div class="diff">${pre(c.before, "before")}${pre(c.after, "after")}</div>`}</details>`).join("");
  openSheet({
    title: t("reviewChanges"), html, note: t("agentCopyNote"),
    apply: async () => {
      if (!window.confirm(t("confirmApply"))) return;
      try {
        const count = await invoke("agent_apply", { task });
        closeSheet();
        toast(t("applied").replace("{n}", String(count)));
        if (agentInfo?.task === task) agentInfo = await invoke("agent_open", { task });
        render();
      } catch (e) { fail(e); }
    },
  });
}

function stopAgentWatch() {
  if (!agentTimers) return;
  clearTimeout(agentTimers.total);
  clearTimeout(agentTimers.silent);
  agentTimers = null;
}

/** Mac Agent watchdogs: 10 minutes per turn and 2 minutes without any engine event. */
function startAgentWatch(requestId) {
  stopAgentWatch();
  const cancel = (key) => {
    stopAgentWatch();
    if (session.active !== requestId) return;
    invoke("cancel", { requestId }).catch(fail);
    toast(t(key), 7000);
  };
  agentTimers = { requestId, cancel, total: setTimeout(() => cancel("agentTimeout"), AGENT_TOTAL_MS), silent: setTimeout(() => cancel("agentSilent"), AGENT_SILENT_MS) };
}

function touchAgentWatch() {
  if (!agentTimers) return;
  if (session.active !== agentTimers.requestId) { stopAgentWatch(); return; }
  clearTimeout(agentTimers.silent);
  const { cancel } = agentTimers;
  agentTimers.silent = setTimeout(() => cancel("agentSilent"), AGENT_SILENT_MS);
}

// ---------- Server (OpenAI-compatible endpoint on the loaded model) ----------

let serverAccess = "local";
let serverFormLoaded = false;

async function refreshServer() {
  try { serverStatus = await invoke("server_status"); } catch (e) { log(`server: ${message(e)}`); }
  if (serverStatus && !serverFormLoaded) {
    serverFormLoaded = true;
    serverAccess = serverStatus.config.host === "0.0.0.0" ? "network" : "local";
    $("server-port").value = String(serverStatus.config.port);
  }
  render();
}

function serverPort() {
  return serverStatus?.running ? serverStatus.config.port : Number($("server-port").value);
}

function renderServer() {
  const running = Boolean(serverStatus?.running);
  for (const button of document.querySelectorAll("[data-access]")) {
    button.classList.toggle("selected", button.dataset.access === serverAccess);
    button.disabled = running || serverBusy;
  }
  $("server-port").disabled = running || serverBusy;
  const model = loadedModel();
  $("server-model").textContent = model ? model.name : t("chooseModel");
  const where = serverStatus?.config?.host === "0.0.0.0" ? t("onlineNetwork") : t("online");
  $("server-state").textContent = running ? `${where} · ${serverStatus.address}` : t("stopped");
  $("server-state").classList.toggle("ok", running);
  $("server-toggle").textContent = t(running ? "stop" : "start");
  $("server-toggle").disabled = serverBusy || (!running && (session.engine !== "ready" || Boolean(session.active)));
  $("server-network-note").hidden = serverAccess !== "network";
  $("server-running").hidden = !running;
  $("server-url").textContent = `http://127.0.0.1:${serverPort()}/v1`;
  $("server-replace-note").hidden = !(running && serverStatus.config.host === "0.0.0.0");
}

async function stopServer() {
  serverBusy = true;
  render();
  try { serverStatus = await invoke("server_stop"); } catch (e) { fail(e); }
  serverBusy = false;
  render();
}

async function toggleServer() {
  if (serverBusy) return;
  if (serverStatus?.running) { await stopServer(); return; }
  const port = Number($("server-port").value);
  const invalid = !Number.isInteger(port) || port < 1024 || port > 65535 || session.engine !== "ready" || Boolean(session.active);
  $("server-invalid").hidden = !invalid;
  $("server-invalid").textContent = invalid ? t("checkPort") : "";
  if (invalid) return;
  serverKey = crypto.randomUUID(); // Mac: a new key at each start
  serverBusy = true;
  render();
  try {
    serverStatus = await invoke("server_start", { config: { host: serverAccess === "network" ? "0.0.0.0" : "127.0.0.1", port, api_key: serverKey } });
  } catch (e) { toast(`${t("serverFailed")}\n${message(e)}`, 8000); }
  serverBusy = false;
  render();
}

// ---------- Lab (Mac LabView) ----------

function labScope() {
  const model = $("lab-model").value;
  return model ? labSamples.filter((s) => s.model === model) : labSamples;
}

function fillSelect(select, items) {
  const previous = select.value;
  select.innerHTML = optionsHtml(items);
  select.value = items.some(([value]) => value === previous) ? previous : items[0]?.[0] ?? "";
}

function renderLab() {
  if (mode !== "lab") return;
  const info = session.info;
  const gpus = system ? [...system.nvidia.map((g) => g.name), ...system.amd] : [];
  const vram = system?.nvidia?.length ? system.nvidia.map((g) => `${Math.round(g.memory_mib / 1024)} GB`).join(" · ") : "—";
  const versions = [info?.exllamav3_version ? `ExLlamaV3 ${info.exllamav3_version}` : "", info?.torch_version ? `torch ${info.torch_version}` : ""].filter(Boolean).join(" · ");
  const rows = [
    ["GPU", gpus.join(" · ") || t("noGpu")], [t("vram"), vram],
    [t("device"), [info?.device, info?.backend].filter(Boolean).join(" · ") || "—"],
    [t("engineVersions"), versions || "—"], ["MLXL3", system?.app_version ?? "—"],
  ];
  $("lab-hw").innerHTML = rows.map(([key, value]) => `<div class="kv-row"><span class="quiet">${escapeHtml(key)}</span><span>${escapeHtml(value)}</span></div>`).join("");
  const names = labModels(labSamples).map((name) => [name, name]);
  fillSelect($("lab-model"), [["", t("allModels")], ...names]);
  fillSelect($("lab-history-model"), [["", t("allModels")], ...names]);
  fillSelect($("lab-metric"), Object.entries(METRICS).map(([id, [label, unit]]) => [id, `${t(label)} (${unit})`]));
  const scoped = labScope();
  $("lab-empty").hidden = labSamples.length > 0;
  $("lab-content").hidden = !labSamples.length;
  $("lab-export").disabled = !scoped.length;

  const bars = $("lab-bars");
  bars.innerHTML = latestByModel(scoped).map((row) => `<div class="lab-bar"><span class="lab-bar-name" title="${escapeHtml(row.model)}">${escapeHtml(row.model)}</span>
      <div class="bar"><i data-pct="${row.percent}"></i></div><span class="mono-meta">${row.value.toFixed(1)} tok/s</span></div>`).join("");
  for (const bar of bars.querySelectorAll("[data-pct]")) bar.style.width = `${bar.dataset.pct}%`;

  const metric = $("lab-metric").value;
  const points = chartPoints(history(labSamples, $("lab-history-model").value, metric), 640, 220);
  const series = [...new Set(points.map((p) => p.model))];
  const grid = [0.25, 0.5, 0.75].map((f) => `<line class="grid" x1="24" x2="616" y1="${24 + f * 172}" y2="${24 + f * 172}"></line>`).join("");
  const lines = series.map((model, i) => {
    const own = points.filter((p) => p.model === model);
    const path = own.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    return `<polyline class="series s${i % 6}" points="${path}"></polyline>${own.map((p) => `<circle class="series s${i % 6}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3"><title>${escapeHtml(`${model}: ${p.value.toFixed(2)} ${METRICS[metric][1]}`)}</title></circle>`).join("")}`;
  }).join("");
  $("lab-chart").innerHTML = `${grid}<line class="axis" x1="24" x2="616" y1="196" y2="196"></line>${lines}`;
  $("lab-no-values").hidden = points.length > 0;

  $("lab-details").innerHTML = scoped.slice(-30).reverse().map((s) => `<div class="lab-card">
      <div class="strong">${escapeHtml(s.model)}</div><div class="quiet small">${escapeHtml(new Date(s.date).toLocaleString(language))} · ${escapeHtml(s.mode)}${s.device ? ` · ${escapeHtml(s.device)}` : ""}</div>
      <div class="mono-meta">${escapeHtml(`${t("labDecode")} ${s.decodeTps.toFixed(1)} tok/s · ${t("labPrefill")} ${s.prefillTps.toFixed(1)} tok/s · ${t("labTtft")} ${(s.ttftSeconds * 1000).toFixed(0)} ms · ${s.promptTokens} → ${s.generatedTokens} tokens`)}</div></div>`).join("");
}

function recordLab(done) {
  const model = loadedModel();
  const s = sampling();
  const sample = labSample(done, {
    model: model?.name, bits: model?.bits, mode: conversationMode, backend: session.info?.backend, device: session.info?.device,
    context: session.info?.context_limit, temperature: s.temperature, topK: s.top_k,
  });
  if (!sample) return;
  labSamples = addSample(labSamples, sample);
  storage((x) => x.setItem(LAB_KEY, JSON.stringify(labSamples)));
  renderLab();
}

async function exportLab() {
  const scoped = labScope();
  if (!scoped.length) return;
  try {
    if (await invoke("save_text_file", { name: "MLXL3-Lab.json", contents: exportJson(scoped) })) toast(t("exported"));
  } catch (e) { fail(e); }
}

async function stop() {
  if (!session.active) return;
  try { await invoke("cancel", { requestId: session.active }); } catch (e) { fail(e); }
}

// ---------- wiring ----------

function wire() {
  // chat sidebar
  $("new-chat").onclick = startNewConversation;
  $("chat-search").oninput = renderChats;
  $("chats").onclick = (event) => {
    const pin = event.target.closest("[data-pin]");
    const del = event.target.closest("[data-delete]");
    if (pin) {
      conversations = togglePinned(conversations, pin.dataset.pin);
      persistConversations();
      renderChats();
    } else if (del) {
      if (!window.confirm(t("confirmDeleteChat"))) return;
      const id = del.dataset.delete;
      if (id === currentId) {
        if (session.active) return;
        currentId = null;
        session = { ...session, turns: [], lastStats: null };
      }
      if (conversations.find((c) => c.id === id)?.agent) {
        invoke("agent_discard", { task: id }).catch((e) => log(`agent: ${message(e)}`));
        if (agentInfo?.task === id) agentInfo = null;
      }
      conversations = removeConversation(conversations, id);
      persistConversations();
      render();
    } else {
      const row = event.target.closest("[data-chat]");
      if (row) openConversation(row.dataset.chat);
    }
  };
  $("chats").onkeydown = (event) => {
    const row = event.target.closest?.("[data-chat]");
    if (row && event.key === "Enter") openConversation(row.dataset.chat);
  };
  $("open-models").onclick = () => { showPage("models"); showModelsTab(modelsTab); };
  $("open-settings").onclick = () => showPage("settings");
  $("update-badge").onclick = () => { showPage("settings"); showSettingsPane("updates"); };
  $("downloads-indicator").onclick = () => { showPage("models"); showModelsTab("hub"); };
  $("welcome-models").onclick = () => { showPage("models"); showModelsTab(models.length ? "library" : "hub"); };

  // header
  $("toggle-sidebar").onclick = () => { $("chat-view").classList.toggle("collapsed"); savePrefs(); };
  $("toggle-inspector").onclick = () => setInspector(!inspectorOpen);
  $("scrim").onclick = () => setInspector(false);
  $("eject").onclick = unloadModel;
  $("model-button").onclick = (event) => { event.stopPropagation(); closeMenus("model-list"); $("model-list").hidden = !$("model-list").hidden; };
  $("model-list").onclick = (event) => {
    const item = event.target.closest("button");
    if (!item) return;
    $("model-list").hidden = true;
    if (item.dataset.load) loadModel(item.dataset.load);
    else if (item.dataset.menu === "add") { showPage("models"); showModelsTab("hub"); }
    else if (item.dataset.menu === "eject") unloadModel();
  };
  $("display-button").onclick = (event) => { event.stopPropagation(); closeMenus("display-pop"); $("display-pop").hidden = !$("display-pop").hidden; };
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".model-menu, .tools-menu, .display-menu, .agent-menu")) closeMenus();
  });

  // appearance controls (header popover and Settings → Appearance share the [data-set] markup)
  document.addEventListener("click", onAppearanceClick);
  for (const range of document.querySelectorAll('input[data-set="textSize"]')) {
    range.addEventListener("input", () => setAppearance({ textSize: Number(range.value) }));
  }
  for (const select of document.querySelectorAll("select[data-set]")) {
    select.addEventListener("change", () => setAppearance({ [select.dataset.set]: select.value }));
  }
  $("custom-accent").addEventListener("input", renderAppearance);
  $("use-custom-accent").onclick = () => setAppearance({ accent: "custom", custom: normalizeHex($("custom-accent").value) });
  $("reset-appearance").onclick = () => setAppearance({ ...DEFAULT_APPEARANCE });

  // composer
  $("composer").onsubmit = send;
  $("stop").onclick = stop;
  $("prompt").addEventListener("input", growPrompt);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("tools").onclick = (event) => { event.stopPropagation(); closeMenus("tools-list"); $("tools-list").hidden = !$("tools-list").hidden; };
  $("thinking-quick").onchange = () => { $("thinking").checked = $("thinking-quick").checked; savePrefs(); };
  $("tools-system").onclick = () => { closeMenus(); setInspector(true); $("system").focus(); };
  $("tools-export").onclick = exportCurrentConversation;

  // messages
  $("messages").addEventListener("click", onMessageAction);
  $("messages").addEventListener("toggle", (event) => {
    const element = event.target.closest(".turn");
    if (element && event.target.matches("details")) openThinking.set(thinkingKey(Number(element.dataset.index)), event.target.open);
  }, true);

  // inspector
  for (const id of ["temperature", "top-p", "top-k", "min-p", "rep"]) {
    $(id).addEventListener("input", () => { renderSliders(); savePrefs(); });
  }
  for (const id of ["max-tokens", "context", "system"]) $(id).addEventListener("change", savePrefs);
  $("thinking").onchange = () => { renderSliders(); savePrefs(); };
  $("reset-sampling").onclick = () => {
    for (const [id, value] of Object.entries(SAMPLING_DEFAULTS)) $(id).value = value;
    renderSliders();
    savePrefs();
  };

  // pages
  for (const button of document.querySelectorAll("[data-back]")) button.onclick = () => showPage("chat");
  $("nav-library").onclick = () => showModelsTab("library");
  $("nav-hub").onclick = () => { hub.detail = null; showModelsTab("hub"); renderHub(); };
  $("import-folder").onclick = importFolder;
  $("refresh-library").onclick = () => refreshModels().catch(fail);
  $("library-list").onclick = (event) => {
    const target = event.target.closest("button");
    if (!target) return;
    if (target.dataset.load) loadModel(target.dataset.load);
    else if (target.dataset.reveal) invoke("reveal_model", { id: target.dataset.reveal }).catch(fail);
    else if (target.dataset.remove) removeModel(target.dataset.remove);
  };
  $("hub-query").addEventListener("input", scheduleSearch);
  $("hub-query").addEventListener("keydown", (e) => { if (e.key === "Enter") { clearTimeout(searchTimer); hub.limit = HUB_PAGE; searchHub(); } });
  $("hub-refresh").onclick = () => searchHub();
  $("hub-more").onclick = () => { hub.limit = Math.min(HUB_MAX, hub.limit + HUB_PAGE); searchHub(); };
  $("hub-scroll").onclick = (event) => {
    const rec = event.target.closest("[data-rec]");
    const row = event.target.closest("[data-repo]");
    if (rec) {
      const entry = recommendedEntry(rec.dataset.rec);
      hub.detail = null;
      openDetail(entry.id, entry.revision, entry.variant ?? undefined);
    } else if (row) {
      hub.detail = null;
      openDetail(row.dataset.repo);
    }
  };
  $("detail-back").onclick = () => { hub.detail = null; showModelsTab("hub"); renderHub(); };
  $("detail-hub").onclick = () => hub.detail && invoke("open_model_page", { repo: hub.detail.id }).catch(fail);
  $("detail-revision").onchange = () => openDetail(hub.detail.id, $("detail-revision").value);
  $("detail-quant").onchange = () => {
    const choice = recommendedChoices(hub.detail.id).find((c) => c.id === $("detail-quant").value);
    if (choice) openDetail(choice.id, choice.revision, "");
  };
  $("detail-variant").onchange = () => { hub.variantId = $("detail-variant").value; renderDetail(); };
  $("detail-download").onclick = startDownload;
  $("side-downloads").onclick = onDownloadsClick;

  // settings
  for (const button of document.querySelectorAll("[data-settings]")) button.onclick = () => showSettingsPane(button.dataset.settings);
  $("language").onchange = () => {
    savePrefs();
    applyLanguage();
    $("recommended").innerHTML = "";
    renderHub();
    render();
    renderRuntime();
    renderUpdate();
    renderStartupModels();
    renderAccount();
    if (hub.detail) renderDetail();
  };
  $("theme").onchange = () => { savePrefs(); applyTheme(); };
  $("notifications").onchange = savePrefs;
  $("load-on-startup").onchange = () => { savePrefs(); renderStartupModels(); };
  $("startup-model").onchange = () => { startupModel = $("startup-model").value; savePrefs(); };
  $("idle-unload").onchange = () => { savePrefs(); resetIdleUnload(); };
  $("export-history").onclick = exportHistory;
  $("import-history").onclick = importHistory;
  $("hf-sign-in").onclick = signIn;
  $("hf-sign-out").onclick = signOut;
  $("hf-token").addEventListener("keydown", (e) => { if (e.key === "Enter") signIn(); });
  $("update-channel").onchange = () => invoke("set_update_channel", { channel: $("update-channel").value }).catch(fail);
  $("backend").onchange = () => { savePrefs(); renderRuntime(); render(); };
  $("distro").onchange = () => { savePrefs(); renderRuntime(); };
  $("install-runtime").onclick = installRuntime;
  $("update-check").onclick = () => invoke("update_check").catch(fail);
  $("update-install").onclick = installUpdate;
  $("update-notes").onclick = () => invoke("open_release_page").catch(fail);

  // modes, projects, workspace page
  for (const button of document.querySelectorAll("[data-mode]")) button.onclick = () => selectMode(button.dataset.mode);
  $("project-select").onchange = () => chooseProject($("project-select").value || null);
  $("open-workspace").onclick = () => showPage("workspace");
  $("inspector-workspace").onclick = () => { setInspector(false); showPage("workspace"); };
  for (const button of document.querySelectorAll("[data-ws]")) {
    button.onclick = () => { wsTab = button.dataset.ws; renderWorkspacePage(); if (wsTab === "documents") refreshDocs(); };
  }
  $("ws-project").onchange = () => { chooseProject($("ws-project").value || null); renderWorkspacePage(); };
  $("ws-create").onclick = createProject;
  $("ws-new-name").addEventListener("keydown", (e) => { if (e.key === "Enter") createProject(); });
  $("ws-name").onchange = () => {
    if ($("ws-name").value.trim()) updateProject({ name: $("ws-name").value });
    else $("ws-name").value = currentProject()?.name ?? "";
  };
  $("ws-instructions").oninput = () => updateProject({ instructions: $("ws-instructions").value });
  $("ws-profile").onchange = () => updateProject({ profileId: $("ws-profile").value || null });
  $("ws-remove-project").onclick = removeProject;
  $("ws-view-projects").onclick = () => { wsTab = "projects"; renderWorkspacePage(); };
  $("ws-add-files").onclick = () => addDocs(false);
  $("ws-add-folder").onclick = () => addDocs(true);
  $("ws-doc-query").oninput = scheduleDocSearch;
  $("ws-doc-list").onclick = async (event) => {
    const remove = event.target.closest("[data-remove-doc]");
    const result = event.target.closest("[data-result]");
    if (remove) {
      try { await invoke("documents_remove", { id: remove.dataset.removeDoc }); } catch (e) { fail(e); }
      await refreshDocs();
    } else if (result) {
      const x = docResults?.[Number(result.dataset.result)];
      if (x) openSheet({ title: x.label, text: x.text, note: x.original_path });
    }
  };
  $("ws-save-profile").onclick = saveProfile;
  $("ws-profile-name").addEventListener("keydown", (e) => { if (e.key === "Enter") saveProfile(); });
  $("ws-profiles").onclick = (event) => {
    const apply = event.target.closest("[data-apply-profile]");
    const del = event.target.closest("[data-delete-profile]");
    const id = (apply ?? del)?.dataset.applyProfile ?? del?.dataset.deleteProfile;
    const profile = workspace.profiles.find((p) => p.id === id);
    if (!profile) return;
    if (apply) applyProfile(profile); else deleteProfile(profile.id);
  };

  // sheet
  $("sheet-close").onclick = closeSheet;
  $("sheet").onclick = (event) => { if (event.target === $("sheet")) closeSheet(); };
  $("sheet-apply").onclick = () => sheetApply?.();

  // attachments, validated JSON
  $("tools-attach").onclick = () => { closeMenus(); importAttachments(() => invoke("pick_attachments")); };
  $("attachments").onclick = onAttachmentsClick;
  $("prompt").addEventListener("paste", onPaste);
  $("schema").addEventListener("change", savePrefs);
  $("schema-clear").onclick = () => { $("schema").value = ""; savePrefs(); };

  // MCP
  $("tools-mcp").onchange = () => setMcp($("tools-mcp").checked);
  $("tools-configure").onclick = () => showPage("mcp");
  $("open-mcp").onclick = () => showPage("mcp");
  $("mcp-card-configure").onclick = () => { setInspector(false); showPage("mcp"); };
  $("mcp-card-reload").onclick = () => setMcp(true, true);
  $("mcp-global").onchange = () => setMcp($("mcp-global").checked);
  $("mcp-reload").onclick = () => setMcp(true, true);
  $("mcp-test").onclick = testMcp;
  $("mcp-new").onclick = () => editServer("");
  $("mcp-servers").onclick = (event) => { const row = event.target.closest("[data-server]"); if (row) editServer(row.dataset.server); };
  for (const button of document.querySelectorAll("[data-transport]")) button.onclick = () => { setTransport(button.dataset.transport); configFromFields(); };
  for (const id of ["mcp-url", "mcp-command", "mcp-args"]) $(id).addEventListener("input", configFromFields);
  $("mcp-enabled").onchange = configFromFields;
  $("mcp-json").addEventListener("input", () => { const config = editorConfig(); if (config) fieldsFromConfig(config); });
  $("mcp-save").onclick = saveServer;
  $("mcp-remove").onclick = removeServer;
  $("mcp-perms").onchange = (event) => {
    const box = event.target.closest("[data-tool-perm]");
    const project = currentProject();
    if (!box || !project) return;
    const allowed = new Set(project.allowedTools);
    if (box.checked) allowed.add(box.dataset.toolPerm); else allowed.delete(box.dataset.toolPerm);
    updateProject({ allowedTools: [...allowed] });
  };

  // Agent
  $("agent-open").onclick = importAgentFolder;
  $("agent-perms").onclick = (event) => { event.stopPropagation(); closeMenus("agent-pop"); $("agent-pop").hidden = !$("agent-pop").hidden; };
  for (const id of ["perm-read", "perm-write", "perm-run", "perm-seconds"]) $(id).addEventListener("change", setAgentPermissions);
  $("agent-review").onclick = reviewAgentChanges;

  // Server
  for (const button of document.querySelectorAll("[data-access]")) button.onclick = () => { serverAccess = button.dataset.access; render(); };
  $("server-port").addEventListener("input", renderServer);
  $("server-toggle").onclick = toggleServer;
  $("server-copy-url").onclick = (e) => copy(`http://127.0.0.1:${serverPort()}/v1`, e.currentTarget);
  $("server-copy-key").onclick = (e) => copy(serverStatus?.config?.api_key ?? "", e.currentTarget, "lock");
  $("server-copy-curl").onclick = (e) => copy(`curl -H 'Authorization: Bearer ${serverStatus?.config?.api_key ?? ""}' http://127.0.0.1:${serverPort()}/v1/models`, e.currentTarget, "terminal");

  // Lab
  for (const id of ["lab-model", "lab-history-model", "lab-metric"]) $(id).onchange = renderLab;
  $("lab-export").onclick = exportLab;

  // keyboard
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (!$("sheet").hidden) closeSheet();
      else if (MENUS.some((id) => !$(id).hidden)) closeMenus();
      else if (!$("models-page").hidden && hub.detail && modelsTab === "hub") { hub.detail = null; showModelsTab("hub"); renderHub(); }
      else if (["models-page", "settings-page", "workspace-page", "mcp-page"].some((id) => !$(id).hidden)) showPage("chat");
      else if (inspectorOpen) setInspector(false);
    } else if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === "n") {
      event.preventDefault();
      startNewConversation();
    }
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { paintAurora($("aurora")); if (inspectorOpen) setInspector(true); }, 120);
  });
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => paintAurora($("aurora")));
}

async function subscribe() {
  await listen("engine-started", ({ payload }) => setSession(engineStarted(session, payload.epoch)));
  await listen("engine-event", ({ payload }) => {
    const finished = payload.event?.type === "done" && payload.event.request_id === session.active;
    const agentTurn = conversationMode === "agent";
    setSession(applyEngineEvent(session, payload.epoch, payload.event));
    touchAgentWatch();
    if (finished) {
      recordLab(payload.event);
      if (agentTurn && payload.event.stop_reason === "max_tokens") toast(t("agentLimit"), 7000);
    }
    // Mac CompletionNotifications: only when the app is in the background, never with conversation content.
    if (finished && !session.active && $("notifications").checked && !document.hasFocus()) {
      invoke("notify_done", { title: t("notificationTitle"), body: t("notificationBody") }).catch((e) => log(`notification: ${message(e)}`));
    }
  });
  await listen("engine-exited", ({ payload }) => {
    const tail = (payload.stderr_tail || "").trim().split("\n").slice(-3).join("\n");
    if (payload.stderr_tail) log(payload.stderr_tail);
    setSession(engineExited(session, payload.epoch, `${t("engineStopped")} (code ${payload.code ?? "?"})${tail ? `: ${tail}` : ""}`));
  });
  await listen("engine-log", ({ payload }) => log(payload.line));
  await listen("runtime-log", ({ payload }) => log(payload.line));
  await listen("runtime-progress", ({ payload }) => {
    const bar = $("runtime-progress");
    bar.max = payload.total;
    bar.value = payload.step;
    $("runtime-status").textContent = `[${payload.step}/${payload.total}] ${payload.label}`;
    log(`[${payload.step}/${payload.total}] ${payload.label}`);
  });
  await listen("download-progress", ({ payload }) => {
    download = { ...payload.download, done: payload.done, total: payload.total };
    renderDownloads();
  });
  await listen("download-finished", async ({ payload }) => {
    download = null;
    toast(t("downloadDone").replace("{name}", payload.model?.name ?? payload.download.name));
    await refreshModels().catch(fail);
    await refreshPending();
    if (hub.detail) renderDetail();
  });
  await listen("download-failed", async ({ payload }) => {
    download = null;
    if (!payload.paused) fail(`${t("downloadFailed")}: ${payload.error}`);
    await refreshPending();
    if (hub.detail) renderDetail();
  });
  await listen("update-state", ({ payload }) => { updateStatus = payload; renderUpdate(); });
  await listen("server-log", ({ payload }) => {
    const el = $("server-log");
    el.textContent = (el.textContent + String(payload) + "\n").slice(-16000);
  });
  // Files dropped on the window become composer attachments (Chat and Agent).
  await listen("tauri://drag-enter", () => $("composer").classList.add("dragging"));
  await listen("tauri://drag-leave", () => $("composer").classList.remove("dragging"));
  await listen("tauri://drag-drop", ({ payload }) => {
    $("composer").classList.remove("dragging");
    const paths = Array.isArray(payload?.paths) ? payload.paths.filter((p) => typeof p === "string") : [];
    if (!paths.length || $("chat-view").hidden || (mode !== "chat" && mode !== "agent")) return;
    importAttachments(() => invoke("read_attachments", { paths }));
  });
}

async function main() {
  fillIcons();
  loadPrefs();
  loadWorkspace();
  applyLanguage();
  applyTheme();
  renderAppearance();
  renderSliders();
  restoreState();
  wire();
  render();
  await subscribe();
  updateStatus = await invoke("update_status").catch(() => null);
  renderUpdate();
  await refreshModels().catch(fail);
  await refreshPending();
  await refreshSystem().catch(fail);
  const preferred = prefs.backend || (system?.recommendation?.backend ?? "cuda");
  if (["cuda", "rocm_wsl"].includes(preferred)) $("backend").value = preferred;
  renderRuntime();
  render();
  refreshAccount();
  refreshServer();
  if (mcpEnabled) setMcp(true); else refreshMcp();
  await autoLoadOnStartup();
}

main().catch(fail);
