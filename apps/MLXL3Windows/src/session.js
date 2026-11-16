// Pure chat/engine state machine. The UI feeds it Tauri events; tests drive it directly.

export function initialSession() {
  return { epoch: 0, engine: "idle", info: null, error: null, turns: [], active: null, lastStats: null, phase: null };
}

export function engineStarted(state, epoch) {
  if (epoch < state.epoch) return state;
  if (epoch === state.epoch && state.engine !== "idle") return state;
  return { ...state, epoch, engine: "loading", info: null, error: null, active: null, phase: null };
}

/**
 * `extra` carries what the composer added: `context` (the text the model sees, with attached
 * files and document excerpts), `attachments` (names), `images` ({mime, data}) and the
 * `sources` cited by the answer.
 */
export function startTurn(state, requestId, prompt, extra = {}) {
  if (state.engine !== "ready" || state.active) throw new Error("engine is not ready");
  const attachments = extra.attachments ?? [];
  if (!prompt.trim() && !attachments.length) throw new Error("message is empty");
  const user = { role: "user", content: prompt, complete: true };
  if (typeof extra.context === "string" && extra.context !== prompt) user.context = extra.context;
  if (attachments.length) user.attachments = attachments;
  if (extra.images?.length) user.images = extra.images;
  const answer = { role: "assistant", content: "", complete: false, requestId };
  if (extra.sources?.length) answer.sources = extra.sources;
  const turns = [...state.turns, user, answer];
  return { ...state, turns, active: requestId, phase: "queued", status: null, error: null };
}

/** Undo startTurn when the request never reached the engine. */
export function abortTurn(state, requestId, message) {
  if (state.active !== requestId) return state;
  const turns = state.turns.map((t) => (t.requestId === requestId ? { ...t, failed: message } : t));
  return { ...state, turns, active: null, phase: null, error: message };
}

function updateTurn(state, requestId, change) {
  const turns = state.turns.map((t) => (t.role === "assistant" && t.requestId === requestId ? change(t) : t));
  return { ...state, turns };
}

/** Events of a newer engine can arrive before its engine-started notice. */
function adopt(state, epoch) {
  return Number.isInteger(epoch) && epoch > state.epoch ? engineStarted(state, epoch) : state;
}

export function applyEngineEvent(state, epoch, event) {
  if (!event || typeof event.type !== "string") return state;
  state = adopt(state, epoch);
  if (epoch !== state.epoch) return state;
  const id = event.request_id;
  switch (event.type) {
    case "loading":
      return { ...state, engine: "loading" };
    case "ready":
      return { ...state, engine: "ready", info: event, error: null };
    case "generation_status":
      return id === state.active ? { ...state, phase: event.phase, status: typeof event.text === "string" ? event.text.slice(0, 200) : null } : state;
    case "delta":
      if (id !== state.active) return state;
      return { ...updateTurn(state, id, (t) => ({ ...t, content: t.content + String(event.text ?? "") })), phase: "decode" };
    case "tool_start":
      if (id !== state.active) return state;
      return {
        ...updateTurn(state, id, (t) => ({ ...t, tools: [...(t.tools ?? []), toolActivity(event, "running")].slice(-MAX_TOOLS) })),
        phase: "tool",
      };
    case "tool_result":
      if (id !== state.active) return state;
      return updateTurn(state, id, (t) => {
        const tools = [...(t.tools ?? [])];
        const at = tools.findLastIndex((x) => x.id === event.tool_call_id);
        const done = { ...toolActivity(event, event.is_error ? "failed" : "complete"), result: String(event.text ?? "").slice(0, MAX_TOOL_TEXT) };
        if (at === -1) tools.push(done); else tools[at] = { ...tools[at], ...done, server: tools[at].server || done.server };
        return { ...t, tools: tools.slice(-MAX_TOOLS) };
      });
    case "done":
      if (id !== state.active) return state;
      return {
        ...updateTurn(state, id, (t) => {
          const next = { ...t, complete: true, stats: event };
          if (event.schema === "valid") next.schema = "valid";
          const context = cleanTurnContext(event.turn_context);
          if (context.length) next.turnContext = context;
          return next;
        }),
        active: null, phase: null, lastStats: event,
      };
    case "cancelled":
      if (id !== state.active) return state;
      return { ...updateTurn(state, id, (t) => ({ ...t, complete: t.content.length > 0, cancelled: true })), active: null, phase: null };
    case "error": {
      const message = String(event.message ?? "engine error");
      if (event.fatal) {
        const failed = state.active ? updateTurn(state, state.active, (t) => ({ ...t, failed: message })) : state;
        return { ...failed, engine: "failed", error: message, active: null, phase: null };
      }
      if (id && id === state.active) {
        return { ...updateTurn(state, id, (t) => ({ ...t, failed: message })), active: null, phase: null, error: message };
      }
      return { ...state, error: message };
    }
    default:
      return state;
  }
}

export function engineExited(state, epoch, detail) {
  state = adopt(state, epoch);
  if (epoch !== state.epoch) return state;
  const message = detail || "engine exited";
  const failed = state.active ? updateTurn(state, state.active, (t) => ({ ...t, failed: message })) : state;
  const wasFailed = state.engine === "failed";
  return { ...failed, engine: wasFailed ? "failed" : "stopped", error: wasFailed ? state.error : message, active: null, phase: null };
}

export function unloaded(state) {
  return { ...state, epoch: state.epoch + 1, engine: "idle", info: null, active: null, phase: null };
}

/**
 * Drop the user turn at `index` and everything after it, for "edit" and "regenerate".
 * Returns the new state and that turn's text.
 */
export function rewind(state, index) {
  if (state.active) throw new Error("stop the current answer first");
  const turn = state.turns[index];
  if (!turn || turn.role !== "user") throw new Error("not a user message");
  return { state: { ...state, turns: state.turns.slice(0, index), error: null }, prompt: turn.content };
}

export function newConversation(state) {
  if (state.active) throw new Error("stop the current answer first");
  return { ...state, turns: [], lastStats: null, error: null };
}

const MAX_SOURCES = 8;
const MAX_TOOLS = 32;
const MAX_TOOL_TEXT = 16 * 1024;
const MAX_CONTEXT_MESSAGES = 24;
const MAX_CONTEXT_TEXT = 64 * 1024;

function toolActivity(event, status) {
  return {
    id: String(event.tool_call_id ?? "").slice(0, 200),
    name: String(event.tool_name ?? "").slice(0, 200),
    server: String(event.server_name ?? "").slice(0, 100),
    status,
  };
}

/** Tool activity rows kept on an answer (Mac ToolActivity). */
export function cleanTools(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(-MAX_TOOLS).flatMap((x) => (x && typeof x.name === "string" && ["running", "complete", "failed"].includes(x.status)
    ? [{
      id: String(x.id ?? "").slice(0, 200), name: x.name.slice(0, 200), server: String(x.server ?? "").slice(0, 100),
      // A row still "running" after a restart was interrupted.
      status: x.status === "running" ? "failed" : x.status,
      ...(typeof x.result === "string" ? { result: x.result.slice(0, MAX_TOOL_TEXT) } : {}),
    }] : []));
}

/**
 * The intermediate tool rounds of an answer (assistant calls, tool results, final raw answer),
 * replayed in later turns so the model keeps its tool results. Anything malformed drops the
 * whole context rather than sending a broken dialogue.
 */
export function cleanTurnContext(value) {
  if (!Array.isArray(value) || value.length > MAX_CONTEXT_MESSAGES) return [];
  let size = 0;
  const out = [];
  for (const m of value) {
    if (!m || !["assistant", "tool"].includes(m.role) || typeof m.content !== "string") return [];
    size += m.content.length;
    const entry = { role: m.role, content: m.content };
    if (m.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length) {
      if (!m.tool_calls.every((c) => c && typeof c.function?.name === "string")) return [];
      entry.tool_calls = m.tool_calls.slice(0, 64).map((c) => ({ type: "function", function: { name: c.function.name, arguments: c.function.arguments ?? {} } }));
      size += JSON.stringify(entry.tool_calls).length;
    }
    if (m.role === "tool" && typeof m.name === "string") entry.name = m.name.slice(0, 256);
    out.push(entry);
  }
  if (size > MAX_CONTEXT_TEXT || out.at(-1)?.role !== "assistant") return [];
  return out;
}
const text = (value, max) => (typeof value === "string" ? value.slice(0, max) : null);

/** Citation excerpts kept on an answer (document_search results). Malformed entries drop. */
export function cleanSources(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, MAX_SOURCES).flatMap((s) => {
    if (!s || typeof s.id !== "string" || !/^[A-Za-z0-9-]{1,80}$/.test(s.id) || typeof s.text !== "string") return [];
    const page = Number.isInteger(s.page) && s.page > 0 ? s.page : null;
    const fileName = text(s.file_name, 260) ?? "";
    return [{
      id: s.id, document_id: text(s.document_id, 80) ?? "", file_name: fileName,
      original_path: text(s.original_path, 4096) ?? "", page, text: s.text.slice(0, 4000),
      label: fileName + (page ? ` · p. ${page}` : ""),
    }];
  });
}

/** Names (and kinds) of the files attached to a user message. */
export function cleanAttachments(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 8).flatMap((a) => (a && typeof a.name === "string" && (a.kind === "text" || a.kind === "image")
    ? [{ name: a.name.slice(0, 260), kind: a.kind }] : []));
}

function persistable(t) {
  const out = { role: t.role, content: t.content, complete: true };
  if (t.role === "user") {
    const attachments = cleanAttachments(t.attachments);
    if (attachments.length) out.attachments = attachments;
    if (typeof t.context === "string" && t.context !== t.content) out.context = t.context;
  } else {
    const sources = cleanSources(t.sources);
    if (sources.length) out.sources = sources;
    if (t.schema === "valid" || t.schema === "invalid") out.schema = t.schema;
    const tools = cleanTools(t.tools);
    if (tools.length) out.tools = tools;
    const context = cleanTurnContext(t.turnContext);
    if (context.length) out.turnContext = context;
  }
  return out;
}

/**
 * Persistable subset; in-flight turns are dropped as incomplete. Image bytes are not kept:
 * a reopened conversation keeps the image names, and the model no longer sees the pixels.
 */
export function serializeTurns(turns) {
  return JSON.stringify(turns.filter((t) => t.complete).map(persistable));
}

export function restoreTurns(raw) {
  try {
    const value = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    return value
      .filter((t) => t && (t.role === "user" || t.role === "assistant") && typeof t.content === "string")
      .map((t) => persistable({ ...t, context: typeof t.context === "string" ? t.context : undefined }));
  } catch {
    return [];
  }
}
