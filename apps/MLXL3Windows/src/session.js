// Pure chat/engine state machine. The UI feeds it Tauri events; tests drive it directly.

export function initialSession() {
  return { epoch: 0, engine: "idle", info: null, error: null, turns: [], active: null, lastStats: null, phase: null };
}

export function engineStarted(state, epoch) {
  if (epoch < state.epoch) return state;
  if (epoch === state.epoch && state.engine !== "idle") return state;
  return { ...state, epoch, engine: "loading", info: null, error: null, active: null, phase: null };
}

export function startTurn(state, requestId, prompt) {
  if (state.engine !== "ready" || state.active) throw new Error("engine is not ready");
  if (!prompt.trim()) throw new Error("message is empty");
  const turns = [
    ...state.turns,
    { role: "user", content: prompt, complete: true },
    { role: "assistant", content: "", complete: false, requestId },
  ];
  return { ...state, turns, active: requestId, phase: "queued", error: null };
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
      return id === state.active ? { ...state, phase: event.phase } : state;
    case "delta":
      if (id !== state.active) return state;
      return { ...updateTurn(state, id, (t) => ({ ...t, content: t.content + String(event.text ?? "") })), phase: "decode" };
    case "done":
      if (id !== state.active) return state;
      return { ...updateTurn(state, id, (t) => ({ ...t, complete: true, stats: event })), active: null, phase: null, lastStats: event };
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

export function newConversation(state) {
  if (state.active) throw new Error("stop the current answer first");
  return { ...state, turns: [], lastStats: null, error: null };
}

/** Persistable subset; in-flight turns are dropped as incomplete. */
export function serializeTurns(turns) {
  return JSON.stringify(turns.filter((t) => t.complete).map(({ role, content }) => ({ role, content, complete: true })));
}

export function restoreTurns(raw) {
  try {
    const value = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    return value
      .filter((t) => t && (t.role === "user" || t.role === "assistant") && typeof t.content === "string")
      .map((t) => ({ role: t.role, content: t.content, complete: true }));
  } catch {
    return [];
  }
}
