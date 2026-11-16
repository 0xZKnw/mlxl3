// Pure helpers behind the Mac-style UI: the conversation list, greetings, the recommended
// catalog, catalog labels and the update badge.
import { restoreTurns, serializeTurns } from "./session.js";

export const MAX_CONVERSATIONS = 200;

export function newId(now = Date.now(), random = Math.random) {
  return `c${now.toString(36)}${Math.floor(random() * 1e9).toString(36)}`;
}

/** First line of the first user message, at most 60 characters. */
export function titleFrom(turns, fallback) {
  const first = turns.find((t) => t.role === "user" && t.content.trim());
  if (!first) return fallback;
  const line = first.content.trim().split("\n")[0].trim();
  return line.length > 60 ? `${line.slice(0, 59)}…` : line;
}

/** Insert or replace `conversation`, newest first, capped. */
export function upsertConversation(list, conversation) {
  const rest = list.filter((c) => c.id !== conversation.id);
  return [conversation, ...rest].sort((a, b) => b.updated - a.updated).slice(0, MAX_CONVERSATIONS);
}

/** Flip the pinned flag of one conversation; order is unchanged. */
export function togglePinned(list, id) {
  return list.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c));
}

/** Sidebar sections, like the Mac: pinned first, then recent (both newest first). */
export function sections(list) {
  return { pinned: list.filter((c) => c.pinned), recent: list.filter((c) => !c.pinned) };
}

export function removeConversation(list, id) {
  return list.filter((c) => c.id !== id);
}

export function filterConversations(list, query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return list;
  return list.filter((c) => c.title.toLowerCase().includes(needle)
    || c.turns.some((t) => t.content.toLowerCase().includes(needle)));
}

export function serializeConversations(list) {
  return JSON.stringify(list.map((c) => ({
    id: c.id, title: c.title, updated: c.updated, pinned: Boolean(c.pinned), turns: JSON.parse(serializeTurns(c.turns)),
    ...(c.projectId ? { projectId: c.projectId } : {}), ...(c.agent ? { agent: true } : {}),
  })));
}

/** Parse stored conversations; drops anything malformed instead of failing. */
export function restoreConversations(raw) {
  let parsed;
  try { parsed = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(parsed)) return [];
  const seen = new Set();
  const result = [];
  for (const item of parsed) {
    if (!item || typeof item.id !== "string" || !/^[a-z0-9]{1,40}$/.test(item.id) || seen.has(item.id)) continue;
    if (typeof item.title !== "string" || !Number.isFinite(item.updated)) continue;
    seen.add(item.id);
    result.push({
      id: item.id, title: item.title.slice(0, 120), updated: item.updated, pinned: item.pinned === true,
      turns: restoreTurns(JSON.stringify(item.turns ?? [])),
      projectId: typeof item.projectId === "string" && /^[A-Za-z0-9-]{1,64}$/.test(item.projectId) ? item.projectId : null,
      agent: item.agent === true,
    });
  }
  return result.sort((a, b) => b.updated - a.updated).slice(0, MAX_CONVERSATIONS);
}

export const BACKUP_FORMAT = "mlxl3-windows-conversations";

/** History backup (Settings → Conversations → Export), versioned like the Mac workspace snapshot. */
export function backupJson(list, now = new Date()) {
  return JSON.stringify({
    format: BACKUP_FORMAT, version: 1, exported: now.toISOString(),
    conversations: JSON.parse(serializeConversations(list)),
  }, null, 2);
}

/**
 * Merge a backup into `list` like the Mac `importWorkspace`: only conversations whose id is
 * unknown are added (an import never overwrites a local conversation). Throws on a file that
 * is not an MLXL3 backup.
 */
export function mergeBackup(list, raw) {
  let parsed;
  try { parsed = JSON.parse(raw); } catch { throw new Error("not a JSON file"); }
  const items = Array.isArray(parsed) ? parsed
    : parsed && parsed.format === BACKUP_FORMAT && Array.isArray(parsed.conversations) ? parsed.conversations : null;
  if (!items) throw new Error("not an MLXL3 conversation backup");
  const known = new Set(list.map((c) => c.id));
  const incoming = restoreConversations(JSON.stringify(items)).filter((c) => !known.has(c.id) && c.turns.length);
  const merged = [...incoming, ...list].sort((a, b) => b.updated - a.updated).slice(0, MAX_CONVERSATIONS);
  const kept = new Set(merged.map((c) => c.id));
  return { list: merged, added: incoming.filter((c) => kept.has(c.id)).length };
}

/**
 * Mac "Branch from here": the completed turns up to and including `index`. Branching from a
 * user message works like the Mac "Edit in new branch": that message goes back to the draft.
 */
export function branchOf(turns, index) {
  if (!Number.isInteger(index) || index < 0 || index >= turns.length) throw new Error("no such message");
  const kept = turns.slice(0, index + 1).filter((turn) => turn.complete);
  const draft = kept.at(-1)?.role === "user" ? kept.pop().content : "";
  return { turns: kept, draft };
}

/** Mac `MarkdownExport.render`: title, then "## User" / "## MLXL3" sections. */
export function markdownExport(conversation, answerOf = (text) => text) {
  const sections = [`# ${String(conversation.title).replace(/\n/g, " ")}`];
  for (const turn of conversation.turns) {
    const user = turn.role === "user";
    let text = `## ${user ? "User" : "MLXL3"}\n\n${user ? turn.content : answerOf(turn.content)}`;
    for (const file of turn.attachments ?? []) text += `\n\n**Attachment:** ${file.name}`;
    if (turn.sources?.length) {
      text += `\n\n### Sources\n\n${turn.sources.map((s) => `- [${s.id}] ${s.label}\n${s.text.split("\n").map((line) => `> ${line}`).join("\n")}`).join("\n\n")}`;
    }
    if (!user && !turn.complete && !turn.failed && !turn.cancelled) text += "\n\n**Generation in progress — partial answer.**";
    if (turn.failed) text += `\n\n**Error:** ${turn.failed}`;
    sections.push(text);
  }
  return `${sections.join("\n\n")}\n`;
}

/** 1234567 → "1.2M", 12345 → "12.3k". */
export function formatCount(value) {
  if (!Number.isFinite(value) || value < 0) return "0";
  if (value >= 1e6) return `${(value / 1e6).toFixed(1).replace(/\.0$/, "")}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1).replace(/\.0$/, "")}k`;
  return String(Math.trunc(value));
}

/** "owner/Model-exl3" → { owner, name }. */
export function splitRepo(id) {
  const [owner, ...rest] = String(id).split("/");
  return { owner, name: rest.join("/") || owner };
}

/** "arch · EXL3 · 4.00 BPW" like the Mac library card. */
export function modelSubtitle(model) {
  const parts = [];
  if (model.architecture) parts.push(model.architecture.replace(/ForCausalLM$|ForConditionalGeneration$/, ""));
  parts.push("EXL3");
  if (Number.isFinite(model.bits)) parts.push(`${model.bits.toFixed(2)} BPW`);
  return parts.join(" · ");
}

export function percent(done, total) {
  if (!Number.isFinite(done) || !Number.isFinite(total) || total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.floor((done / total) * 100)));
}

/** Sidebar badge for the update status sent by the shell; null when nothing to show. */
export function updateBadge(status, t) {
  if (!status || !status.update) return null;
  const version = status.update.version;
  if (status.phase === "ready") return { kind: "ready", label: t("updateReady").replace("{v}", version) };
  if (status.phase === "downloading") {
    return { kind: "progress", label: `${t("updateDownloading").replace("{v}", version)} ${percent(status.done, status.total)}%` };
  }
  if (status.phase === "failed") return { kind: "failed", label: t("updateFailed") };
  return null;
}

// ---------- welcome greeting (same pools and pick rule as the Mac app) ----------

const GREETINGS = {
  en: {
    morning: ["Good morning", "Awake yet?", "Coffee and an idea?", "Where do we start?"],
    afternoon: ["Good afternoon", "Back at it?", "What are we building?"],
    evening: ["Good evening", "One more idea?", "One last one tonight?"],
    night: ["Still up?", "Ideas come at night", "Pulling an all-nighter?"],
    anytime: ["What's on your mind?", "Where to begin?", "I'm listening"],
  },
  fr: {
    morning: ["Bonjour", "Bien réveillé ?", "Un café, une idée ?", "On commence par quoi ?"],
    afternoon: ["Bon après-midi", "On s'y remet ?", "Qu'est-ce qu'on construit ?"],
    evening: ["Bonsoir", "Encore une idée ?", "Une dernière pour ce soir ?"],
    night: ["Encore debout ?", "Les idées viennent la nuit", "Nuit blanche ?"],
    anytime: ["De quoi parle-t-on ?", "Par où commencer ?", "Je vous écoute"],
  },
};

export function greetingPeriod(hour) {
  if (hour >= 5 && hour <= 11) return "morning";
  if (hour >= 12 && hour <= 17) return "afternoon";
  if (hour >= 18 && hour <= 22) return "evening";
  return "night";
}

/** One greeting for `hour` (0–23) and a non-negative integer `seed`. */
export function greeting(hour, seed, language = "en") {
  const pools = GREETINGS[language] || GREETINGS.en;
  const value = Math.abs(Math.trunc(seed)) || 0;
  const pool = value % 4 === 3 ? pools.anytime : pools[greetingPeriod(hour)];
  return pool[Math.floor(value / 4) % pool.length];
}

export const GREETING_POOLS = GREETINGS;

// ---------- recommended Hugging Face models (same list and pinned revisions as the Mac app) ----------

export const RECOMMENDED = [
  { family: "LFM", models: [
    { id: "0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw", title: "LFM3", parameters: "1.7B · Heretic NX", quantization: "8 bpw",
      icon: "wave", revision: "25511dcbc1f7c099509361cf1e2e9de034c56e22", variant: "" },
    { id: "0xzknw/LFM3-1.7B-EXL3-8bpw", title: "LFM3", parameters: "1.7B · Original", quantization: "4 · 4.49 · 8 bpw",
      icon: "wave", revision: "19aa14c461813d7a5abc462bdf2af2d90f7b771f", variant: "", choices: [
        { id: "0xzknw/LFM3-1.7B-EXL3-8bpw", label: "8 bpw", revision: "19aa14c461813d7a5abc462bdf2af2d90f7b771f" },
        { id: "0xzknw/LFM3-1.7B-EXL3-4.49bpw", label: "4.49 bpw", revision: "783640451c6b960b0db8c79b066c9416d3a782dc" },
        { id: "0xzknw/LFM3-1.7B-EXL3-4bpw", label: "4 bpw", revision: "26f5e42026aab68cd0797207f655748fe760c68c" },
      ] },
    { id: "0xzknw/LFM3-VL-2B-EXL3-8bpw", title: "LFM3-VL", parameters: "2B · Vision", quantization: "4 · 4.49 · 8 bpw",
      icon: "photo", revision: "1e634d4bf8d3ede00988c1405121e8030b8bcde6", variant: "", choices: [
        { id: "0xzknw/LFM3-VL-2B-EXL3-8bpw", label: "8 bpw", revision: "1e634d4bf8d3ede00988c1405121e8030b8bcde6" },
        { id: "0xzknw/LFM3-VL-2B-EXL3-4.49bpw", label: "4.49 bpw", revision: "00f0e5078ad49f1ca1afabd7459bcaf07ea85cd3" },
        { id: "0xzknw/LFM3-VL-2B-EXL3-4bpw", label: "4 bpw", revision: "112c47cddfa6929d9c266bc45ed0154ed7665fab" },
      ] },
    { id: "0xzknw/LFM2.5-24B-A2B-EXL3-4bpw", title: "LFM2.5", parameters: "24B · A2B", quantization: "4 bpw · v2",
      icon: "wave", revision: "649b501fcc29910453eea313a763b7f7a12f4b41", variant: null },
  ] },
  { family: "Qwen", models: [
    { id: "UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw", title: "Qwen3.6", parameters: "35B · A3B", quantization: "2.49 bpw",
      icon: "sparkles", revision: "91a4efc46c1fece6426ece167f127d29d075f392", variant: null },
    { id: "turboderp/Qwen3.8-27B-exl3", title: "Qwen3.8", parameters: "27B", quantization: "2.00 bpw",
      icon: "stack", revision: "e5e1f4b37c77641e465e24adf5f44d01bdd48de2", variant: null },
    { id: "turboderp/Qwen3.8-Flash-Next-exl3", title: "Qwen3.8", parameters: "Flash Next", quantization: "2.05 bpw",
      icon: "bolt", revision: "65c895314393431c09050b2e04e250836b3a6eb4", variant: null },
  ] },
];

/** Bitrate siblings of a recommended repository (empty when it has none). */
export function recommendedChoices(repo) {
  for (const family of RECOMMENDED) {
    for (const model of family.models) {
      if (model.choices?.some((c) => c.id === repo)) return model.choices;
    }
  }
  return [];
}

/** The variant to preselect: the preferred id when present, else the first one. */
export function pickVariant(variants, preferred) {
  if (!Array.isArray(variants) || !variants.length) return null;
  if (typeof preferred === "string") {
    const match = variants.find((v) => v.id === preferred);
    if (match) return match;
  }
  return variants[0];
}

/** "Revision 1a2b3c4d" for a pinned commit, the name otherwise. */
export function revisionLabel(revision, t) {
  return /^[0-9a-f]{40}$/.test(revision) ? `${t("revision")} ${revision.slice(0, 8)}` : revision;
}
