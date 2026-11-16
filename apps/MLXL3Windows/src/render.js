// Pure helpers shared by the UI and its Node tests. No DOM access here.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

export const SOURCE_SCHEME = "mlxl3-source://";

// Links on already-escaped text. Citations become numbered chips when the id is one of the
// message's sources (an invented id stays plain text); http(s) links open in the browser
// through the shell, never inside the WebView. Other schemes stay text.
function links(escaped, sources) {
  return escaped.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (whole, label, target) => {
    if (target.startsWith(SOURCE_SCHEME)) {
      const id = target.slice(SOURCE_SCHEME.length);
      // Mac: the source host is compared case-insensitively.
      const index = sources.findIndex((source) => escapeHtml(source.id).toLowerCase() === id.toLowerCase());
      if (index === -1) return label;
      return `<button class="cite" data-source="${escapeHtml(sources[index].id)}" title="${escapeHtml(sources[index].label ?? "")}">${index + 1}</button>`;
    }
    if (/^https?:\/\//i.test(target)) return `<a href="#" class="link" data-url="${target}">${label}</a>`;
    return whole;
  });
}

function inline(text, sources = []) {
  // Code spans first so their content is not formatted; everything is escaped.
  return text.split(/(`[^`\n]+`)/g).map((part) => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
    }
    return links(escapeHtml(part), sources)
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  }).join("");
}

const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
const isDivider = (line) => /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(line);

/**
 * Minimal, safe Markdown: fenced code, headings, lists, quotes, GFM tables, paragraphs and
 * links (see `links`). `sources` are the message's document excerpts, for citations.
 */
export function renderMarkdown(source, sources = []) {
  const inline_ = (text) => inline(text, sources);
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  let list = null;
  const flushParagraph = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map(inline_).join("<br>")}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline_(i)}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
    if (fence) {
      flushParagraph(); flushList();
      const body = [];
      for (i++; i < lines.length && !/^\s*```\s*$/.test(lines[i]); i++) body.push(lines[i]);
      const lang = fence[1] ? ` data-lang="${escapeHtml(fence[1])}"` : "";
      out.push(`<pre${lang}><code>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isDivider(lines[i + 1]) && cells(line).length === cells(lines[i + 1]).length) {
      flushParagraph(); flushList();
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i++) rows.push(cells(lines[i]));
      i--;
      const row = (values, tag) => `<tr>${head.map((_, c) => `<${tag}>${inline_(values[c] ?? "")}</${tag}>`).join("")}</tr>`;
      out.push(`<div class="table-wrap"><table><thead>${row(head, "th")}</thead><tbody>${rows.map((r) => row(r, "td")).join("")}</tbody></table></div>`);
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const quote = line.match(/^>\s?(.*)$/);
    if (heading) {
      flushParagraph(); flushList();
      const level = heading[1].length + 1;
      out.push(`<h${level}>${inline_(heading[2])}</h${level}>`);
    } else if (bullet || ordered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      if (list && list.tag !== tag) flushList();
      list = list || { tag, items: [] };
      list.items.push((bullet || ordered)[1]);
    } else if (quote) {
      flushParagraph(); flushList();
      out.push(`<blockquote>${inline_(quote[1])}</blockquote>`);
    } else if (!line.trim()) {
      flushParagraph(); flushList();
    } else {
      flushList();
      paragraph.push(line);
    }
  }
  flushParagraph(); flushList();
  return out.join("");
}

/** Answer of a raw engine output, whose `<think>` may have been opened by the template. */
function rawAnswer(text) {
  const close = text.lastIndexOf("</think>");
  if (close !== -1 && !text.includes("<think>")) return text.slice(close + 8).trim();
  return splitThinking(text).answer.trim();
}

/** Split reasoning (<think>…</think>) from the answer; an open block is still streaming. */
export function splitThinking(text) {
  const start = text.indexOf("<think>");
  if (start === -1) return { thinking: null, answer: text, open: false };
  const end = text.indexOf("</think>", start);
  if (end === -1) {
    return { thinking: text.slice(start + 7), answer: text.slice(0, start), open: true };
  }
  return {
    thinking: text.slice(start + 7, end),
    answer: text.slice(0, start) + text.slice(end + 8),
    open: false,
  };
}

export function formatStats(done) {
  if (!done) return "";
  const parts = [];
  if (typeof done.decode_tokens_per_second === "number") parts.push(`${done.decode_tokens_per_second.toFixed(1)} tok/s`);
  if (typeof done.generated_tokens === "number") parts.push(`${done.generated_tokens} tokens`);
  if (typeof done.prefill_seconds === "number" && done.prompt_tokens) {
    parts.push(`prefill ${done.prompt_tokens} tok in ${done.prefill_seconds.toFixed(2)} s`);
  }
  if (done.cached_tokens) parts.push(`${done.cached_tokens} cached`);
  return parts.join(" · ");
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit++; }
  return `${value.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

/**
 * Conversation messages for the engine. A user turn whose answer failed or was
 * cancelled before any text is dropped together with that answer, so templates
 * that require strict user/assistant alternation still render.
 */
export function engineMessages(system, turns) {
  const messages = [];
  if (system && system.trim()) messages.push({ role: "system", content: system.trim() });
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    const next = turns[i + 1];
    const answered = (t) => t && t.role === "assistant" && t.complete && t.content;
    if (turn.role === "assistant") {
      if (!answered(turn)) continue;
      // Tool rounds are replayed (calls, results, final answer) so later turns keep the results.
      if (Array.isArray(turn.turnContext) && turn.turnContext.length) {
        for (const step of turn.turnContext) {
          const entry = { role: step.role, content: step.role === "assistant" ? rawAnswer(step.content) : step.content };
          if (step.tool_calls?.length) entry.tool_calls = step.tool_calls;
          if (step.name) entry.name = step.name;
          messages.push(entry);
        }
      } else {
        messages.push({ role: "assistant", content: splitThinking(turn.content).answer.trim() });
      }
      continue;
    }
    if (next && next.role === "assistant" && !answered(next)) continue;
    // `context` is what the model saw: the text plus attachments and document excerpts.
    const entry = { role: turn.role, content: turn.context ?? turn.content };
    if (Array.isArray(turn.images) && turn.images.length) entry.images = turn.images.map((image) => image.data);
    messages.push(entry);
  }
  return messages;
}
