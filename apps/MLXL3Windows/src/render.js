// Pure helpers shared by the UI and its Node tests. No DOM access here.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

function inline(text) {
  // Code spans first so their content is not formatted; everything is escaped.
  return text.split(/(`[^`\n]+`)/g).map((part) => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
    }
    return escapeHtml(part)
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  }).join("");
}

/** Minimal, safe Markdown: fenced code, headings, lists, quotes, paragraphs. Links stay text. */
export function renderMarkdown(source) {
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let paragraph = [];
  let list = null;
  const flushParagraph = () => {
    if (paragraph.length) out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
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
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const quote = line.match(/^>\s?(.*)$/);
    if (heading) {
      flushParagraph(); flushList();
      const level = heading[1].length + 1;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (bullet || ordered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      if (list && list.tag !== tag) flushList();
      list = list || { tag, items: [] };
      list.items.push((bullet || ordered)[1]);
    } else if (quote) {
      flushParagraph(); flushList();
      out.push(`<blockquote>${inline(quote[1])}</blockquote>`);
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
      if (answered(turn)) messages.push({ role: "assistant", content: splitThinking(turn.content).answer.trim() });
      continue;
    }
    if (next && next.role === "assistant" && !answered(next)) continue;
    messages.push({ role: turn.role, content: turn.content });
  }
  return messages;
}
