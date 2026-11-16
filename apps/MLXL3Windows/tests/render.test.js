import assert from "node:assert/strict";
import test from "node:test";
import { engineMessages, escapeHtml, formatBytes, formatStats, renderMarkdown, splitThinking } from "../src/render.js";

test("escapes every HTML metacharacter", () => {
  assert.equal(escapeHtml(`<a href="x" onclick='y'>&</a>`), "&lt;a href=&quot;x&quot; onclick=&#39;y&#39;&gt;&amp;&lt;/a&gt;");
});

test("markdown never emits raw HTML from the source", () => {
  const hostile = [
    "<img src=x onerror=alert(1)>", "**<script>x</script>**", "`<b>`", "```\n</code></pre><script>\n```",
    "# <i>h</i>", "- <svg onload=1>", "> <iframe>", "[x](javascript:alert(1))", "```js\"><script>\nx\n```",
  ];
  for (const source of hostile) {
    const html = renderMarkdown(source);
    assert.doesNotMatch(html, /<(?!\/?(p|br|strong|em|code|pre|h[2-5]|ul|ol|li|blockquote)[ >])/, source);
    assert.doesNotMatch(html, /<pre data-lang="[^"]*"[^>]*>.*<script/s, source);
  }
});

test("markdown structure", () => {
  assert.equal(renderMarkdown("a\nb\n\nc"), "<p>a<br>b</p><p>c</p>");
  assert.equal(renderMarkdown("## T\n- x\n- **y**\n1. z"), "<h3>T</h3><ul><li>x</li><li><strong>y</strong></li></ul><ol><li>z</li></ol>");
  assert.equal(renderMarkdown("```py\nprint(1)\n```"), '<pre data-lang="py"><code>print(1)</code></pre>');
  assert.equal(renderMarkdown("```\nunterminated"), "<pre><code>unterminated</code></pre>");
  assert.equal(renderMarkdown("use `a*b*c` and *it*"), "<p>use <code>a*b*c</code> and <em>it</em></p>");
});

test("thinking blocks", () => {
  assert.deepEqual(splitThinking("plain"), { thinking: null, answer: "plain", open: false });
  assert.deepEqual(splitThinking("<think>r"), { thinking: "r", answer: "", open: true });
  assert.deepEqual(splitThinking("<think>r</think>\nA"), { thinking: "r", answer: "\nA", open: false });
});

test("engine messages skip incomplete assistant turns and strip reasoning", () => {
  const turns = [
    { role: "user", content: "q1" },
    { role: "assistant", content: "<think>r</think>a1", complete: true },
    { role: "user", content: "q2" },
    { role: "assistant", content: "partial", complete: false },
    { role: "user", content: "q3" },
  ];
  assert.deepEqual(engineMessages(" sys ", turns), [
    { role: "system", content: "sys" }, { role: "user", content: "q1" }, { role: "assistant", content: "a1" },
    { role: "user", content: "q3" },
  ]);
  assert.deepEqual(engineMessages("", [{ role: "user", content: "x" }]), [{ role: "user", content: "x" }]);
});

test("formatting", () => {
  assert.equal(formatStats({ decode_tokens_per_second: 41.26, generated_tokens: 12, prompt_tokens: 30, prefill_seconds: 0.5, cached_tokens: 0 }),
    "41.3 tok/s · 12 tokens · prefill 30 tok in 0.50 s");
  assert.equal(formatStats({ decode_tokens_per_second: null, generated_tokens: 0 }), "0 tokens");
  assert.equal(formatBytes(4.2e9), "4.2 GB");
  assert.equal(formatBytes(0), "");
});

import { KEYS, LANGUAGES, pickLanguage, table, translator } from "../src/i18n.js";

test("every language defines every key", () => {
  for (const language of LANGUAGES) assert.deepEqual(Object.keys(table(language)).sort(), [...KEYS].sort(), language);
});

test("language selection falls back to English", () => {
  assert.equal(pickLanguage(["fr-FR", "en"]), "fr");
  assert.equal(pickLanguage(["de-DE", "en-US"]), "en");
  assert.equal(pickLanguage([]), "en");
  assert.equal(translator("fr")("send"), "Envoyer");
  assert.equal(translator("xx")("missing-key"), "missing-key");
});
