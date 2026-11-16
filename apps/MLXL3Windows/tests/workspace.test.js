import assert from "node:assert/strict";
import { test } from "node:test";
import {
  admitAttachments, attachmentContext, ATTACHMENT_LIMITS, composeSystem, DOCUMENT_INSTRUCTION, documentContext,
  normalizeWorkspace, profileSampling,
} from "../src/workspace.js";
import { engineMessages, renderMarkdown } from "../src/render.js";
import { restoreTurns, serializeTurns, startTurn } from "../src/session.js";

const source = { id: "d1-0", document_id: "d1", file_name: "guide.pdf", original_path: "C:/g.pdf", page: 2, text: "Alpha", label: "guide.pdf · p. 2" };

test("document excerpts use the Mac context block", () => {
  assert.equal(documentContext([]), "");
  assert.equal(documentContext([source, { ...source, id: "d1-1", label: "guide.pdf", text: "Beta" }]),
    "\n\n<document_excerpts>\n[d1-0] guide.pdf · p. 2\nAlpha\n\n[d1-1] guide.pdf\nBeta\n</document_excerpts>");
});

test("attached text files follow the message, images are not inlined", () => {
  const files = [{ kind: "text", file_name: "a.md", text: "# A" }, { kind: "image", file_name: "p.png" }];
  assert.equal(attachmentContext("Read", files),
    'Read\n\nAttached documents (reference material):\n\n--- File: "a.md" ---\n# A\n--- End of file ---');
  assert.equal(attachmentContext("Hi", [files[1]]), "Hi");
});

test("attachments respect the count, per-file and total limits", () => {
  const text = (name, size) => ({ kind: "text", file_name: name, path: name, text: "x".repeat(size) });
  const image = (name, bytes) => ({ kind: "image", file_name: name, path: name, file_bytes: bytes });
  let { accepted, rejected } = admitAttachments([], [text("a", 10), { kind: "error", file_name: "b", error: "bad" }, text("a", 10)]);
  assert.deepEqual(accepted.map((f) => f.file_name), ["a"]);
  assert.deepEqual(rejected, [{ name: "b", reason: "bad" }]);
  ({ rejected } = admitAttachments([], [text("big", ATTACHMENT_LIMITS.textPerFile + 1)]));
  assert.equal(rejected[0].reason, "attachTextTooLarge");
  ({ accepted, rejected } = admitAttachments([], [text("h1", 200 * 1024), text("h2", 200 * 1024), text("h3", 200 * 1024)]));
  assert.deepEqual([accepted.length, rejected[0].reason], [2, "attachTotalTooLarge"]);
  ({ accepted, rejected } = admitAttachments([], [image("i1", 5e6), image("i2", 5e6)]));
  assert.deepEqual([accepted.length, rejected[0].reason], [1, "attachImagesTooLarge"]);
  const many = Array.from({ length: 9 }, (_, i) => text(`f${i}`, 1));
  ({ accepted, rejected } = admitAttachments([], many));
  assert.deepEqual([accepted.length, rejected.length, rejected[0].reason], [8, 1, "attachTooMany"]);
});

test("workspace normalisation drops bad ids, duplicates and dangling profiles", () => {
  const ws = normalizeWorkspace({
    profiles: [{ id: "p1", name: " Precise ", temperature: 9, topK: 3.7, repetitionPenalty: 1.1, contextLength: 4096, systemPrompt: "Be exact" },
      { id: "p1", name: "dup" }, { id: "../x", name: "bad" }],
    projects: [{ id: "j1", name: "Thesis", profileId: "p1", instructions: "Use French" },
      { id: "j2", name: "Other", profileId: "gone" }, { id: "j3", name: "  " }],
  });
  assert.deepEqual(ws.profiles.map((p) => [p.id, p.name, p.temperature, p.topK, p.topP]), [["p1", "Precise", 5, 3, null]]);
  assert.deepEqual(ws.projects.map((p) => [p.id, p.profileId]), [["j1", "p1"], ["j2", null]]);
  assert.deepEqual(normalizeWorkspace("junk"), { projects: [], profiles: [] });
});

test("system prompt stacks profile, project and citation rules", () => {
  const profile = { systemPrompt: "Be exact" };
  const project = { instructions: "Use French" };
  assert.equal(composeSystem({ base: "Base", project, profile, hasDocuments: true }), `Be exact\n\nUse French\n\n${DOCUMENT_INSTRUCTION}`);
  assert.equal(composeSystem({ base: " Base ", project: null, profile: null, hasDocuments: false }), "Base");
  assert.equal(composeSystem({ base: "", project: { instructions: " " }, profile: { systemPrompt: "" }, hasDocuments: false }), "");
  const sampling = { temperature: 0.6, top_k: 20, top_p: 0.95, repetition_penalty: 1 };
  assert.deepEqual(profileSampling(sampling, { temperature: 0.2, topK: 5, topP: null, repetitionPenalty: 1.1 }),
    { temperature: 0.2, top_k: 5, top_p: 0.95, repetition_penalty: 1.1 });
  assert.equal(profileSampling(sampling, null), sampling);
});

test("citations become chips only for known sources; links never navigate the WebView", () => {
  const html = renderMarkdown("Fact [source](mlxl3-source://d1-0), fake [source](mlxl3-source://zz-9), [site](https://e.com/?a=1&b=2), [x](javascript:alert(1))", [source]);
  assert.match(html, /<button class="cite" data-source="d1-0" title="guide.pdf · p. 2">1<\/button>/);
  assert.match(html, /fake source,/);
  assert.match(html, /<a href="#" class="link" data-url="https:\/\/e.com\/\?a=1&amp;b=2">site<\/a>/);
  assert.match(html, /\[x\]\(javascript:alert\(1\)\)/);
  assert.doesNotMatch(html, /href="javascript/);
});

test("GFM tables render with escaped cells", () => {
  const html = renderMarkdown("| A | <b> |\n| --- | :-: |\n| 1 | **2** |\n| 3 |\n\nafter");
  assert.equal(html, '<div class="table-wrap"><table><thead><tr><th>A</th><th>&lt;b&gt;</th></tr></thead><tbody><tr><td>1</td><td><strong>2</strong></td></tr><tr><td>3</td><td></td></tr></tbody></table></div><p>after</p>');
  assert.equal(renderMarkdown("a | b\nnot a divider"), "<p>a | b<br>not a divider</p>");
});

test("turns keep attachments, model context and sources across a save, but not image bytes", () => {
  const state = { engine: "ready", active: null, turns: [] };
  const next = startTurn(state, "r1", "Q", {
    context: "Q + files", attachments: [{ name: "a.md", kind: "text" }, { name: "p.png", kind: "image" }],
    images: [{ mime: "image/png", data: "AAAA" }], sources: [source],
  });
  assert.equal(next.turns[0].context, "Q + files");
  assert.deepEqual(engineMessages("", next.turns.slice(0, 1)), [{ role: "user", content: "Q + files", images: ["AAAA"] }]);
  const done = next.turns.map((t) => ({ ...t, complete: true, content: t.content || "A" }));
  const restored = restoreTurns(serializeTurns(done));
  assert.deepEqual(restored[0], { role: "user", content: "Q", complete: true, attachments: [{ name: "a.md", kind: "text" }, { name: "p.png", kind: "image" }], context: "Q + files" });
  assert.deepEqual(restored[1].sources, [source]);
  assert.equal(restored[0].images, undefined);
  assert.throws(() => startTurn(state, "r2", " "), /empty/);
  assert.equal(startTurn(state, "r3", "", { attachments: [{ name: "a", kind: "text" }] }).turns.length, 2);
  // Hostile stored values are dropped.
  const hostile = restoreTurns(JSON.stringify([{ role: "assistant", content: "x", sources: [{ id: "<img>", text: "t" }, { id: "ok-1", text: "t", page: -2 }] }]));
  assert.deepEqual(hostile[0].sources.map((s) => [s.id, s.page]), [["ok-1", null]]);
});
