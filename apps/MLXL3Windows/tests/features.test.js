// Tool activity, tool-round replay, project/agent persistence, exports and Lab samples.
import assert from "node:assert/strict";
import test from "node:test";
import { addSample, chartPoints, exportJson, history, labSample, latestByModel, MAX_SAMPLES, restoreSamples } from "../src/lab.js";
import { engineMessages, renderMarkdown } from "../src/render.js";
import { applyEngineEvent, cleanTurnContext, engineStarted, initialSession, restoreTurns, serializeTurns, startTurn } from "../src/session.js";
import { markdownExport, restoreConversations, serializeConversations } from "../src/store.js";

const ready = () => applyEngineEvent(engineStarted(initialSession(), 1), 1, { type: "ready", bridge_protocol: 1 });
const CALL = { type: "function", function: { name: "exa.search", arguments: { q: "x" } } };
const CONTEXT = [
  { role: "assistant", content: "", tool_calls: [CALL] },
  { role: "tool", content: "result text", name: "exa.search" },
  { role: "assistant", content: "<think>plan</think>Final [s](mlxl3-source://Doc-1)" },
];

test("tool rows follow tool_start / tool_result and survive persistence", () => {
  let s = startTurn(ready(), "r1", "find");
  s = applyEngineEvent(s, 1, { type: "tool_start", request_id: "r1", tool_call_id: "c1", tool_name: "exa.search", server_name: "exa" });
  assert.equal(s.phase, "tool");
  assert.deepEqual(s.turns[1].tools, [{ id: "c1", name: "exa.search", server: "exa", status: "running" }]);
  s = applyEngineEvent(s, 1, { type: "tool_result", request_id: "r1", tool_call_id: "c1", tool_name: "exa.search", server_name: "exa", text: "ok", is_error: false });
  s = applyEngineEvent(s, 1, { type: "tool_start", request_id: "r1", tool_call_id: "c2", tool_name: "fs.read", server_name: "fs" });
  s = applyEngineEvent(s, 1, { type: "tool_result", request_id: "other", tool_call_id: "c2", text: "ignored" });
  s = applyEngineEvent(s, 1, { type: "delta", request_id: "r1", text: "Final" });
  s = applyEngineEvent(s, 1, { type: "done", request_id: "r1", generated_tokens: 3, tool_rounds: 1, turn_context: CONTEXT });
  const answer = s.turns[1];
  assert.deepEqual(answer.tools.map((x) => [x.name, x.status, x.result]), [["exa.search", "complete", "ok"], ["fs.read", "running", undefined]]);
  assert.equal(answer.turnContext.length, 3);
  const restored = restoreTurns(serializeTurns(s.turns))[1];
  assert.deepEqual(restored.tools.map((x) => x.status), ["complete", "failed"], "a row left running was interrupted");
  assert.equal(restored.turnContext.length, 3);
});

test("a validated JSON answer is flagged", () => {
  let s = startTurn(ready(), "r1", "json");
  s = applyEngineEvent(s, 1, { type: "delta", request_id: "r1", text: "{\"a\":1}" });
  s = applyEngineEvent(s, 1, { type: "done", request_id: "r1", schema: "valid", attempts: 2 });
  assert.equal(s.turns[1].schema, "valid");
  assert.equal(restoreTurns(serializeTurns(s.turns))[1].schema, "valid");
});

test("malformed tool contexts are dropped whole", () => {
  assert.equal(cleanTurnContext(CONTEXT).length, 3);
  for (const bad of [
    null, [{ role: "user", content: "x" }], [{ role: "tool", content: 3 }],
    [{ role: "assistant", content: "", tool_calls: [{ function: {} }] }, { role: "assistant", content: "x" }],
    [{ role: "tool", content: "r", name: "a" }], // must end with the answer
    Array.from({ length: 25 }, () => ({ role: "assistant", content: "x" })),
    [{ role: "assistant", content: "x".repeat(70 * 1024) }],
  ]) assert.deepEqual(cleanTurnContext(bad), [], JSON.stringify(bad)?.slice(0, 60));
});

test("later turns replay the tool rounds instead of the visible answer", () => {
  const turns = [
    { role: "user", content: "find", complete: true },
    { role: "assistant", content: "Final", complete: true, turnContext: CONTEXT },
    { role: "user", content: "and?", complete: true },
  ];
  assert.deepEqual(engineMessages("S", turns), [
    { role: "system", content: "S" },
    { role: "user", content: "find" },
    { role: "assistant", content: "", tool_calls: [CALL] },
    { role: "tool", content: "result text", name: "exa.search" },
    { role: "assistant", content: "Final [s](mlxl3-source://Doc-1)" },
    { role: "user", content: "and?" },
  ]);
  const opened = [{ role: "user", content: "q", complete: true },
    { role: "assistant", content: "A", complete: true, turnContext: [{ role: "assistant", content: "hidden</think>A" }] }];
  assert.equal(engineMessages("", opened).at(-1).content, "A", "a template-opened reasoning block is stripped");
});

test("citations match source ids case-insensitively and only known ids", () => {
  const sources = [{ id: "Doc-1", label: "a.pdf · p. 2" }];
  assert.match(renderMarkdown("x [1](mlxl3-source://doc-1)", sources), /<button class="cite" data-source="Doc-1"/);
  assert.doesNotMatch(renderMarkdown("x [1](mlxl3-source://nope)", sources), /cite/);
});

test("conversations keep their project and agent flags", () => {
  const list = [
    { id: "a1", title: "A", updated: 2, pinned: false, turns: [], projectId: "p-1", agent: true },
    { id: "b2", title: "B", updated: 1, pinned: false, turns: [] },
    { id: "c3", title: "C", updated: 0, pinned: false, turns: [], projectId: "../x" },
  ];
  const restored = restoreConversations(serializeConversations(list));
  assert.deepEqual(restored.map((c) => [c.id, c.projectId, c.agent]), [["a1", "p-1", true], ["b2", null, false], ["c3", null, false]]);
});

test("markdown export lists attachments and quoted sources", () => {
  const text = markdownExport({ title: "T", turns: [
    { role: "user", content: "q", complete: true, attachments: [{ name: "a.pdf", kind: "text" }] },
    { role: "assistant", content: "A", complete: true, sources: [{ id: "d-1", label: "a.pdf · p. 2", text: "line1\nline2" }] },
  ] });
  assert.match(text, /\*\*Attachment:\*\* a\.pdf/);
  assert.match(text, /### Sources\n\n- \[d-1\] a\.pdf · p\. 2\n> line1\n> line2/);
});

test("lab samples follow the Mac validity rule and metrics", () => {
  const meta = { model: "Qwen3", mode: "chat", bits: 4, context: 8192 };
  const done = { prompt_tokens: 100, cached_tokens: 20, generated_tokens: 50, prefill_seconds: 0.5, decode_seconds: 2, wall_seconds: 2.6 };
  const s = labSample(done, meta, new Date("2026-11-16T10:00:00Z"));
  assert.equal(s.decodeTps, 25);
  assert.equal(s.prefillTps, 160);
  assert.equal(s.ttftSeconds, 0.5);
  assert.equal(s.elapsedSeconds, 2.6);
  for (const bad of [{ ...done, generated_tokens: 0 }, { ...done, prefill_seconds: NaN }, { ...done, decode_seconds: -1 }, { ...done, prompt_tokens: undefined }]) {
    assert.equal(labSample(bad, meta), null);
  }
  assert.equal(labSample(done, { model: "" }), null);
  let samples = [];
  for (let i = 0; i < MAX_SAMPLES + 5; i++) samples = addSample(samples, { ...s, id: String(i) });
  assert.equal(samples.length, MAX_SAMPLES);
  assert.equal(samples[0].id, "5");
  assert.equal(addSample(samples, null), samples);
});

test("lab views: latest per model, history, chart and export", () => {
  const base = { promptTokens: 1, generatedTokens: 1, cachedTokens: 0, ttftSeconds: 0.1, prefillTps: 1, elapsedSeconds: 1 };
  const samples = [
    { ...base, model: "A", date: "2026-01-01T00:00:00Z", decodeTps: 10 },
    { ...base, model: "B", date: "2026-01-02T00:00:00Z", decodeTps: 40 },
    { ...base, model: "A", date: "2026-01-03T00:00:00Z", decodeTps: 20 },
  ];
  assert.deepEqual(latestByModel(samples), [{ model: "B", value: 40, percent: 100 }, { model: "A", value: 20, percent: 50 }]);
  assert.deepEqual(history(samples, "A", "decode").map((p) => p.value), [10, 20]);
  assert.deepEqual(history(samples, "", "nope"), []);
  const points = chartPoints(history(samples, "", "decode"), 200, 100, 10);
  assert.deepEqual(points.map((p) => [Math.round(p.x), Math.round(p.y)]), [[10, 70], [100, 10], [190, 50]]);
  assert.deepEqual(restoreSamples(exportJson(samples)).length, 3);
  assert.deepEqual(restoreSamples("[{\"model\":\"x\"}]"), []);
  assert.deepEqual(restoreSamples("nope"), []);
  assert.match(exportJson([{ z: 1, a: 2 }]), /^\[\n {2}\{\n {4}"a": 2,\n {4}"z": 1/);
});
