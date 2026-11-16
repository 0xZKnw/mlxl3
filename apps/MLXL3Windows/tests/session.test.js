import assert from "node:assert/strict";
import test from "node:test";
import {
  abortTurn, applyEngineEvent, engineExited, engineStarted, initialSession, newConversation,
  restoreTurns, serializeTurns, startTurn, unloaded,
} from "../src/session.js";

const ready = () => applyEngineEvent(engineStarted(initialSession(), 1), 1, { type: "ready", bridge_protocol: 1, device: "GPU" });

test("streams a full answer", () => {
  let s = startTurn(ready(), "r1", "hi");
  s = applyEngineEvent(s, 1, { type: "generation_status", request_id: "r1", phase: "prefill" });
  assert.equal(s.phase, "prefill");
  s = applyEngineEvent(s, 1, { type: "delta", request_id: "r1", text: "Hel" });
  s = applyEngineEvent(s, 1, { type: "delta", request_id: "r1", text: "lo" });
  s = applyEngineEvent(s, 1, { type: "done", request_id: "r1", generated_tokens: 2 });
  assert.equal(s.active, null);
  assert.deepEqual(s.turns.map((t) => [t.role, t.content, t.complete]), [["user", "hi", true], ["assistant", "Hello", true]]);
  assert.equal(s.lastStats.generated_tokens, 2);
});

test("ignores stale epochs and other requests", () => {
  let s = startTurn(ready(), "r1", "hi");
  const before = s;
  for (const [epoch, event] of [
    [0, { type: "delta", request_id: "r1", text: "x" }],
    [1, { type: "delta", request_id: "other", text: "x" }],
    [1, { type: "done", request_id: "other" }],
    [1, { type: "cancelled", request_id: "other" }],
    [1, { type: "error", request_id: "other", message: "m" }],
    [1, { type: "unknown" }],
    [1, null],
    [1, { type: 3 }],
  ]) {
    s = applyEngineEvent(s, epoch, event);
    assert.deepEqual(s.turns, before.turns);
    assert.equal(s.active, "r1");
  }
  assert.equal(s.error, "m", "untargeted errors are still shown");
  assert.equal(engineExited(s, 0, "old"), s);
});

test("cancel keeps partial text, empty cancel is not sent back", () => {
  let s = startTurn(ready(), "r1", "hi");
  s = applyEngineEvent(s, 1, { type: "delta", request_id: "r1", text: "par" });
  s = applyEngineEvent(s, 1, { type: "cancelled", request_id: "r1" });
  assert.equal(s.turns[1].complete, true);
  let e = startTurn(s, "r2", "again");
  e = applyEngineEvent(e, 1, { type: "cancelled", request_id: "r2" });
  assert.equal(e.turns[3].complete, false);
});

test("errors fail the active turn; fatal and exit stop the engine", () => {
  let s = startTurn(ready(), "r1", "hi");
  const failed = applyEngineEvent(s, 1, { type: "error", request_id: "r1", message: "boom" });
  assert.equal(failed.turns[1].failed, "boom");
  assert.equal(failed.engine, "ready");
  assert.equal(failed.active, null);
  const fatal = applyEngineEvent(s, 1, { type: "error", request_id: "", fatal: true, message: "oom" });
  assert.equal(fatal.engine, "failed");
  assert.equal(fatal.turns[1].failed, "oom");
  assert.equal(engineExited(fatal, 1, "exit 1").error, "oom", "keep the fatal cause");
  const exited = engineExited(s, 1, "exit 3");
  assert.equal(exited.engine, "stopped");
  assert.equal(exited.turns[1].failed, "exit 3");
  assert.throws(() => startTurn(exited, "r2", "x"));
});

test("start preconditions and aborts", () => {
  assert.throws(() => startTurn(initialSession(), "r", "x"));
  assert.throws(() => startTurn(ready(), "r", "   "));
  const s = startTurn(ready(), "r1", "x");
  assert.throws(() => startTurn(s, "r2", "y"));
  assert.throws(() => newConversation(s));
  const aborted = abortTurn(s, "r1", "no model");
  assert.equal(aborted.active, null);
  assert.equal(aborted.turns[1].failed, "no model");
  assert.equal(abortTurn(aborted, "zz", "x"), aborted);
  assert.equal(newConversation(aborted).turns.length, 0);
  const u = unloaded(aborted);
  assert.equal(u.engine, "idle");
  assert.equal(applyEngineEvent(u, 1, { type: "ready" }).engine, "idle", "events of the unloaded engine are stale");
});

test("persistence keeps only complete turns and rejects garbage", () => {
  const s = startTurn(ready(), "r1", "hi");
  assert.deepEqual(restoreTurns(serializeTurns(s.turns)), [{ role: "user", content: "hi", complete: true }]);
  for (const raw of ["", "{", "null", "{}", '[{"role":"system","content":"x"},{"role":"user"},3]']) {
    assert.deepEqual(restoreTurns(raw), [], raw);
  }
});

test("events may arrive before engine-started; late notices do not reset", () => {
  let s = applyEngineEvent(initialSession(), 4, { type: "ready", bridge_protocol: 1 });
  assert.equal(s.epoch, 4);
  assert.equal(s.engine, "ready");
  assert.equal(engineStarted(s, 4).engine, "ready", "late engine-started keeps the ready engine");
  assert.equal(engineStarted(s, 3), s);
  const failedFast = engineExited(initialSession(), 2, "exit 1");
  assert.equal(failedFast.engine, "stopped");
  assert.equal(engineStarted(failedFast, 2).engine, "stopped");
  assert.equal(engineStarted(unloaded(s), 5).engine, "loading");
});

test("rewind drops a user turn and what follows, and refuses while generating", async () => {
  const { rewind } = await import("../src/session.js");
  const base = { ...initialSession(), engine: "ready", turns: [
    { role: "user", content: "one", complete: true }, { role: "assistant", content: "a", complete: true },
    { role: "user", content: "two", complete: true }, { role: "assistant", content: "b", complete: true },
  ] };
  const { state, prompt } = rewind(base, 2);
  assert.equal(prompt, "two");
  assert.equal(state.turns.length, 2);
  assert.equal(base.turns.length, 4);
  assert.throws(() => rewind(base, 1), /not a user message/);
  assert.throws(() => rewind(base, 9), /not a user message/);
  assert.throws(() => rewind({ ...base, active: "r1" }, 0), /stop the current answer/);
  assert.equal(rewind(base, 0).state.turns.length, 0);
});
