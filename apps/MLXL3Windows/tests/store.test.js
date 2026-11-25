import assert from "node:assert/strict";
import { test } from "node:test";
import {
  GREETING_POOLS, greeting, greetingPeriod, MAX_CONVERSATIONS, modelSubtitle, pickVariant, recommendedChoices, RECOMMENDED,
  restoreConversations, revisionLabel, sections, serializeConversations, titleFrom, togglePinned, updateBadge, loadConversationHistory,
  upsertConversation, formatCount, percent, splitRepo, backupJson, branchOf, mergeBackup, markdownExport, BACKUP_FORMAT,
} from "../src/store.js";
import { translator } from "../src/i18n.js";

const t = translator("en");
const turn = (role, content) => ({ role, content, complete: true });

test("greeting periods follow the Mac hour ranges", () => {
  const expected = { 0: "night", 4: "night", 5: "morning", 11: "morning", 12: "afternoon", 17: "afternoon", 18: "evening", 22: "evening", 23: "night" };
  for (const [hour, period] of Object.entries(expected)) assert.equal(greetingPeriod(Number(hour)), period, `hour ${hour}`);
});

test("greeting uses the anytime pool when seed % 4 == 3 and indexes by seed / 4", () => {
  assert.equal(greeting(8, 0, "en"), "Good morning");
  assert.equal(greeting(8, 4, "en"), "Awake yet?");
  assert.equal(greeting(8, 3, "en"), "What's on your mind?");
  assert.equal(greeting(8, 7, "en"), "Where to begin?");
  assert.equal(greeting(20, 0, "fr"), "Bonsoir");
  assert.equal(greeting(2, 16, "en"), "Ideas come at night"); // 16/4 = 4 → 4 % 3 = 1
  assert.equal(greeting(8, 0, "xx"), "Good morning");
});

test("every hour and seed gives a non-empty greeting from the right pool", () => {
  for (const language of ["en", "fr"]) {
    const pools = GREETING_POOLS[language];
    assert.equal(Object.keys(pools).length, 5);
    for (let hour = 0; hour < 24; hour++) {
      for (let seed = 0; seed < 64; seed++) {
        const text = greeting(hour, seed, language);
        const pool = seed % 4 === 3 ? pools.anytime : pools[greetingPeriod(hour)];
        assert.ok(pool.includes(text), `${language} ${hour} ${seed}`);
      }
    }
  }
  assert.ok(greeting(8, Number.NaN, "en"));
  assert.ok(greeting(8, -5, "en"));
});

test("recommended catalog mirrors the Mac list with pinned commits", () => {
  assert.deepEqual(RECOMMENDED.map((f) => f.family), ["LFM", "Qwen"]);
  const all = RECOMMENDED.flatMap((f) => f.models);
  assert.equal(all.length, 7);
  const ids = new Set();
  for (const model of all) {
    assert.match(model.id, /^[\w.-]+\/[\w.-]+$/);
    assert.match(model.revision, /^[0-9a-f]{40}$/);
    assert.ok(!ids.has(model.id), model.id);
    ids.add(model.id);
    for (const choice of model.choices ?? []) assert.match(choice.revision, /^[0-9a-f]{40}$/);
  }
  assert.deepEqual(recommendedChoices("0xzknw/LFM3-1.7B-EXL3-4bpw").map((c) => c.label), ["8 bpw", "4.49 bpw", "4 bpw"]);
  assert.deepEqual(recommendedChoices("turboderp/Qwen3.8-27B-exl3"), []);
  assert.deepEqual(recommendedChoices("someone/else"), []);
});

test("pickVariant prefers the requested id, including the root variant", () => {
  const variants = [{ id: "4.0bpw" }, { id: "" }, { id: "8.0bpw" }];
  assert.equal(pickVariant(variants, "").id, "");
  assert.equal(pickVariant(variants, "8.0bpw").id, "8.0bpw");
  assert.equal(pickVariant(variants, "missing").id, "4.0bpw");
  assert.equal(pickVariant(variants, undefined).id, "4.0bpw");
  assert.equal(pickVariant([], ""), null);
  assert.equal(pickVariant(undefined, ""), null);
});

test("revision labels shorten pinned commits only", () => {
  assert.equal(revisionLabel("19aa14c461813d7a5abc462bdf2af2d90f7b771f", t), "Revision 19aa14c4");
  assert.equal(revisionLabel("main", t), "main");
  assert.equal(revisionLabel("4.0bpw", t), "4.0bpw");
});

test("pinning keeps order and survives a save/restore round trip", () => {
  let list = [];
  list = upsertConversation(list, { id: "a1", title: "A", updated: 1, turns: [turn("user", "a")] });
  list = upsertConversation(list, { id: "b2", title: "B", updated: 2, turns: [turn("user", "b")] });
  list = togglePinned(list, "a1");
  assert.deepEqual(list.map((c) => c.id), ["b2", "a1"]);
  const { pinned, recent } = sections(list);
  assert.deepEqual(pinned.map((c) => c.id), ["a1"]);
  assert.deepEqual(recent.map((c) => c.id), ["b2"]);
  const restored = restoreConversations(serializeConversations(list));
  assert.deepEqual(restored.map((c) => [c.id, c.pinned]), [["b2", false], ["a1", true]]);
  assert.deepEqual(togglePinned(togglePinned(list, "a1"), "a1"), list);
});

test("restore drops malformed rows, duplicates and non-boolean pins", () => {
  const raw = JSON.stringify([
    { id: "ok1", title: "x", updated: 3, pinned: "yes", turns: [turn("user", "hi")] },
    { id: "ok1", title: "dup", updated: 2, turns: [] },
    { id: "BAD!", title: "x", updated: 1, turns: [] },
    { id: "nt", title: 5, updated: 1, turns: [] },
    { id: "nu", title: "x", updated: "1", turns: [] },
    null,
  ]);
  const restored = restoreConversations(raw);
  assert.equal(restored.length, 1);
  assert.equal(restored[0].pinned, false);
  assert.deepEqual(restoreConversations("not json"), []);
  assert.deepEqual(restoreConversations("{}"), []);
});

test("conversation list is capped", () => {
  let list = [];
  for (let i = 0; i < MAX_CONVERSATIONS + 5; i++) list = upsertConversation(list, { id: `c${i}`, title: "t", updated: i, turns: [] });
  assert.equal(list.length, MAX_CONVERSATIONS);
  assert.equal(list[0].id, `c${MAX_CONVERSATIONS + 4}`);
});

test("titles, labels and counters", () => {
  assert.equal(titleFrom([], "New"), "New");
  assert.equal(titleFrom([turn("user", "  hello\nworld")], "New"), "hello");
  assert.equal(titleFrom([turn("user", "x".repeat(80))], "New").length, 60);
  assert.equal(modelSubtitle({ architecture: "Lfm2ForCausalLM", bits: 4 }), "Lfm2 · EXL3 · 4.00 BPW");
  assert.equal(modelSubtitle({}), "EXL3");
  assert.equal(formatCount(1234567), "1.2M");
  assert.equal(formatCount(12000), "12k");
  assert.equal(formatCount(-1), "0");
  assert.deepEqual(splitRepo("owner/name"), { owner: "owner", name: "name" });
  assert.equal(percent(5, 10), 50);
  assert.equal(percent(20, 10), 100);
  assert.equal(percent(1, 0), 0);
});

test("update badge only shows for downloading, ready and failed updates", () => {
  const update = { version: "2.0.4" };
  assert.equal(updateBadge(null, t), null);
  assert.equal(updateBadge({ phase: "current" }, t), null);
  assert.equal(updateBadge({ phase: "checking", update }, t), null);
  assert.deepEqual(updateBadge({ phase: "ready", update }, t), { kind: "ready", label: "Update 2.0.4 ready" });
  assert.equal(updateBadge({ phase: "downloading", update, done: 1, total: 4 }, t).label, "Downloading 2.0.4 25%");
  assert.equal(updateBadge({ phase: "failed", update }, t).kind, "failed");
});

test("backup round-trips and import only adds unknown conversations", () => {
  const a = { id: "a1", title: "A", updated: 10, pinned: true, turns: [turn("user", "hi"), turn("assistant", "yo")] };
  const b = { id: "b2", title: "B", updated: 20, pinned: false, turns: [turn("user", "q")] };
  const json = backupJson([a, b], new Date(0));
  const parsed = JSON.parse(json);
  assert.equal(parsed.format, BACKUP_FORMAT);
  assert.equal(parsed.exported, "1970-01-01T00:00:00.000Z");
  // Same-id local conversation wins: never overwritten by the backup.
  const local = [{ ...a, title: "Local A", turns: [turn("user", "mine")] }];
  const { list, added } = mergeBackup(local, json);
  assert.equal(added, 1);
  assert.deepEqual(list.map((c) => c.id), ["b2", "a1"]);
  assert.equal(list.find((c) => c.id === "a1").title, "Local A");
  assert.equal(mergeBackup(list, json).added, 0, "importing twice adds nothing");
  // A bare array (older export) is accepted too; empty or malformed entries are dropped.
  const bare = JSON.stringify([{ id: "c3", title: "C", updated: 5, turns: [turn("user", "x")] }, { id: "BAD ID", title: "x", updated: 1, turns: [] }, { id: "d4", title: "D", updated: 1, turns: [] }]);
  assert.deepEqual(mergeBackup([], bare).list.map((c) => c.id), ["c3"]);
  for (const bad of ["", "{", "{}", '{"format":"other","conversations":[]}', "42"]) {
    assert.throws(() => mergeBackup([], bad), /JSON|backup/, bad);
  }
});

test("import respects the conversation cap", () => {
  const many = Array.from({ length: MAX_CONVERSATIONS }, (_, i) => ({ id: `l${i}`, title: "L", updated: 1000 + i, pinned: false, turns: [turn("user", "x")] }));
  const old = backupJson([{ id: "old1", title: "Old", updated: 1, pinned: false, turns: [turn("user", "x")] }]);
  const { list, added } = mergeBackup(many, old);
  assert.equal(list.length, MAX_CONVERSATIONS);
  assert.equal(added, 0, "an older conversation that does not fit is not counted");
  const incoming = backupJson([0, 1, 2].map((i) => ({
    id: `new${i}`, title: "Backup", updated: 3000 + i, turns: [turn("user", "imported")],
  })));
  for (const free of [0, 1, 2, 3]) {
    const local = many.slice(free);
    const merged = mergeBackup(local, incoming);
    assert.equal(merged.added, free);
    assert.equal(merged.skipped, 3 - free);
    assert.ok(local.every((chat) => merged.list.includes(chat)), "import must preserve every existing chat");
    assert.deepEqual(merged.list.filter((chat) => chat.id.startsWith("new")).map((chat) => chat.id),
      ["new2", "new1", "new0"].slice(0, free));
  }
  const oversized = backupJson([...many, { id: "more", title: "Extra", updated: 9000, turns: [turn("user", "x")] }]);
  const merged = mergeBackup([], oversized);
  assert.equal(merged.added, MAX_CONVERSATIONS);
  assert.equal(merged.skipped, 1);
});

test("a failed legacy migration preserves the only copy and retries on the next launch", () => {
  const oldKey = "mlxl3.conversation.v1";
  const newKey = "mlxl3.conversations.v2";
  const turns = [turn("user", "Only copy")];
  const legacy = JSON.stringify(turns);
  const items = new Map([[oldKey, legacy]]);
  let quotaExceeded = true;
  let errors = 0;
  const storage = {
    getItem: (key) => items.get(key) ?? null,
    setItem(key, value) {
      if (quotaExceeded) throw new Error("QuotaExceededError");
      items.set(key, value);
    },
    removeItem: (key) => items.delete(key),
  };
  const load = () => loadConversationHistory(storage, "Chat", () => { errors++; });
  assert.deepEqual(load()[0].turns, turns);
  assert.equal(errors, 1);
  assert.equal(items.get(oldKey), legacy);
  assert.ok(!items.has(newKey));
  quotaExceeded = false;
  const migrated = load();
  assert.deepEqual(migrated[0].turns, turns);
  assert.ok(items.has(newKey) && !items.has(oldKey));
  assert.equal(load()[0].id, migrated[0].id);
  assert.equal(errors, 1);
});

test("Markdown export matches the Mac layout", () => {
  const md = markdownExport({ title: "Two\nlines", turns: [
    turn("user", "Question"),
    { role: "assistant", content: "<think>x</think>Answer", complete: true },
    { role: "assistant", content: "par", complete: false },
    { role: "assistant", content: "", complete: false, failed: "boom" },
  ] }, (text) => text.replace(/<think>[\s\S]*?<\/think>/, ""));
  assert.equal(md, "# Two lines\n\n## User\n\nQuestion\n\n## MLXL3\n\nAnswer\n\n## MLXL3\n\npar\n\n**Generation in progress — partial answer.**\n\n## MLXL3\n\n\n\n**Error:** boom\n");
});

test("branching keeps completed turns up to the message and moves a user message to the draft", () => {
  const turns = [
    { role: "user", content: "a", complete: true },
    { role: "assistant", content: "b", complete: true },
    { role: "user", content: "c", complete: true },
    { role: "assistant", content: "par", complete: false },
  ];
  assert.deepEqual(branchOf(turns, 1), { turns: turns.slice(0, 2), draft: "" });
  assert.deepEqual(branchOf(turns, 2), { turns: turns.slice(0, 2), draft: "c" });
  assert.deepEqual(branchOf(turns, 0), { turns: [], draft: "a" });
  // an unfinished answer is dropped, so its question returns to the draft
  assert.deepEqual(branchOf(turns, 3), { turns: turns.slice(0, 2), draft: "c" });
  for (const bad of [-1, 4, 1.5, NaN]) assert.throws(() => branchOf(turns, bad));
});
