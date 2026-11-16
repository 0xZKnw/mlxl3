// Static checks of the page: every translated key exists in both languages, every element
// app.js looks up exists, and nothing violates the CSP (no inline style or script).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { KEYS, table } from "../src/i18n.js";
import { icon, ICONS, monogram } from "../src/icons.js";

const html = readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
const app = readFileSync(new URL("../src/app.js", import.meta.url), "utf8");

test("English and French define the same keys", () => {
  assert.deepEqual(Object.keys(table("fr")).sort(), [...KEYS].sort());
  for (const key of KEYS) assert.ok(table("fr")[key].trim(), key);
});

test("every key used by the page and the script is translated", () => {
  const used = new Set();
  for (const m of html.matchAll(/data-i18n(?:-placeholder|-title)?="([^"]+)"/g)) used.add(m[1]);
  for (const m of app.matchAll(/\bt\("([A-Za-z]+)"\)/g)) used.add(m[1]);
  assert.ok(used.size > 60, String(used.size));
  for (const key of used) assert.ok(KEYS.includes(key), `missing key ${key}`);
});

test("every id the script looks up exists exactly once", () => {
  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(new Set(ids).size, ids.length, "duplicate id");
  const looked = new Set([...app.matchAll(/\$\("([a-z0-9-]+)"\)/g)].map((m) => m[1]));
  for (const m of app.matchAll(/\$\(`\$\{id\}-value`\)/g)) void m;
  for (const id of looked) assert.ok(ids.includes(id), `missing #${id}`);
  for (const id of ["temperature", "top-p", "top-k", "min-p", "rep"]) assert.ok(ids.includes(`${id}-value`), id);
});

test("the page respects the CSP: no inline styles, scripts or handlers", () => {
  assert.doesNotMatch(html, /\sstyle=/);
  assert.doesNotMatch(html, /\son[a-z]+=/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)/);
  assert.doesNotMatch(app, /style="/);
});

test("every icon named by the page and the script exists", () => {
  const names = new Set([...html.matchAll(/data-icon="([^"]+)"/g)].map((m) => m[1]));
  for (const m of app.matchAll(/icon\("([A-Za-z]+)"/g)) names.add(m[1]);
  for (const m of app.matchAll(/setIcon\([^,]+, "([A-Za-z]+)"\)/g)) names.add(m[1]);
  for (const name of names) assert.ok(ICONS.includes(name), `missing icon ${name}`);
  assert.match(icon("copy"), /^<svg[^>]+viewBox="0 0 20 20"/);
  assert.equal(icon("nope").includes("<path"), false);
  assert.match(monogram(35), /width="35"/);
});

test("recommended card icons exist", async () => {
  const { RECOMMENDED } = await import("../src/store.js");
  for (const model of RECOMMENDED.flatMap((f) => f.models)) assert.ok(ICONS.includes(model.icon), model.icon);
});
