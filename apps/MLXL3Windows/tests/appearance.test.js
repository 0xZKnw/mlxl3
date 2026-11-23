import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ACCENTS, applyAppearance, cssVariables, DEFAULT_APPEARANCE, idleSeconds, normalizeAppearance, normalizeHex,
  normalizeTextSize, TEXT_SIZE,
} from "../src/appearance.js";

test("defaults match the Mac preferences", () => {
  assert.deepEqual(normalizeAppearance(undefined), { ...DEFAULT_APPEARANCE });
  const css = cssVariables(DEFAULT_APPEARANCE);
  assert.equal(css["--tint-dark"], "#BBA4F3");
  assert.equal(css["--tint-light"], "#6642A4");
  assert.equal(css["--message-max"], "820px");
  assert.equal(css["--composer-max"], "790px");
  assert.equal(css["--text-size"], "15px");
  assert.equal(css["--turn-gap"], "38px");
  assert.equal(css["--sidebar-width"], "232px");
});

test("invalid stored values fall back field by field", () => {
  const a = normalizeAppearance({ accent: "neon", width: "huge", textSize: "x", density: 3, font: "comic", bubble: "x", sidebar: null, custom: "#12" });
  assert.deepEqual(a, { ...DEFAULT_APPEARANCE });
  assert.equal(normalizeAppearance({ accent: "coral" }).accent, "rose", "legacy Mac accents map like AppAccent.load");
  assert.equal(normalizeAppearance({ accent: "sage" }).accent, "neutral");
  assert.equal(normalizeAppearance({ accent: "toString" }).accent, "violet", "prototype keys are not accents");
  assert.equal(normalizeAppearance({ width: "constructor" }).width, "centered");
});

test("text size is rounded and clamped to 12...24", () => {
  assert.equal(normalizeTextSize(18.4), 18);
  assert.equal(normalizeTextSize(30), TEXT_SIZE.max);
  assert.equal(normalizeTextSize(-1), TEXT_SIZE.min);
  for (const bad of [true, null, "", NaN, Infinity, "abc"]) assert.equal(normalizeTextSize(bad), TEXT_SIZE.default, String(bad));
  assert.equal(normalizeTextSize("20"), 20);
});

test("custom accent hex is validated and used for both themes", () => {
  assert.equal(normalizeHex("#a1b2c3"), "A1B2C3");
  assert.equal(normalizeHex("zzzzzz"), "BBA4F3");
  const css = cssVariables({ accent: "custom", custom: "00ff88" });
  assert.equal(css["--tint-dark"], "#00FF88");
  assert.equal(css["--tint-light"], "#00FF88");
  for (const name of Object.keys(ACCENTS)) assert.ok(cssVariables({ accent: name })["--tint-dark"]);
});

test("wide reading width removes the caps; applyAppearance writes every variable", () => {
  const css = cssVariables({ width: "wide", density: "tight", font: "monospaced", sidebar: "wide" });
  assert.equal(css["--message-max"], "none");
  assert.equal(css["--composer-max"], "none");
  assert.equal(css["--turn-gap"], "12px");
  assert.equal(css["--message-font"], "var(--mono)");
  assert.equal(css["--sidebar-width"], "280px");
  const written = {};
  const root = { style: { setProperty: (k, v) => { written[k] = v; } }, dataset: {} };
  const applied = applyAppearance(root, { bubble: "tinted", textSize: 99 });
  assert.equal(root.dataset.bubble, "tinted");
  assert.equal(applied.textSize, 24);
  assert.deepEqual(Object.keys(written).sort(), Object.keys(css).sort());
});

test("idle unload delays mirror ModelIdleUnloadDelay", () => {
  assert.equal(idleSeconds("never"), 0);
  assert.equal(idleSeconds("1m"), 60);
  assert.equal(idleSeconds("2h"), 7200);
  assert.equal(idleSeconds("bogus"), 900, "default is 15 minutes");
  assert.equal(idleSeconds("__proto__"), 900);
});

test("background is ASCII by default, keeps soft, rejects anything else", async () => {
  const { BACKGROUNDS } = await import("../src/appearance.js");
  assert.deepEqual(BACKGROUNDS, ["ascii", "soft"]);
  assert.equal(normalizeAppearance({}).background, "ascii");
  assert.equal(normalizeAppearance({ background: "soft" }).background, "soft");
  for (const bad of ["mesh", "__proto__", "toString", 1, null]) assert.equal(normalizeAppearance({ background: bad }).background, "ascii", String(bad));
  const root = { style: { setProperty() {} }, dataset: {} };
  applyAppearance(root, { background: "soft" });
  assert.equal(root.dataset.background, "soft");
  applyAppearance(root, { background: "nope" });
  assert.equal(root.dataset.background, "ascii");
});

test("every appearance menu option survives normalisation", async () => {
  const { readFileSync } = await import("node:fs");
  const html = readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
  const menus = [...html.matchAll(/<select data-set="([a-zA-Z]+)">([\s\S]*?)<\/select>/g)];
  assert.ok(menus.some((m) => m[1] === "background"), "Settings → Appearance offers the background");
  for (const [, field, body] of menus) {
    const values = [...body.matchAll(/value="([^"]*)"/g)].map((m) => m[1]);
    assert.ok(values.length >= 2, field);
    for (const value of values) assert.equal(normalizeAppearance({ [field]: value })[field], value, `${field}=${value}`);
  }
});
