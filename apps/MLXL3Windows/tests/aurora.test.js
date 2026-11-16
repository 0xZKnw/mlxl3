import assert from "node:assert/strict";
import { test } from "node:test";
import { cellGlyph, curtainFrames, light, PLATE_HEIGHT, PLATE_WIDTH } from "../src/aurora.js";

test("light stays finite and inside [0, 1] over the whole plate", () => {
  for (let i = 0; i <= 200; i++) {
    for (let j = 0; j <= 200; j++) {
      const x = -1 + (2 * i) / 200;
      const y = j / 200;
      const { rgb, alpha } = light(x, y);
      for (const v of [...rgb, alpha]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 1, `${x},${y}: ${v}`);
    }
  }
});

test("light fades to zero at the plate edges and peaks on the hot foot", () => {
  assert.equal(light(-1, 0.5).alpha, 0);
  assert.equal(light(1, 0.5).alpha, 0);
  assert.equal(light(0, 0).alpha, 0);
  assert.equal(light(0, 1).alpha, 0);
  const foot = light(0, 0.70 - 0.055);
  assert.ok(foot.alpha > 0.9, String(foot.alpha));
  assert.ok(foot.rgb[0] > 0.9 && foot.rgb[1] > 0.6, foot.rgb.join());
  // Above the column (top of the plate, off-centre) is violet: blue dominates red and green.
  const column = light(0, 0.3);
  assert.ok(column.rgb[2] > column.rgb[1], column.rgb.join());
});

test("glyph grid uses the 7-glyph alphabet and leaves dark cells empty", () => {
  let printed = 0;
  for (let row = 0; row < PLATE_HEIGHT / 8; row++) {
    for (let col = 0; col < PLATE_WIDTH / 5; col++) {
      const g = cellGlyph(col, row);
      assert.ok(Number.isInteger(g) && g >= -1 && g <= 6);
      if (g >= 0) printed++;
    }
  }
  assert.ok(printed > 500, String(printed));
  assert.equal(cellGlyph(0, 0), -1);
});

test("curtains follow the Mac full-height layout", () => {
  const frames = curtainFrames(1400, 900);
  assert.equal(frames.length, 4);
  const plateHeight = 900 * 0.79;
  assert.ok(Math.abs(frames[1].height - plateHeight * 1.08) < 1e-9);
  assert.ok(Math.abs(frames[1].width - Math.min(1400 * 0.66, plateHeight * 2.1) * 1.08) < 1e-9);
  assert.ok(Math.abs(frames[0].cx - 1400 * 0.045) < 1e-9); // sin(0) = 0
  assert.deepEqual(frames.map((f) => f.hue), [-38, 0, 9, -26]);
  for (const f of curtainFrames(1, 1)) assert.ok(f.width > 0 && f.height > 0);
});
