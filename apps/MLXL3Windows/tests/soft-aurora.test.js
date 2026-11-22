import assert from "node:assert/strict";
import { test } from "node:test";
import {
  alternate, AuroraPlayback, bezier, BUSY, EASE_IN_OUT, fieldFrame, GLIDE, IDLE, LIGHTS, MESH_SIZE, MESH_X, MESH_Y, meshColor,
  meshParameter, PALETTE, SETTLE, SoftAurora, Tween,
} from "../src/soft-aurora.js";

const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// Deterministic PRNG for property checks.
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

test("mesh colours stay finite and inside [0, 1] for both palettes", () => {
  for (const palette of Object.values(PALETTE)) {
    for (let i = 0; i <= 120; i++) {
      for (let j = 0; j <= 120; j++) {
        const { rgb, alpha } = meshColor(palette, i / 120, j / 120);
        for (const v of [...rgb, alpha]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 1, `${i},${j}: ${v}`);
      }
    }
    for (const bad of [NaN, -5, 7, Infinity]) {
      const { rgb, alpha } = meshColor(palette, bad, bad);
      for (const v of [...rgb, alpha]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 1, String(bad));
    }
  }
});

test("the mesh passes through the Swift control colours", () => {
  // The Swift MeshGradient table, written out independently of the module's table.
  const expected = [
    [["cool", 1], ["violet", 1], ["rose", 1], ["amber", 1]],
    [["cool", 1], ["cool", 1], ["violet", 1], ["amber", 1]],
    [["cool", 0.20], ["violet", 0.28], ["rose", 0.10], ["amber", 0.17]],
  ];
  const points = [
    [[0, 0], [0.32, 0], [0.68, 0], [1, 0]],
    [[0, 0.18], [0.32, 0.42], [0.68, 0.15], [1, 0.26]],
    [[0, 0.52], [0.32, 0.73], [0.68, 0.40], [1, 0.59]],
  ];
  for (const scheme of ["dark", "light"]) {
    const palette = PALETTE[scheme];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 4; col++) {
        const [u, v] = points[row][col];
        const [name, alpha] = expected[row][col];
        const got = meshColor(palette, u, v);
        assert.ok(close(got.alpha, alpha, 1e-9), `${scheme} ${row},${col} alpha ${got.alpha}`);
        for (let k = 0; k < 3; k++) assert.ok(close(got.rgb[k], palette[name][k], 1e-9), `${scheme} ${row},${col}`);
      }
    }
    for (const u of [0, 0.2, 0.5, 0.9, 1]) assert.ok(meshColor(palette, u, 1).alpha < 1e-9, "bottom row is clear");
  }
  assert.deepEqual(PALETTE.dark.cool, [0.29, 0.42, 0.74]);
  assert.deepEqual(PALETTE.light.amber, [1, 0.82, 0.65]);
});

test("mesh parameters cover [0, 3]² and grow with v down every column", () => {
  assert.deepEqual(meshParameter(0, 0), [0, 0]);
  assert.deepEqual(meshParameter(1, 1), [3, 3]);
  assert.deepEqual(meshParameter(0.32, 0.42), [1, 1]);
  for (let i = 0; i <= 50; i++) {
    let last = -1;
    for (let j = 0; j <= 200; j++) {
      const [p, q] = meshParameter(i / 50, j / 200);
      assert.ok(p >= 0 && p <= 3 && q >= 0 && q <= 3);
      assert.ok(q >= last - 1e-12, `${i},${j}`);
      last = q;
    }
  }
  assert.equal(MESH_X.length, 4);
  for (let col = 0; col < 4; col++) for (let row = 0; row < 3; row++) assert.ok(MESH_Y[row][col] < MESH_Y[row + 1][col]);
});

test("timing curves invert their parametric definition", () => {
  const curves = [[0.22, 1, 0.36, 1, GLIDE], [0.5, -0.06, 0.25, 1.16, SETTLE], [0.42, 0, 0.58, 1, EASE_IN_OUT]];
  for (const [x1, y1, x2, y2, curve] of curves) {
    const axis = (a, b, t) => 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3;
    for (let k = 1; k < 100; k++) {
      const t = k / 100;
      assert.ok(close(curve(axis(x1, x2, t)), axis(y1, y2, t), 1e-6), `${x1},${y1} at ${t}`);
    }
    assert.equal(curve(0), 0);
    assert.equal(curve(1), 1);
    assert.equal(curve(-1), 0);
    assert.equal(curve(NaN), 0);
  }
  assert.ok(close(EASE_IN_OUT(0.5), 0.5, 1e-9));
  assert.ok(Math.max(...Array.from({ length: 99 }, (_, i) => SETTLE((i + 1) / 100))) > 1, "the descent overshoots");
  assert.ok(close(bezier(0, 0, 1, 1)(0.3), 0.3, 1e-6), "the diagonal is linear");
});

test("alternate swings 0 → 1 → 0 over twice its duration", () => {
  assert.equal(alternate(0, 44), 0);
  assert.ok(close(alternate(44, 44), 1));
  assert.ok(close(alternate(88, 44), 0));
  assert.ok(close(alternate(22, 44), 0.5));
});

test("playback integral matches a numerical integral of its rate curve", () => {
  const simpson = (f, a, b, n = 2000) => {
    const h = (b - a) / n;
    let sum = f(a) + f(b);
    for (let i = 1; i < n; i++) sum += f(a + i * h) * (i % 2 ? 4 : 2);
    return (sum * h) / 3;
  };
  const cases = [
    { from: 1, target: 1 }, // coasting (quartic ease-out)
    { from: 1, target: 4.4 }, // accelerating with the back dip
    { from: 0.5, target: 4.4 }, // restart too low for the dip
    { from: 4.4, target: 1 },
  ];
  for (const c of cases) {
    const play = new AuroraPlayback(0);
    Object.assign(play, c);
    for (const p of [0.1, 0.25, 0.5, 0.7, 1]) {
      assert.ok(close(play.integral(p), simpson((s) => play.curve(s), 0, p), 1e-7), `${JSON.stringify(c)} at ${p}`);
    }
    assert.ok(close(play.curve(0), 0, 1e-12) && close(play.curve(1), 1, 1e-12));
  }
  const dip = new AuroraPlayback(0);
  Object.assign(dip, { from: 1, target: 4.4 });
  assert.ok(dip.back > 0);
  const lowest = Math.min(...Array.from({ length: 101 }, (_, i) => dip.curve(i / 100)));
  assert.ok(lowest < 0 && lowest > -0.11, String(lowest));
});

test("playback phase is continuous, never runs backwards and settles at 1× or 4.4×", () => {
  const play = new AuroraPlayback(0);
  assert.ok(close(play.phase(1), 1));
  play.setGenerating(true, 10);
  assert.ok(close(play.phase(10), 10), "a speed change keeps the current position");
  const a = play.phase(20);
  assert.ok(close(play.phase(21) - a, 4.4, 1e-9), "busy runs 4.4× after the 2 s ramp");
  play.setGenerating(false, 30);
  assert.ok(close(play.phase(40) - play.phase(39), 1, 1e-9), "idle runs 1× after the 2.6 s ramp");
  play.setGenerating(false, 41);
  assert.equal(play.target, 1, "repeating the same state is ignored");

  const random = lcg(7);
  for (let round = 0; round < 40; round++) {
    const p = new AuroraPlayback(0);
    let time = 0;
    let last = p.phase(0);
    for (let step = 0; step < 300; step++) {
      time += random() * 0.6;
      if (random() < 0.2) {
        const before = p.phase(time);
        p.setGenerating(random() < 0.5, time);
        assert.ok(close(p.phase(time), before, 1e-9), "toggle is continuous");
      }
      const now = p.phase(time);
      assert.ok(Number.isFinite(now) && now >= last - 1e-9, `round ${round} step ${step}: ${last} → ${now}`);
      last = now;
    }
  }
});

test("tweens retarget from their current value and honour the delay", () => {
  const tween = new Tween(0);
  assert.equal(tween.value(5), 0);
  assert.ok(tween.settled(0));
  tween.set(10, 0, { duration: 2, delay: 1 });
  assert.equal(tween.value(0.5), 0, "still waiting for the delay");
  assert.ok(close(tween.value(2), 5), "linear by default");
  assert.equal(tween.value(3), 10);
  assert.ok(!tween.settled(2.9) && tween.settled(3));
  tween.set(10, 4, { duration: 9 });
  assert.equal(tween.value(4), 10, "same target is ignored");
  tween.set(0, 4, { duration: 2 });
  tween.set(20, 5, { duration: 2 });
  assert.ok(close(tween.value(5), 5), "the interrupted value is the new start");
  tween.set(1, 6, {});
  assert.equal(tween.value(6), 1, "zero duration jumps");
});

test("frame geometry follows the Swift layout at rest", () => {
  const frame = fieldFrame({ phase: 0, band: IDLE.band, saturation: IDLE.saturation, descents: [0, 0, 0, 0, 0] }, 1400, 900);
  assert.ok(close(frame.bandHeight, 900 * 0.51));
  assert.ok(close(frame.mesh.cx, 700 - 0.05 * 1400), "drift 0 shifts the mesh left by 5%");
  assert.ok(close(frame.mesh.width, 1400 * 1.12));
  assert.ok(close(frame.hue, -7));
  assert.equal(frame.saturation, 1.03);
  assert.deepEqual(frame.lights.map((l) => l.color), ["cool", "violet", "rose", "amber", "amber"]);
  LIGHTS.forEach((light, i) => {
    const got = frame.lights[i];
    assert.ok(close(got.cx, 1400 * light.x));
    assert.ok(close(got.cy, 900 * 0.035));
    assert.ok(close(got.opacity, light.opacity));
    assert.ok(close(got.rx, (1400 * 0.49) / 2) && close(got.ry, (900 * 0.34) / 2));
  });
  const busy = fieldFrame({ phase: 0, band: BUSY.band, saturation: BUSY.saturation, descents: LIGHTS.map((l) => l.descent) }, 1400, 900);
  LIGHTS.forEach((light, i) => assert.ok(close(busy.lights[i].cy, 900 * (0.035 + light.descent))));
  for (let k = 0; k < 200; k++) {
    const f = fieldFrame({ phase: k * 3.7, band: 0.6, saturation: 1.2 }, 1, 1);
    for (const l of f.lights) for (const v of [l.cx, l.cy, l.rx, l.ry, l.opacity]) assert.ok(Number.isFinite(v));
    assert.ok(f.hue >= -7 - 1e-9 && f.hue <= 6 + 1e-9);
    for (const l of f.lights) assert.ok(l.opacity >= 0.42 - 1e-9 && l.opacity <= 1 + 1e-9);
  }
});

// A minimal DOM: canvases record what is drawn, timers and frames are queued by hand.
function fakeWindow({ reduce = false, light = false } = {}) {
  let clock = 0;
  const timeouts = [];
  const frames = [];
  const log = [];
  const env = { reduce, light };
  const context = (canvas) => ({
    filter: "none", globalAlpha: 1, globalCompositeOperation: "source-over", fillStyle: null,
    imageSmoothingEnabled: false, imageSmoothingQuality: "low",
    setTransform() {}, save() {}, restore() {}, translate() {}, scale() {}, fillRect() {},
    clearRect() { log.push({ op: "clear", canvas: canvas.id }); },
    drawImage(image) { log.push({ op: "draw", canvas: canvas.id, image: image.id, filter: this.filter }); },
    createRadialGradient() { return { addColorStop() {} }; },
    createLinearGradient() {
      const gradient = { stops: [], addColorStop(offset, color) { gradient.stops.push([offset, color]); } };
      log.push({ op: "fade", gradient });
      return gradient;
    },
    createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
    putImageData(image) { log.push({ op: "mesh", canvas: canvas.id, image }); },
  });
  let ids = 0;
  const makeCanvas = () => {
    const canvas = { id: ++ids, width: 300, height: 150, clientWidth: 0, clientHeight: 0, ownerDocument: doc };
    canvas.getContext = () => (canvas.ctx ??= context(canvas));
    return canvas;
  };
  const win = {
    performance: { now: () => clock * 1000 },
    matchMedia: (query) => ({
      matches: query.includes("reduced-motion") ? env.reduce : query.includes("light") ? env.light : false,
      addEventListener() {},
    }),
    setTimeout(fn, ms) { timeouts.push({ fn, ms, live: true }); return timeouts.length; },
    clearTimeout(id) { if (timeouts[id - 1]) timeouts[id - 1].live = false; },
    requestAnimationFrame(fn) { frames.push({ fn, live: true }); return frames.length; },
    cancelAnimationFrame(id) { if (frames[id - 1]) frames[id - 1].live = false; },
  };
  const listeners = {};
  const doc = {
    hidden: false, documentElement: { dataset: {} }, defaultView: win,
    createElement: () => makeCanvas(),
    addEventListener(type, fn) { listeners[type] = fn; },
  };
  const canvas = makeCanvas();
  canvas.clientWidth = 1000;
  canvas.clientHeight = 600;
  const pending = () => ({
    timeouts: timeouts.filter((t) => t.live).map((t) => t.ms),
    frames: frames.filter((f) => f.live).length,
  });
  const fire = () => {
    const next = [...frames, ...timeouts].find((t) => t.live);
    assert.ok(next, "something is scheduled");
    next.live = false;
    next.fn();
  };
  return { win, doc, canvas, log, listeners, env, pending, fire, set clock(v) { clock = v; }, get clock() { return clock; } };
}

test("soft aurora ticks at 6 fps idle, display rate while settling, 12 fps busy", () => {
  const w = fakeWindow();
  const aurora = new SoftAurora(w.canvas);
  aurora.setBusy(false);
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 }, "nothing runs before start");
  aurora.start();
  assert.equal(w.canvas.width, 500, "half resolution backing store");
  assert.equal(w.canvas.height, 300);
  assert.deepEqual(w.pending(), { timeouts: [167], frames: 0 });
  w.clock = 1;
  w.fire();
  assert.ok(close(aurora.state(1).phase, 1), "idle field runs at 1×");
  assert.deepEqual(w.pending(), { timeouts: [167], frames: 0 });

  aurora.setBusy(true);
  assert.deepEqual(w.pending(), { timeouts: [], frames: 1 }, "settling runs at display rate");
  let frames = 0;
  while (w.pending().frames) {
    w.clock += 1 / 60;
    w.fire();
    frames++;
  }
  // Slowest lobe: 2.4 s settle after a 0.38 s delay.
  assert.ok(w.clock >= 1 + 2.78 && w.clock < 1 + 2.78 + 0.05, String(w.clock));
  assert.ok(frames > 150, String(frames));
  assert.deepEqual(w.pending(), { timeouts: [83], frames: 0 });
  const busy = aurora.state(w.clock);
  assert.ok(close(busy.band, BUSY.band) && close(busy.saturation, BUSY.saturation));
  LIGHTS.forEach((light, i) => assert.ok(close(busy.descents[i], light.descent)));

  const last = w.log.filter((e) => e.op === "draw" && e.canvas === w.canvas.id).at(-1);
  assert.match(last.filter, /^hue-rotate\(-?\d+\.\d{2}deg\) saturate\(1\.650\)$/);
  assert.equal(w.canvas.ctx.filter, "none", "the filter is reset after drawing");
  const fade = w.log.filter((e) => e.op === "fade").at(-1).gradient.stops;
  assert.deepEqual(fade, [[0, "rgba(0,0,0,1)"], [0.15, "rgba(0,0,0,1)"], [0.48, "rgba(0,0,0,0.6)"], [1, "rgba(0,0,0,0)"]]);
  assert.equal(w.log.filter((e) => e.op === "mesh").length, 1, "the mesh bitmap is built once per theme");

  aurora.stop();
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 });
});

test("hidden window, other pages and stop pause the field without a jump", () => {
  const w = fakeWindow();
  const aurora = new SoftAurora(w.canvas);
  aurora.start();
  w.clock = 5;
  w.fire();
  w.doc.hidden = true;
  w.listeners.visibilitychange();
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 }, "nothing runs while hidden");
  const frozen = aurora.state(5).phase;
  assert.ok(close(aurora.state(50).phase, frozen), "the clock is paused");
  w.clock = 50;
  w.doc.hidden = false;
  w.listeners.visibilitychange();
  assert.ok(close(aurora.state(50).phase, frozen), "resuming continues from the same place");
  assert.ok(close(aurora.state(51).phase, frozen + 1));
  assert.deepEqual(w.pending(), { timeouts: [167], frames: 0 });

  w.clock = 52;
  aurora.setAnimated(false);
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 }, "settings pages show a still field");
  assert.ok(close(aurora.state(90).phase, frozen + 2));
  aurora.setAnimated(false);
  w.clock = 60;
  aurora.setAnimated(true);
  assert.ok(close(aurora.state(60).phase, frozen + 2));
  aurora.stop();
  w.clock = 70;
  assert.ok(close(aurora.state(70).phase, frozen + 2), "stopping also pauses");
});

test("reduce motion keeps phase 0 and applies generation changes at once", () => {
  const w = fakeWindow({ reduce: true });
  const aurora = new SoftAurora(w.canvas);
  aurora.start();
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 });
  w.clock = 3;
  aurora.setBusy(true);
  assert.deepEqual(w.pending(), { timeouts: [], frames: 0 }, "no animation frames");
  const state = aurora.state(3);
  assert.equal(state.phase, 0);
  assert.equal(state.band, BUSY.band);
  assert.deepEqual(state.descents, LIGHTS.map((l) => l.descent));
  const last = w.log.filter((e) => e.op === "draw" && e.canvas === w.canvas.id).at(-1);
  assert.equal(last.filter, "hue-rotate(-7.00deg) saturate(1.650)");
});

test("the mesh follows the dark and light palettes", () => {
  const w = fakeWindow({ light: true });
  const aurora = new SoftAurora(w.canvas);
  aurora.start();
  const meshes = () => w.log.filter((e) => e.op === "mesh").map((e) => e.image);
  assert.equal(meshes().length, 1);
  const [mw, mh] = MESH_SIZE;
  assert.equal(meshes()[0].width, mw);
  assert.equal(meshes()[0].height, mh);
  const first = meshes()[0].data;
  assert.ok(Math.abs(first[1] - 0.77 * 255) < 4, `light cool green ${first[1]}`);
  assert.ok(first[3] > 250, "top edge is opaque");
  assert.ok(first[4 * (mw * (mh - 1)) + 3] < 8, "bottom edge is clear");
  w.doc.documentElement.dataset.theme = "dark";
  aurora.start();
  assert.equal(meshes().length, 2);
  assert.ok(Math.abs(meshes()[1].data[1] - 0.42 * 255) < 4, `dark cool green ${meshes()[1].data[1]}`);
  w.doc.documentElement.dataset.theme = "light";
  aurora.start();
  assert.equal(meshes().length, 2, "cached per theme");
  aurora.stop();
});
