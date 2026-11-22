// The Mac app's soft aurora from the 2.0 betas (Desktop 2.0.0-beta.19/20, StudioAurora.swift at
// 7becaaba3), before the ASCII print replaced it: a 4×4 mesh of cool, violet, rose and amber light
// across the top of the window with five breathing lobes drifting through it. While a reply is
// generating the band grows, the lobes settle lower, the colours saturate and the field speeds up.
// It ticks at 6 fps idle and 12 fps busy, runs at display rate only while a transition settles,
// and stops when the window is hidden, another page is shown or Reduce motion is on.

const clamp01 = (v) => Math.min(1, Math.max(0, v));

export const PALETTE = {
  dark: { cool: [0.29, 0.42, 0.74], violet: [0.47, 0.36, 0.64], rose: [0.69, 0.38, 0.52], amber: [0.77, 0.48, 0.32] },
  light: { cool: [0.61, 0.77, 0.99], violet: [0.80, 0.73, 0.96], rose: [0.98, 0.77, 0.84], amber: [1, 0.82, 0.65] },
};

// MeshGradient(width: 4, height: 4): every row shares the column x positions; rows bend in y.
export const MESH_X = [0, 0.32, 0.68, 1];
export const MESH_Y = [
  [0, 0, 0, 0],
  [0.18, 0.42, 0.15, 0.26],
  [0.52, 0.73, 0.40, 0.59],
  [1, 1, 1, 1],
];
const MESH_COLORS = [
  [["cool", 1], ["violet", 1], ["rose", 1], ["amber", 1]],
  [["cool", 1], ["cool", 1], ["violet", 1], ["amber", 1]],
  [["cool", 0.20], ["violet", 0.28], ["rose", 0.10], ["amber", 0.17]],
  [[null, 0], [null, 0], [null, 0], [null, 0]],
];

// Travel, breathing periods, directions and stagger of the five lobes (Swift `lights`).
export const LIGHTS = [
  { x: 0.12, dx: 0.19, dy: 0.11, travelTime: 21, breathTime: 13, opacity: 0.95, endOpacity: 0.50, descent: 0.14, settle: 1.7, delay: 0.05 },
  { x: 0.34, dx: -0.13, dy: 0.17, travelTime: 27, breathTime: 17, opacity: 0.55, endOpacity: 1, descent: 0.08, settle: 2.2, delay: 0.26 },
  { x: 0.50, dx: 0.14, dy: -0.08, travelTime: 23, breathTime: 11, opacity: 0.90, endOpacity: 0.42, descent: 0.17, settle: 1.5, delay: 0.13 },
  { x: 0.70, dx: -0.17, dy: 0.14, travelTime: 19, breathTime: 15, opacity: 0.60, endOpacity: 1, descent: 0.10, settle: 2.4, delay: 0.38 },
  { x: 0.91, dx: -0.21, dy: -0.10, travelTime: 25, breathTime: 19, opacity: 1, endOpacity: 0.58, descent: 0.15, settle: 1.9, delay: 0.19 },
];
export const LIGHT_COLORS = ["cool", "violet", "rose", "amber", "amber"];

export const IDLE = { band: 0.51, saturation: 1.03, interval: 167 };
export const BUSY = { band: 0.77, saturation: 1.65, interval: 83 };
export const RESOLUTION = 0.5; // backing pixels per CSS pixel: the field is a blur, half resolution is invisible
export const MESH_SIZE = [128, 64];

/** Cubic Bézier timing curve through (0,0), (x1,y1), (x2,y2), (1,1), like SwiftUI's timingCurve. */
export function bezier(x1, y1, x2, y2) {
  const axis = (a, b, t) => 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3;
  return (p) => {
    if (!(p > 0)) return 0;
    if (p >= 1) return 1;
    let lo = 0;
    let hi = 1;
    for (let k = 0; k < 40; k++) {
      const mid = (lo + hi) / 2;
      if (axis(x1, x2, mid) < p) lo = mid; else hi = mid;
    }
    return axis(y1, y2, (lo + hi) / 2);
  };
}

export const GLIDE = bezier(0.22, 1, 0.36, 1); // UnitCurve.bezier of the lobes' travel
export const SETTLE = bezier(0.5, -0.06, 0.25, 1.16); // the lobes' descent, with a small overshoot
export const EASE_IN_OUT = bezier(0.42, 0, 0.58, 1); // SwiftUI .easeInOut

/** 0 → 1 → 0 cosine swing with a full period of 2 × `duration` seconds of phase. */
export function alternate(phase, duration) {
  return (1 - Math.cos((Math.PI * phase) / duration)) / 2;
}

// Catmull-Rom through four values, between b (t = 0) and c (t = 1).
function catmull(a, b, c, d, t) {
  return b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
}

/** Mesh parameters (column, row), each in [0, 3], for a point (u, v) of the unit band. */
export function meshParameter(u, v) {
  const x = clamp01(Number.isFinite(u) ? u : 0);
  const y = clamp01(Number.isFinite(v) ? v : 0);
  let i = 0;
  while (i < 2 && x > MESH_X[i + 1]) i++;
  const t = (x - MESH_X[i]) / (MESH_X[i + 1] - MESH_X[i]);
  const rowY = (j) => MESH_Y[j][i] + (MESH_Y[j][i + 1] - MESH_Y[j][i]) * t;
  let j = 0;
  while (j < 2 && y > rowY(j + 1)) j++;
  const top = rowY(j);
  const bottom = rowY(j + 1);
  return [i + t, j + (bottom > top ? clamp01((y - top) / (bottom - top)) : 0)];
}

/** Mesh colour (0–1 per channel) and alpha at (u, v), interpolated smoothly in premultiplied space. */
export function meshColor(palette, u, v) {
  const [p, q] = meshParameter(u, v);
  const i = Math.min(2, Math.floor(p));
  const j = Math.min(2, Math.floor(q));
  const at = (row, col) => {
    const [name, alpha] = MESH_COLORS[Math.min(3, Math.max(0, row))][Math.min(3, Math.max(0, col))];
    const rgb = name ? palette[name] : [0, 0, 0];
    return [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha];
  };
  const out = [0, 0, 0, 0];
  for (let k = 0; k < 4; k++) {
    const rows = [-1, 0, 1, 2].map((dr) => {
      const [a, b, c, d] = [-1, 0, 1, 2].map((dc) => at(j + dr, i + dc)[k]);
      return catmull(a, b, c, d, p - i);
    });
    out[k] = catmull(rows[0], rows[1], rows[2], rows[3], q - j);
  }
  const alpha = clamp01(out[3]);
  const rgb = alpha > 0 ? [clamp01(out[0] / alpha), clamp01(out[1] / alpha), clamp01(out[2] / alpha)] : [0, 0, 0];
  return { rgb, alpha };
}

/** Integrates the playback rate so changing speed never resets the field's position (Swift AuroraPlayback). */
export class AuroraPlayback {
  constructor(time = 0) {
    this.start = time;
    this.origin = 0;
    this.from = 1;
    this.target = 1;
    this.duration = 2;
  }

  setGenerating(busy, time) {
    const next = busy ? 4.4 : 1;
    if (next === this.target) return;
    const progress = clamp01((time - this.start) / this.duration);
    this.origin = this.phase(time);
    this.from += (this.target - this.from) * this.curve(progress);
    this.start = time;
    this.target = next;
    this.duration = busy ? 2 : 2.6;
  }

  phase(time) {
    const elapsed = Math.max(0, time - this.start);
    const p = Math.min(1, elapsed / this.duration);
    return this.origin + this.duration * (this.from * p + (this.target - this.from) * this.integral(p))
      + Math.max(0, elapsed - this.duration) * this.target;
  }

  // The back curve dips by about 10%. Skip that dip when a rapid restart could reverse the field.
  get back() {
    return this.from < this.target && this.from - 0.11 * (this.target - this.from) >= 0.12 ? 1.70158 * 1.525 : 0;
  }

  // Back-in/out for acceleration, quartic ease-out for coasting.
  curve(p) {
    if (this.target === 1) return 1 - (1 - p) ** 4;
    const c = this.back;
    const u = p < 0.5 ? 2 * p : 2 * (1 - p);
    const value = (u * u * ((c + 1) * u - c)) / 2;
    return p < 0.5 ? value : 1 - value;
  }

  integral(p) {
    if (this.target === 1) return p + ((1 - p) ** 5 - 1) / 5;
    const c = this.back;
    const u = Math.min(p, 1 - p);
    const leading = (c + 1) * u ** 4 - ((2 * c) / 3) * u ** 3;
    return p <= 0.5 ? leading : p - 0.5 + leading;
  }
}

/** A value that eases to each new target from wherever it is, like an interrupted SwiftUI animation. */
export class Tween {
  constructor(value) {
    this.from = value;
    this.to = value;
    this.start = 0;
    this.duration = 0;
    this.delay = 0;
    this.curve = (p) => p;
  }

  set(to, time, { duration = 0, delay = 0, curve = (p) => p } = {}) {
    if (to === this.to) return;
    this.from = this.value(time);
    this.to = to;
    this.start = time;
    this.duration = duration;
    this.delay = delay;
    this.curve = curve;
  }

  value(time) {
    if (!(this.duration > 0)) return this.to;
    const p = (time - this.start - this.delay) / this.duration;
    if (p <= 0) return this.from;
    if (p >= 1) return this.to;
    return this.from + (this.to - this.from) * this.curve(p);
  }

  settled(time) {
    return !(this.duration > 0) || time >= this.start + this.delay + this.duration;
  }
}

/** Everything one frame draws for a `width` × `height` window, in CSS px. */
export function fieldFrame({ phase, band, saturation, descents }, width, height) {
  const drift = alternate(phase, 44);
  return {
    bandHeight: height * band,
    mesh: { cx: width / 2 + (drift - 0.5) * width * 0.10, width: width * 1.12 },
    hue: -7 + 13 * alternate(phase, 52),
    saturation,
    lights: LIGHTS.map((light, i) => {
      const travel = GLIDE(alternate(phase, light.travelTime));
      const breath = alternate(phase, light.breathTime);
      const scale = 1 + 0.16 * travel;
      return {
        color: LIGHT_COLORS[i],
        opacity: light.opacity + (light.endOpacity - light.opacity) * breath,
        cx: width * (light.x + light.dx * travel),
        cy: height * (0.035 + light.dy * travel + (descents?.[i] ?? 0)),
        rx: (width * 0.49 * scale) / 2,
        ry: (height * 0.34 * scale) / 2,
      };
    }),
  };
}

const rgba = (rgb, alpha) => `rgba(${rgb.map((v) => Math.round(v * 255)).join(",")},${alpha})`;

/** The animated soft aurora on the window's background canvas. */
export class SoftAurora {
  constructor(canvas, { now } = {}) {
    this.canvas = canvas;
    this.doc = canvas.ownerDocument;
    this.win = this.doc.defaultView;
    this.now = now ?? (() => this.win.performance.now() / 1000);
    this.playback = new AuroraPlayback(0);
    this.base = 0; // the field's own clock only runs while it moves, so pausing never makes it jump
    this.since = null;
    this.busy = false;
    this.animated = true;
    this.active = false;
    this.timer = 0;
    this.frame = 0;
    this.band = new Tween(IDLE.band);
    this.saturation = new Tween(IDLE.saturation);
    this.descents = LIGHTS.map(() => new Tween(0));
    this.meshes = {};
    this.layer = null;
    this.listening = false;
    this.reschedule = () => this.schedule();
  }

  reduceMotion() {
    return this.win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  isDark() {
    const theme = this.doc.documentElement.dataset.theme;
    if (theme) return theme !== "light";
    return !this.win.matchMedia("(prefers-color-scheme: light)").matches;
  }

  clock(time) {
    return this.base + (this.since === null ? 0 : time - this.since);
  }

  start() {
    this.active = true;
    if (!this.listening) {
      this.listening = true;
      this.doc.addEventListener("visibilitychange", this.reschedule);
      this.win.matchMedia("(prefers-reduced-motion: reduce)").addEventListener?.("change", this.reschedule);
    }
    this.schedule();
  }

  stop() {
    this.active = false;
    this.cancel();
    this.pause(this.now());
  }

  setAnimated(animated) {
    if (this.animated === Boolean(animated)) return;
    this.animated = Boolean(animated);
    this.schedule();
  }

  setBusy(busy) {
    const next = Boolean(busy);
    if (next === this.busy) return;
    this.busy = next;
    const time = this.now();
    const still = this.reduceMotion();
    this.playback.setGenerating(next, this.clock(time));
    const ease = still ? {} : { duration: next ? 2.3 : 2.6, curve: EASE_IN_OUT };
    const target = next ? BUSY : IDLE;
    this.band.set(target.band, time, ease);
    this.saturation.set(target.saturation, time, ease);
    LIGHTS.forEach((light, i) => {
      this.descents[i].set(next ? light.descent : 0, time, still ? {} : {
        duration: next ? light.settle : 2.6, delay: next ? light.delay : 0, curve: SETTLE,
      });
    });
    this.schedule();
  }

  settled(time) {
    return this.band.settled(time) && this.saturation.settled(time) && this.descents.every((d) => d.settled(time));
  }

  pause(time) {
    if (this.since === null) return;
    this.base += time - this.since;
    this.since = null;
  }

  cancel() {
    if (this.timer) this.win.clearTimeout(this.timer);
    if (this.frame) this.win.cancelAnimationFrame(this.frame);
    this.timer = 0;
    this.frame = 0;
  }

  state(time) {
    return {
      phase: this.reduceMotion() ? 0 : this.playback.phase(this.clock(time)),
      band: this.band.value(time),
      saturation: this.saturation.value(time),
      descents: this.descents.map((d) => d.value(time)),
    };
  }

  schedule() {
    this.cancel();
    if (!this.active) return;
    const time = this.now();
    const moving = this.animated && !this.reduceMotion() && !this.doc.hidden;
    if (moving && this.since === null) this.since = time;
    if (!moving) this.pause(time);
    if (this.doc.hidden) return; // drawn again on visibilitychange
    this.draw(time);
    if (!this.settled(time)) {
      this.frame = this.win.requestAnimationFrame(() => { this.frame = 0; this.schedule(); });
    } else if (moving) {
      this.timer = this.win.setTimeout(() => { this.timer = 0; this.schedule(); }, this.busy ? BUSY.interval : IDLE.interval);
    }
  }

  mesh(theme) {
    if (this.meshes[theme]) return this.meshes[theme];
    const [w, h] = MESH_SIZE;
    const canvas = this.doc.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(w, h);
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const { rgb, alpha } = meshColor(PALETTE[theme], (col + 0.5) / w, (row + 0.5) / h);
        const i = 4 * (row * w + col);
        image.data[i] = Math.round(rgb[0] * 255);
        image.data[i + 1] = Math.round(rgb[1] * 255);
        image.data[i + 2] = Math.round(rgb[2] * 255);
        image.data[i + 3] = Math.round(alpha * 255);
      }
    }
    ctx.putImageData(image, 0, 0);
    this.meshes[theme] = canvas;
    return canvas;
  }

  draw(time) {
    const width = Math.max(1, Math.round(this.canvas.clientWidth));
    const height = Math.max(1, Math.round(this.canvas.clientHeight));
    const bw = Math.max(1, Math.round(width * RESOLUTION));
    const bh = Math.max(1, Math.round(height * RESOLUTION));
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
    this.layer ??= this.doc.createElement("canvas");
    if (this.layer.width !== bw || this.layer.height !== bh) {
      this.layer.width = bw;
      this.layer.height = bh;
    }
    const theme = this.isDark() ? "dark" : "light";
    const palette = PALETTE[theme];
    const frame = fieldFrame(this.state(time), width, height);

    // The band: mesh, then the five lobes, then the top-to-bottom fade.
    const band = this.layer.getContext("2d");
    band.setTransform(1, 0, 0, 1, 0, 0);
    band.globalCompositeOperation = "source-over";
    band.globalAlpha = 1;
    band.clearRect(0, 0, bw, bh);
    band.setTransform(RESOLUTION, 0, 0, RESOLUTION, 0, 0);
    band.imageSmoothingEnabled = true;
    band.imageSmoothingQuality = "high";
    band.drawImage(this.mesh(theme), frame.mesh.cx - frame.mesh.width / 2, 0, frame.mesh.width, frame.bandHeight);
    for (const light of frame.lights) {
      const rgb = palette[light.color];
      band.save();
      band.translate(light.cx, light.cy);
      band.scale(light.rx, light.ry);
      const glow = band.createRadialGradient(0, 0, 0, 0, 0, 1);
      glow.addColorStop(0, rgba(rgb, 0.8));
      glow.addColorStop(0.5, rgba(rgb, 0.32));
      glow.addColorStop(1, rgba(rgb, 0));
      band.globalAlpha = light.opacity;
      band.fillStyle = glow;
      band.fillRect(-1, -1, 2, 2);
      band.restore();
    }
    band.globalCompositeOperation = "destination-in";
    const fade = band.createLinearGradient(0, 0, 0, Math.max(1, frame.bandHeight));
    fade.addColorStop(0, "rgba(0,0,0,1)");
    fade.addColorStop(0.15, "rgba(0,0,0,1)");
    fade.addColorStop(0.48, "rgba(0,0,0,0.6)");
    fade.addColorStop(1, "rgba(0,0,0,0)");
    band.fillStyle = fade;
    band.fillRect(0, 0, width, height);
    band.globalCompositeOperation = "source-over";

    // Hue drift and generation saturation apply to the whole band, like the Swift modifiers.
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, bw, bh);
    ctx.filter = `hue-rotate(${frame.hue.toFixed(2)}deg) saturate(${frame.saturation.toFixed(3)})`;
    ctx.drawImage(this.layer, 0, 0);
    ctx.filter = "none";
  }
}
