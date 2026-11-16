// The Mac app's "StudioAurora" background: a violet column of light printed in a small
// ASCII alphabet, drawn as four hue-shifted curtains behind the whole window.
// `light` is a direct port of the Swift glow function; the plate uses the same grid,
// glyph set and colour mix. Static (the Mac's still print), redrawn on resize/theme.

export const PLATE_WIDTH = 740;
export const PLATE_HEIGHT = 620;
const ALPHABET = ".:-=+xX";
const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Glow colour (0–1 per channel) and alpha at x ∈ [-1, 1], y ∈ [0, 1] (top to bottom). */
export function light(x, y) {
  const foot = 0.70 + 0.20 * x * x + 0.027 * Math.sin(4.5 * x);
  const d = foot - y;
  const span = 0.25 + 0.16 * Math.max(0, d);
  const column = Math.exp(-((x / span) ** 2) - ((d - 0.22) / 0.43) ** 2);
  const rim = Math.exp(-((x / 0.64) ** 2) - (d / 0.064) ** 2);
  const haze = Math.exp(-((x / 0.66) ** 2) - ((d - 0.17) / 0.46) ** 2);
  const hot = Math.exp(-((x / 0.32) ** 2) - ((d - 0.055) / 0.10) ** 2);
  const pink = Math.exp(-(((d - 0.14) / 0.19) ** 2));
  const rgb = [
    clamp01(0.20 + 0.78 * pink + 0.64 * rim + 0.30 * hot),
    clamp01(0.022 + 0.10 * Math.max(0, d) + 0.61 * hot),
    clamp01(0.98 - 0.55 * rim - 0.61 * hot),
  ];
  const fade = clamp01(y * 8) * clamp01((1 - Math.abs(x)) * 8) * clamp01((1 - y) * 9);
  const alpha = Math.min(1, 0.91 * column + 0.94 * rim + 0.24 * haze) * fade
    * Math.exp(-((Math.max(0, -d) / 0.09) ** 2));
  return { rgb, alpha };
}

/** Glyph index for one plate cell, or -1 when the cell stays empty. */
export function cellGlyph(col, row) {
  const x = col * 5 + 2;
  const y = row * 8 + 5;
  const { alpha } = light((2 * x) / PLATE_WIDTH - 1, y / PLATE_HEIGHT);
  if (!(alpha > 0.13)) return -1;
  const grain = Math.sin(col * 17 + row * 31) * 0.06;
  const density = clamp01(alpha + 0.13 * Math.sin(y * 0.09) + grain);
  return Math.floor(density * (ALPHABET.length - 1));
}

const CURTAINS = [
  { x: 0.045, scale: 0.91, hue: -38, strength: 0.61, y: 0.64 },
  { x: 0.77, scale: 1.08, hue: 0, strength: 1, y: 0.64 },
  { x: 1.05, scale: 0.93, hue: 9, strength: 0.61, y: 0.64 },
  { x: 0.34, scale: 0.67, hue: -26, strength: 0.52, y: -0.03 },
];

/** Where each curtain lands in a view of `width` × `height` (centre and size, in CSS px). */
export function curtainFrames(width, height) {
  const plateHeight = height * 0.79;
  return CURTAINS.map((c, i) => {
    const w = Math.min(width * 0.66, plateHeight * 2.1) * c.scale;
    const h = plateHeight * c.scale;
    return {
      cx: width * (c.x + 0.026 * Math.sin(i)),
      cy: h * (c.y + 0.035 * Math.sin(i)),
      width: w, height: h, hue: c.hue, strength: c.strength,
    };
  });
}

let plateCache = null;

function plate(document) {
  if (plateCache) return plateCache;
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = PLATE_WIDTH * scale;
  canvas.height = PLATE_HEIGHT * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  // Soft glow underneath (half resolution, like the Mac bitmap).
  const gw = PLATE_WIDTH / 2;
  const gh = PLATE_HEIGHT / 2;
  const glow = document.createElement("canvas");
  glow.width = gw;
  glow.height = gh;
  const gctx = glow.getContext("2d");
  const image = gctx.createImageData(gw, gh);
  for (let row = 0; row < gh; row++) {
    for (let col = 0; col < gw; col++) {
      const { rgb, alpha } = light((2 * (col + 0.5)) / gw - 1, (row + 0.5) / gh);
      const i = 4 * (row * gw + col);
      image.data[i] = rgb[0] * 255;
      image.data[i + 1] = rgb[1] * 255;
      image.data[i + 2] = rgb[2] * 255;
      image.data[i + 3] = alpha * 255;
    }
  }
  gctx.putImageData(image, 0, 0);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(glow, 0, 0, PLATE_WIDTH, PLATE_HEIGHT);
  ctx.font = '7px Consolas, "Cascadia Mono", Menlo, monospace';
  ctx.textBaseline = "alphabetic";
  for (let row = 0; row < PLATE_HEIGHT / 8; row++) {
    for (let col = 0; col < Math.floor(PLATE_WIDTH / 5); col++) {
      const glyph = cellGlyph(col, row);
      if (glyph < 0) continue;
      const x = col * 5 + 2;
      const y = row * 8 + 5;
      const { rgb, alpha } = light((2 * x) / PLATE_WIDTH - 1, y / PLATE_HEIGHT);
      const r = Math.round((rgb[0] * 0.60 + 0.40) * 255);
      const g = Math.round((rgb[1] * 0.58 + 0.42) * 255);
      const b = Math.round((rgb[2] * 0.62 + 0.38) * 255);
      ctx.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.47).toFixed(3)})`;
      ctx.fillText(ALPHABET[glyph], x, y + 2);
    }
  }
  plateCache = canvas;
  return canvas;
}

/** Paint the aurora into `canvas`, sized to its CSS box. */
export function paint(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const source = plate(canvas.ownerDocument);
  for (const frame of curtainFrames(width, height)) {
    ctx.save();
    ctx.globalAlpha = frame.strength;
    ctx.filter = frame.hue ? `hue-rotate(${frame.hue}deg)` : "none";
    ctx.drawImage(source, frame.cx - frame.width / 2, frame.cy - frame.height / 2, frame.width, frame.height);
    ctx.restore();
  }
}
