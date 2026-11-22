// Appearance preferences, mirroring the Mac app's DesktopPreferences / DesktopExperience enums.
export const ACCENTS = {
  violet: { light: "#6642A4", dark: "#BBA4F3" },
  rose: { light: "#97345D", dark: "#E89BB9" },
  amber: { light: "#8E4B18", dark: "#F0B782" },
  neutral: { light: "rgb(46 46 46)", dark: "rgb(219 219 219)" },
};
export const DEFAULT_CUSTOM = "BBA4F3";
export const READING_WIDTHS = { narrow: [620, 590], centered: [820, 790], wide: [null, null] };
export const DENSITIES = { tight: 12, compact: 22, comfortable: 38, spacious: 54, extraSpacious: 72 };
export const FONTS = {
  standard: "var(--sans)",
  rounded: 'ui-rounded, "Segoe UI Variable Display", var(--sans)',
  serif: "var(--serif)",
  monospaced: "var(--mono)",
};
export const BUBBLES = ["subtle", "outlined", "tinted"];
// Window background: the v2 ASCII print, or the soft animated aurora of the 2.0 betas (Windows only).
export const BACKGROUNDS = ["ascii", "soft"];
export const SIDEBAR_WIDTHS = { compact: 208, standard: 232, wide: 280 };
export const TEXT_SIZE = { min: 12, max: 24, default: 15 };
export const IDLE_DELAYS = { never: 0, "1m": 60, "5m": 300, "10m": 600, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200 };

export const DEFAULT_APPEARANCE = Object.freeze({
  accent: "violet", custom: DEFAULT_CUSTOM, width: "centered", textSize: TEXT_SIZE.default,
  density: "comfortable", font: "standard", bubble: "subtle", sidebar: "standard", background: "ascii",
});

const pick = (value, allowed, fallback) => (Object.hasOwn(allowed, value) ? value : fallback);

export function normalizeHex(value) {
  const text = String(value ?? "").replace(/^#/, "");
  return /^[0-9a-fA-F]{6}$/.test(text) ? text.toUpperCase() : DEFAULT_CUSTOM;
}

export function normalizeTextSize(value) {
  const number = Number(value);
  if (typeof value === "boolean" || value === null || value === "" || !Number.isFinite(number)) return TEXT_SIZE.default;
  return Math.min(TEXT_SIZE.max, Math.max(TEXT_SIZE.min, Math.round(number)));
}

// Accepts anything read back from storage and returns a complete, valid appearance.
export function normalizeAppearance(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const legacy = { sage: "neutral", mint: "neutral", coral: "rose", custom: "custom" };
  const accent = Object.hasOwn(legacy, source.accent) ? legacy[source.accent] : pick(source.accent, ACCENTS, DEFAULT_APPEARANCE.accent);
  return {
    accent,
    custom: normalizeHex(source.custom),
    width: pick(source.width, READING_WIDTHS, DEFAULT_APPEARANCE.width),
    textSize: normalizeTextSize(source.textSize),
    density: pick(source.density, DENSITIES, DEFAULT_APPEARANCE.density),
    font: pick(source.font, FONTS, DEFAULT_APPEARANCE.font),
    bubble: BUBBLES.includes(source.bubble) ? source.bubble : DEFAULT_APPEARANCE.bubble,
    sidebar: pick(source.sidebar, SIDEBAR_WIDTHS, DEFAULT_APPEARANCE.sidebar),
    background: BACKGROUNDS.includes(source.background) ? source.background : DEFAULT_APPEARANCE.background,
  };
}

// CSS custom properties for an appearance. The tint has a light and a dark value; the stylesheet
// selects one per theme, so a system theme change needs no JavaScript.
export function cssVariables(appearance) {
  const a = normalizeAppearance(appearance);
  const tint = a.accent === "custom" ? { light: `#${a.custom}`, dark: `#${a.custom}` } : ACCENTS[a.accent];
  const [message, composer] = READING_WIDTHS[a.width];
  return {
    "--tint-light": tint.light,
    "--tint-dark": tint.dark,
    "--message-max": message ? `${message}px` : "none",
    "--composer-max": composer ? `${composer}px` : "none",
    "--text-size": `${a.textSize}px`,
    "--turn-gap": `${DENSITIES[a.density]}px`,
    "--message-font": FONTS[a.font],
    "--sidebar-width": `${SIDEBAR_WIDTHS[a.sidebar]}px`,
  };
}

export function applyAppearance(root, appearance) {
  const a = normalizeAppearance(appearance);
  for (const [name, value] of Object.entries(cssVariables(a))) root.style.setProperty(name, value);
  root.dataset.bubble = a.bubble;
  root.dataset.background = a.background;
  return a;
}

export function idleSeconds(key) {
  return Object.hasOwn(IDLE_DELAYS, key) ? IDLE_DELAYS[key] : IDLE_DELAYS["15m"];
}
