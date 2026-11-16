// Lab mode (Mac LabSample): one measurement per completed Chat/Agent answer, kept locally.
// Pure: the UI and the Node tests share it.

export const MAX_SAMPLES = 1000;

/** Metric id → [label key, unit, value of a sample]. Units match the Mac labels. */
export const METRICS = {
  decode: ["labDecode", "tok/s", (s) => s.decodeTps],
  prefill: ["labPrefill", "tok/s", (s) => s.prefillTps],
  ttft: ["labTtft", "ms", (s) => s.ttftSeconds * 1000],
  duration: ["labDuration", "s", (s) => s.elapsedSeconds],
  prompt: ["labPrompt", "tokens", (s) => s.promptTokens],
  generated: ["labGenerated", "tokens", (s) => s.generatedTokens],
};

const finite = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const rate = (tokens, seconds) => (seconds > 0 ? tokens / seconds : 0);

/**
 * Sample from a `done` event of the Windows engine (`prompt_tokens`, `generated_tokens`,
 * `cached_tokens`, `prefill_seconds`, `decode_seconds`, `wall_seconds`). `null` when the
 * Mac validity rule fails: a model, generated tokens > 0 and finite non-negative timings.
 */
export function labSample(done, meta, now = new Date()) {
  if (!done || !meta?.model) return null;
  const prompt = done.prompt_tokens;
  const generated = done.generated_tokens;
  const cached = finite(done.cached_tokens) ? done.cached_tokens : 0;
  const prefill = done.prefill_seconds;
  const decode = done.decode_seconds;
  if (!finite(prompt) || !finite(generated) || generated <= 0 || !finite(prefill) || !finite(decode)) return null;
  const sample = {
    id: `${now.getTime().toString(36)}${Math.random().toString(36).slice(2, 7)}`,
    date: now.toISOString(),
    model: String(meta.model).slice(0, 200),
    bits: finite(meta.bits) ? meta.bits : null,
    mode: meta.mode === "agent" ? "agent" : "chat",
    backend: String(meta.backend ?? ""),
    device: String(meta.device ?? ""),
    context: Number.isInteger(meta.context) ? meta.context : null,
    temperature: finite(meta.temperature) ? meta.temperature : null,
    topK: Number.isInteger(meta.topK) ? meta.topK : null,
    promptTokens: prompt,
    generatedTokens: generated,
    cachedTokens: cached,
    ttftSeconds: prefill,
    prefillTps: rate(Math.max(0, prompt - cached), prefill),
    decodeTps: rate(generated, decode),
    elapsedSeconds: finite(done.wall_seconds) ? done.wall_seconds : prefill + decode,
    toolRounds: Number.isInteger(done.tool_rounds) ? done.tool_rounds : 0,
  };
  return sample;
}

export function addSample(samples, sample) {
  return sample ? [...samples, sample].slice(-MAX_SAMPLES) : samples;
}

/** Stored samples; malformed entries drop. */
export function restoreSamples(raw) {
  let parsed;
  try { parsed = JSON.parse(raw); } catch { return []; }
  if (!Array.isArray(parsed)) return [];
  return parsed.filter((s) => s && typeof s.model === "string" && s.model && typeof s.date === "string"
    && !Number.isNaN(Date.parse(s.date)) && Object.values(METRICS).every(([, , value]) => finite(value(s))))
    .slice(-MAX_SAMPLES);
}

export function models(samples) {
  return [...new Set(samples.map((s) => s.model))].sort();
}

/** Mac "Latest decode speed by model": latest sample per model, fastest first, width in %. */
export function latestByModel(samples) {
  const latest = new Map();
  for (const s of samples) {
    const seen = latest.get(s.model);
    if (!seen || s.date >= seen.date) latest.set(s.model, s);
  }
  const rows = [...latest.values()].sort((a, b) => b.decodeTps - a.decodeTps);
  const top = rows[0]?.decodeTps || 0;
  return rows.map((s) => ({ model: s.model, value: s.decodeTps, percent: top > 0 ? Math.round((s.decodeTps / top) * 100) : 0 }));
}

/** Points of the history chart for `model` ("" = all) and `metric`, in date order. */
export function history(samples, model, metric) {
  const value = METRICS[metric]?.[2];
  if (!value) return [];
  return samples.filter((s) => !model || s.model === model)
    .map((s) => ({ date: Date.parse(s.date), model: s.model, value: value(s) }))
    .filter((p) => Number.isFinite(p.value))
    .sort((a, b) => a.date - b.date);
}

/** SVG coordinates for the points inside a `width`×`height` box with `pad` margins. */
export function chartPoints(points, width, height, pad = 24) {
  if (!points.length) return [];
  const xs = points.map((p) => p.date);
  const ys = points.map((p) => p.value);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const y1 = Math.max(...ys) || 1;
  return points.map((p) => ({
    ...p,
    x: x1 > x0 ? pad + ((p.date - x0) / (x1 - x0)) * (width - 2 * pad) : width / 2,
    y: height - pad - (p.value / y1) * (height - 2 * pad),
  }));
}

/** Mac export: a pretty-printed array with sorted keys. */
export function exportJson(samples) {
  const sorted = (value) => (Array.isArray(value) ? value.map(sorted)
    : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map((k) => [k, sorted(value[k])])) : value);
  return `${JSON.stringify(sorted(samples), null, 2)}\n`;
}
