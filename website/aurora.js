// The GPU samples the same four Desktop prints as one continuous sheet.
const vertexSource = `
  attribute vec2 position;
  varying vec2 uv;
  void main() {
    uv = position * vec2(0.5, -0.5) + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;
const fragmentSource = `
  precision highp float;
  varying vec2 uv;
  uniform sampler2D alphabet, flow, halo;
  uniform float time;
  uniform vec2 grid, cells;
  vec4 fluidAt(vec2 p) { return texture2D(flow,(p*(grid-1.0)+0.5)/grid); }
  vec2 lightUV(vec2 p, float t) {
    vec2 drift = vec2(0.014*(sin(4.0*p.y+0.44*t)-sin(4.0*p.y)),
                      0.028*(sin(5.0*p.x-0.49*t)-sin(5.0*p.x)));
    vec2 at = clamp(p+(drift-0.095*vec2(grid.y/grid.x,1.0)*((fluidAt(p).rg*255.0-128.0)/127.0))*smoothstep(vec2(0.0),vec2(0.12),min(p,1.0-p)),0.0,1.0);
    return at;
  }
  float glyph(float index, vec2 local) {
    return texture2D(alphabet,vec2((index+local.x)/7.0,local.y)).a;
  }
  void main() {
    vec2 p = uv;
    float t = time;
    vec2 cell = floor(p*cells);
    vec2 center = (cell+0.5)/cells;
    vec2 local = fract(p*cells);
    float arc = p.y + 0.065*sin(5.0*p.x + 0.24*t) + 0.035*sin(10.0*p.x - 0.31*t);
    float ribbon = 0.5 + 0.5*cos(10.5*arc - 0.82*t + 1.3*p.x);
    float glint = 0.5 + 0.5*sin(18.0*arc + 3.3*p.x - 0.58*t);
    float veil = 0.5 + 0.5*cos(4.7*p.x - 2.6*p.y + 0.37*t);
    float light = 0.77 + 0.26*ribbon + 0.12*glint*glint + 0.05*veil;
    vec3 emission = vec3(light + 0.08*ribbon*glint, light, light + 0.085*(1.0-ribbon)*veil);
    vec4 lamp = texture2D(halo,lightUV(p,t));
    float intensity = texture2D(halo,lightUV(center,t)).a/0.58;
    float wave = 0.5+0.5*cos(10.5*(center.y+0.065*sin(5.0*center.x+0.24*t))-0.82*t+1.3*center.x);
    float level = 6.0*clamp(intensity*(0.65+0.60*wave)+0.80*fluidAt(center).b,0.0,1.0);
    float index = floor(level);
    float mark = mix(glyph(index,local),glyph(min(6.0,index+1.0),local),smoothstep(0.32,0.68,fract(level)));
    float dye = 0.33*fluidAt(p).b;
    vec3 pigment = mix(vec3(0.36,0.17,0.70),vec3(0.83,0.25,0.36),p.y);
    float alpha = dye + lamp.a*(1.0-dye);
    vec3 color = pigment*dye + min(lamp.rgb*emission,vec3(lamp.a))*(1.0-dye);
    gl_FragColor = vec4(mix(color,vec3(alpha),0.45*mark*smoothstep(0.25,1.65,level)),alpha);
  }
`;

/** @param {WebGLRenderingContext} gl */
function createRenderer(gl) {
  const program = gl.createProgram();
  if (!program) return;
  for (const [type, source] of /** @type {const} */ ([
    [gl.VERTEX_SHADER, vertexSource],
    [gl.FRAGMENT_SHADER, fragmentSource],
  ])) {
    const shader = gl.createShader(type);
    if (!shader) {
      gl.deleteProgram(program);
      return;
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return;
  }
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  const texture = gl.createTexture();
  const motionTexture = gl.createTexture();
  const haloTexture = gl.createTexture();
  if (!buffer || !texture || !motionTexture || !haloTexture) {
    gl.deleteTexture(haloTexture);
    gl.deleteTexture(motionTexture);
    gl.deleteBuffer(buffer);
    gl.deleteTexture(texture);
    gl.deleteProgram(program);
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 3, -1, -1, 3]),
    gl.STATIC_DRAW,
  );
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  for (const [unit, tex] of /** @type {const} */ ([
    [gl.TEXTURE2, haloTexture],
    [gl.TEXTURE1, motionTexture],
    [gl.TEXTURE0, texture],
  ])) {
    gl.activeTexture(unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  const time = gl.getUniformLocation(program, "time");
  gl.uniform1i(gl.getUniformLocation(program, "flow"), 1);
  gl.uniform1i(gl.getUniformLocation(program, "halo"), 2);
  const grid = gl.getUniformLocation(program, "grid");
  const cells = gl.getUniformLocation(program, "cells");
  return {
    /** @param {HTMLImageElement} alphabet @param {HTMLCanvasElement} light @param {DOMRect} box */
    upload(alphabet, light, box) {
      gl.uniform2f(cells, box.width / 5, box.height / 8);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      for (const [unit, layer] of /** @type {const} */ ([
        [gl.TEXTURE0, alphabet],
        [gl.TEXTURE2, light],
      ])) {
        gl.activeTexture(unit);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          layer,
        );
      }
    },
    /** @param {FlowField} field @param {boolean} allocate */
    uploadFlow(field, allocate) {
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.activeTexture(gl.TEXTURE1);
      if (allocate) {
        gl.uniform2f(grid, field.width, field.height);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          field.width,
          field.height,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          field.pixels,
        );
      } else {
        gl.texSubImage2D(
          gl.TEXTURE_2D,
          0,
          0,
          0,
          field.width,
          field.height,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          field.pixels,
        );
      }
    },
    /** @param {number} phase */
    draw(phase) {
      gl.uniform1f(time, phase);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}

function mountAurora() {
  const element = document.querySelector("#aurora");
  if (!(element instanceof window.HTMLCanvasElement)) return;
  const canvas = element;
  const hero = canvas.parentElement;
  if (!hero) return;
  const gl = canvas.getContext("webgl", {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
  });
  let renderer = gl ? createRenderer(gl) : undefined;
  const plate = document.createElement("canvas");
  const ink = plate.getContext("2d");
  if (!ink) return;
  // This also keeps all four prints visible on a Mac with WebGL disabled or lost.
  plate.setAttribute("aria-hidden", "true");
  plate.className = "aurora-still";
  canvas.before(plate);
  const sprite = new window.Image();
  const light = new window.Image(),
    alphabet = new window.Image();
  const lightCanvas = document.createElement("canvas");
  const lightInk = lightCanvas.getContext("2d");
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let visible = true,
    frame = 0,
    previous = 0,
    phase = 0;
  /** @type {FlowField | undefined} */
  let field;
  /** @type {{x:number,y:number,id:number} | undefined} */
  let previousPoint;
  function resize() {
    if (!ink || !sprite.naturalWidth) return;
    const box = canvas.getBoundingClientRect();
    const w = Math.min(
      3072,
      Math.max(
        1,
        Math.round(box.width * Math.min(2, window.devicePixelRatio || 1)),
      ),
    );
    const h = Math.min(
      3072,
      Math.max(1, Math.round((w * box.height) / Math.max(1, box.width))),
    );
    canvas.width = plate.width = w;
    canvas.height = plate.height = h;
    lightCanvas.width = Math.max(1, Math.round(w / 2));
    lightCanvas.height = Math.max(1, Math.round(h / 2));
    const layersReady = Boolean(
      light.naturalWidth && alphabet.naturalWidth && lightInk,
    );
    for (const [cx, scale, hue, strength, cy, index] of [
      [0.045, 0.91, -38, 0.61, 0.64, 0],
      [0.77, 1.08, 0, 1, 0.64, 1],
      [1.05, 0.93, 9, 0.61, 0.64, 2],
      [0.34, 0.67, -26, 0.52, -0.03, 3],
    ]) {
      const height = h * 0.79 * scale,
        width = Math.min(w * 0.66, h * 0.79 * 2.1) * scale;
      const left = w * (cx + 0.026 * Math.sin(index)) - width / 2;
      const top = height * (cy + 0.035 * Math.sin(index) - 0.5);
      ink.globalAlpha = 0.58 * strength;
      ink.filter = hue ? `hue-rotate(${hue}deg)` : "none";
      ink.drawImage(sprite, left, top, width, height);
      if (layersReady && lightInk) {
        lightInk.setTransform(
          lightCanvas.width / w,
          0,
          0,
          lightCanvas.height / h,
          0,
          0,
        );
        lightInk.globalAlpha = ink.globalAlpha;
        lightInk.filter = ink.filter;
        lightInk.drawImage(light, left, top, width, height);
      }
    }
    if (renderer && layersReady) {
      const longest = Math.max(w, h);
      field = new FlowField(
        Math.max(8, Math.round((96 * w) / longest)),
        Math.max(8, Math.round((96 * h) / longest)),
      );
      renderer.upload(alphabet, lightCanvas, box);
      renderer.uploadFlow(field, true);
    }
    previousPoint = undefined;
    hero?.classList.add("ready");
    sync();
  }

  /** @param {number} now */
  function tick(now) {
    frame = 0;
    if (!renderer || document.hidden || !visible || motion.matches) return;
    const dt = previous ? Math.min(0.05, (now - previous) / 1000) : 0;
    previous = now;
    phase += dt;
    if (field?.step(dt)) renderer.uploadFlow(field, false);
    renderer.draw(phase);
    frame = window.requestAnimationFrame(tick);
  }

  function sync() {
    window.cancelAnimationFrame(frame);
    frame = 0;
    previous = 0;
    previousPoint = undefined;
    const still = motion.matches || !renderer || !field;
    plate.hidden = !still;
    canvas.style.opacity = still ? "0" : "1";
    if (motion.matches) {
      phase = 0;
      previousPoint = undefined;
      if (field?.active) {
        field = new FlowField(field.width, field.height);
        renderer?.uploadFlow(field, true);
      }
    }
    if (still || !sprite.naturalWidth || !visible || document.hidden) return;
    renderer?.draw(phase);
    frame = window.requestAnimationFrame(tick);
  }
  /** @param {PointerEvent} event */
  function point(event) {
    if (motion.matches || !renderer || !field || !event.isPrimary) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / Math.max(rect.width, 1);
    const y = (event.clientY - rect.top) / Math.max(rect.height, 1);
    if (previousPoint?.id === event.pointerId && event.type === "pointermove") {
      field.stir(previousPoint.x, previousPoint.y, x, y);
    }
    previousPoint = { x, y, id: event.pointerId };
  }
  hero.addEventListener("pointermove", point);
  hero.addEventListener("pointerdown", point);
  for (const kind of ["pointerleave", "pointercancel", "pointerup"])
    hero.addEventListener(kind, () => {
      previousPoint = undefined;
    });
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    renderer = undefined;
    sync();
  });
  canvas.addEventListener("webglcontextrestored", () => {
    renderer = gl ? createRenderer(gl) : undefined;
    resize();
  });
  new window.ResizeObserver(resize).observe(canvas);
  new window.IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    sync();
  }).observe(canvas);
  motion.addEventListener("change", sync);
  document.addEventListener("visibilitychange", sync);
  if (gl) {
    light.onload = alphabet.onload = resize;
    light.src = "assets/aurora-light.webp";
    alphabet.src = "assets/ascii-glyphs.png";
  }
  sprite.onload = resize;
  sprite.src = "assets/aurora.webp";
}
// Velocities live on the right and bottom faces of each fixed grid cell.
// Dye is a passive scalar carried by the projected velocity.
export class FlowField {
  /** @param {number} width @param {number} height */
  constructor(width, height) {
    this.width = width;
    this.height = height;
    const cells = width * height;
    this.u = new Float32Array(cells);
    this.v = new Float32Array(cells);
    this.dye = new Float32Array(cells);
    this.nextU = new Float32Array(cells);
    this.nextV = new Float32Array(cells);
    this.nextDye = new Float32Array(cells);
    this.pressure = new Float32Array(cells);
    this.divergence = new Float32Array(cells);
    this.pixels = new Uint8Array(cells * 4);
    this.active = false;
    this.pack();
  }

  /** @param {number} x0 @param {number} y0 @param {number} x1 @param {number} y1 */
  stir(x0, y0, x1, y1) {
    const w = this.width,
      h = this.height;
    const x = x0 * (w - 1),
      y = y0 * (h - 1);
    const vx = (x1 - x0) * (w - 1),
      vy = (y1 - y0) * (h - 1);
    const length = Math.hypot(vx, vy);
    if (!length) return;
    const radius = Math.max(2.2, Math.min(w, h) * 0.075);
    for (
      let j = Math.max(0, Math.floor(Math.min(y, y + vy) - radius));
      j <= Math.min(h - 1, y + Math.max(0, vy) + radius);
      j++
    ) {
      for (
        let i = Math.max(0, Math.floor(Math.min(x, x + vx) - radius));
        i <= Math.min(w - 1, x + Math.max(0, vx) + radius);
        i++
      ) {
        const along = Math.max(
          0,
          Math.min(1, ((i - x) * vx + (j - y) * vy) / (length * length)),
        );
        const r2 =
          ((i - x - along * vx) ** 2 + (j - y - along * vy) ** 2) /
          (radius * radius);
        if (r2 >= 1) continue;
        const k = j * w + i,
          gain = 18 * (1 - r2) ** 2;
        this.dye[k] =
          1 - (1 - this.dye[k]) * Math.exp(-0.28 * length * (1 - r2) ** 2);
        if (i < w - 1)
          this.u[k] = Math.max(-32, Math.min(32, this.u[k] + gain * vx));
        if (j < h - 1)
          this.v[k] = Math.max(-32, Math.min(32, this.v[k] + gain * vy));
      }
    }
    this.active = true;
  }

  /** @param {Float32Array} data @param {number} x @param {number} y */
  sample(data, x, y) {
    x = Math.max(0, Math.min(this.width - 1, x));
    y = Math.max(0, Math.min(this.height - 1, y));
    const i = Math.floor(x),
      j = Math.floor(y),
      k = j * this.width + i;
    const right = Math.min(i + 1, this.width - 1) + j * this.width;
    const down = j === this.height - 1 ? 0 : this.width;
    const fx = x - i,
      fy = y - j;
    return (
      (1 - fy) * (data[k] * (1 - fx) + data[right] * fx) +
      fy * (data[k + down] * (1 - fx) + data[right + down] * fx)
    );
  }

  project() {
    const { width: w, height: h, u, v, pressure: p, divergence } = this;
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        divergence[k] = u[k] - (i ? u[k - 1] : 0) + v[k] - (j ? v[k - w] : 0);
      }
    p.fill(0);
    // Gauss–Seidel on a closed box with zero normal flow at the outside faces.
    for (let pass = 0; pass < 16; pass++) {
      for (let j = 0; j < h; j++)
        for (let i = 0; i < w; i++) {
          const k = j * w + i;
          const neighbors =
            4 -
            Number(i === 0) -
            Number(j === 0) -
            Number(i === w - 1) -
            Number(j === h - 1);
          p[k] =
            ((i ? p[k - 1] : 0) +
              (i < w - 1 ? p[k + 1] : 0) +
              (j ? p[k - w] : 0) +
              (j < h - 1 ? p[k + w] : 0) -
              divergence[k]) /
            neighbors;
        }
    }
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        if (i < w - 1) u[k] -= p[k + 1] - p[k];
        if (j < h - 1) v[k] -= p[k + w] - p[k];
      }
  }

  /** @param {number} dt */
  step(dt) {
    if (!this.active || dt <= 0) return false;
    const { width: w, height: h } = this;
    const drag = Math.exp(-1.8 * dt),
      settle = Math.exp(-0.95 * dt);
    // Semi-Lagrangian backtraces sample the other face component at its staggered offset.
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        const vertical = this.sample(this.v, i + 0.5, j - 0.5);
        const horizontal = this.sample(this.u, i - 0.5, j + 0.5);
        this.nextU[k] =
          i === w - 1
            ? 0
            : drag * this.sample(this.u, i - dt * this.u[k], j - dt * vertical);
        this.nextV[k] =
          j === h - 1
            ? 0
            : drag *
              this.sample(this.v, i - dt * horizontal, j - dt * this.v[k]);
      }
    [this.u, this.nextU] = [this.nextU, this.u];
    [this.v, this.nextV] = [this.nextV, this.v];
    this.project();
    let peak = 0;
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        const vx = (this.u[k] + (i ? this.u[k - 1] : 0)) * 0.5,
          vy = (this.v[k] + (j ? this.v[k - w] : 0)) * 0.5;
        const x = i - dt * vx,
          y = j - dt * vy;
        this.nextDye[k] = settle * this.sample(this.dye, x, y);
        peak = Math.max(
          peak,
          Math.abs(this.u[k]),
          Math.abs(this.v[k]),
          this.nextDye[k],
        );
      }
    [this.dye, this.nextDye] = [this.nextDye, this.dye];
    if (peak < 0.002) {
      this.active = false;
      for (const field of [this.u, this.v, this.dye]) field.fill(0);
    }
    this.pack();
    return true;
  }

  pack() {
    for (let k = 0; k < this.dye.length; k++) {
      const u = (this.u[k] + (k % this.width ? this.u[k - 1] : 0)) * 0.5;
      const v =
        (this.v[k] + (k >= this.width ? this.v[k - this.width] : 0)) * 0.5;
      this.pixels[4 * k] =
        128 + Math.round(127 * Math.max(-1, Math.min(1, u / 32)));
      this.pixels[4 * k + 1] =
        128 + Math.round(127 * Math.max(-1, Math.min(1, v / 32)));
      this.pixels[4 * k + 2] = Math.round(255 * this.dye[k]);
      this.pixels[4 * k + 3] = 255;
    }
  }
}

if (typeof document !== "undefined") mountAurora();
