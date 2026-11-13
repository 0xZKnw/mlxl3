import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { after, before, test } from "node:test";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { extname } from "node:path";
import { chromium } from "playwright";
import { FlowField } from "./aurora.js";
import AxeBuilder from "@axe-core/playwright";
const root = new URL("./", import.meta.url);
const command = "curl -fsSL https://mlxl3.0xzknw.tech/install.sh | bash";
let server, browser, origin;
before(async () => {
  server = createServer(async (req, res) => {
    try {
      const path = new URL(req.url, "http://localhost").pathname;
      const name = path === "/" ? "index.html" : path.slice(1);
      const data = await readFile(new URL(name, root));
      const mime = {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".webp": "image/webp",
        ".woff2": "font/woff2",
        ".png": "image/png",
        ".sh": "text/plain",
      };
      res.writeHead(200, {
        "Content-Type": mime[extname(name)] ?? "text/plain",
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ args: ["--enable-unsafe-swiftshader"] });
});
after(async () => {
  await browser?.close();
  if (server?.listening) await new Promise((r) => server.close(r));
});
async function visit(options, fn) {
  const context = await browser.newContext({
    reducedMotion: "reduce",
    ...options,
  });
  const page = await context.newPage();
  try {
    await page.goto(origin);
    await page.evaluate(() => document.fonts.ready);
    await fn(page);
  } finally {
    await context.close();
  }
}
test("one screen, visible controls and local assets across 10 viewport and text-size combinations", async () => {
  await visit({}, async (p) => {
    for (const [width, height] of [
      [320, 568],
      [390, 844],
      [667, 375],
      [1024, 768],
      [1440, 900],
    ]) {
      await p.setViewportSize({ width, height });
      for (const zoom of [100, 180]) {
        await p.evaluate((v) => {
          document.documentElement.style.fontSize = v + "%";
        }, zoom);
        const state = await p.evaluate(() => ({
          width: document.documentElement.scrollWidth,
          height: document.documentElement.scrollHeight,
          rects: [...document.querySelectorAll("section,.command,footer")].map(
            (el) => JSON.parse(JSON.stringify(el.getBoundingClientRect())),
          ),
        }));
        assert.ok(
          state.width <= width && state.height <= height,
          JSON.stringify({ width, height, zoom, state }),
        );
        for (const rect of state.rects)
          assert.ok(
            rect.left >= -1 &&
              rect.right <= width + 1 &&
              rect.top >= -1 &&
              rect.bottom <= height + 1,
          );
        const [section, , footer] = state.rects;
        assert.ok(section.top >= 0 && section.bottom <= footer.top + 1);
      }
      await p.evaluate(() => {
        document.documentElement.style.fontSize = "100%";
      });
      if (process.env.MLXL3_SCREENSHOTS && [390, 1440].includes(width)) {
        await mkdir(process.env.MLXL3_SCREENSHOTS, { recursive: true });
        await p.screenshot({
          path: `${process.env.MLXL3_SCREENSHOTS}/fullscreen-${width}.png`,
        });
      }
    }
    for (const name of ["main.js", "aurora.js", "styles.css"]) {
      const hash = createHash("sha256")
        .update(await readFile(new URL(name, root)))
        .digest("hex")
        .slice(0, 12);
      assert.equal(
        await p
          .locator(`[src^="${name}"],[href^="${name}"]`)
          .getAttribute(name.endsWith(".css") ? "href" : "src"),
        name + "?v=" + hash,
      );
    }
    assert.equal(await p.locator("footer svg").count(), 2);
    for (const [name, url] of [
      ["GitHub", "https://github.com/0xZKnw/mlxl3"],
      ["Discord", "https://discord.gg/9hU46NVNnY"],
    ])
      assert.equal(
        await p.getByRole("link", { name, exact: true }).getAttribute("href"),
        url,
      );
    assert.deepEqual(
      (await new AxeBuilder({ page: p }).analyze()).violations.map((v) => v.id),
      [],
    );
  });
});
test("copy exactly the stable install command, including from the keyboard", async () => {
  await visit(
    { permissions: ["clipboard-read", "clipboard-write"] },
    async (p) => {
      await p.getByRole("button", { name: "Copy install command" }).focus();
      await p.keyboard.press("Enter");
      await p.waitForFunction(
        () => document.querySelector("#status").textContent === "Copied.",
      );
      assert.equal(
        await p.evaluate(() => navigator.clipboard.readText()),
        command,
      );
      const response = await p.request.get(origin + "/install.sh");
      assert.equal(response.status(), 200);
      assert.match(await response.text(), /--channel stable or --channel beta/);
    },
  );
});
test("unavailable clipboard selects the full command; no-JS page keeps links and command", async () => {
  await visit({}, async (p) => {
    await p.evaluate(() =>
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText: () => Promise.reject(new Error("denied")) },
        configurable: true,
      }),
    );
    await p.locator("#copy").click();
    await p.waitForFunction(() =>
      document.querySelector("#status").textContent.includes("selected"),
    );
    assert.equal(
      await p.evaluate(() => window.getSelection().toString()),
      command,
    );
  });
  await visit({ javaScriptEnabled: false }, async (p) => {
    assert.equal(await p.locator("#copy").isVisible(), false);
    assert.equal(await p.locator("code").textContent(), command);
    assert.ok(await p.getByRole("link", { name: "Discord" }).isVisible());
  });
});
test("continuous waves, local pointer response and motion preferences use the real renderer", async () => {
  await visit({ viewport: { width: 900, height: 600 } }, async (p) => {
    await p.locator("main.ready").waitFor();
    const plate = p.locator(".aurora-still");
    assert.ok(await plate.isVisible());
    assert.equal(
      await p
        .locator(".command")
        .innerText()
        .then((text) => /[❯>]/.test(text)),
      false,
    );
    const original = await plate.evaluate((c) => c.toDataURL());
    await p.evaluate(() => {
      const gl = document.querySelector("#aurora").getContext("webgl");
      if (!gl) throw new Error("WebGL required for animated-path test");
      const pending = new Map();
      let id = 0,
        draws = 0,
        sample;
      window.requestAnimationFrame = (callback) => {
        pending.set(++id, callback);
        return id;
      };
      window.cancelAnimationFrame = (id) => pending.delete(id);
      const draw = gl.drawArrays.bind(gl);
      gl.drawArrays = (...args) => {
        draw(...args);
        draws++;
        if (!sample) return;
        // Read inside the draw, before the compositor discards its buffer.
        for (const x of [50, 655]) {
          const data = new Uint8Array(130 * 100 * 4);
          gl.readPixels(x, 150, 130, 100, gl.RGBA, gl.UNSIGNED_BYTE, data);
          let hash = 2166136261,
            visible = 0;
          for (const value of data) {
            hash = Math.imul(hash ^ value, 16777619);
            if (value > 5) visible++;
          }
          sample.push({ hash, visible });
        }
      };
      window.advance = (frames) => {
        for (let step = 0; step < frames; step++) {
          sample = step === frames - 1 ? [] : undefined;
          const callbacks = [...pending.values()];
          pending.clear();
          for (const callback of callbacks) callback(1000 + step * 16);
        }
        return { sample, draws, pending: pending.size };
      };
    });
    const advance = (frames) => p.evaluate((n) => window.advance(n), frames);
    const motion = async (reduced) => {
      await p.emulateMedia({
        reducedMotion: reduced ? "reduce" : "no-preference",
      });
      await p.waitForFunction(
        (value) => document.querySelector(".aurora-still").hidden === value,
        !reduced,
        { polling: 10 },
      );
    };
    await p.mouse.move(720, 400);
    assert.equal((await advance(8)).draws, 0);
    assert.equal(await plate.evaluate((c) => c.toDataURL()), original);
    await motion(false);
    const start = (await advance(1)).sample;
    const idle = (await advance(46)).sample;
    assert.ok(idle.every((region) => region.visible > 100));
    assert.notEqual(start[1].hash, idle[1].hash);
    await motion(true);
    assert.equal((await advance(8)).pending, 0);
    assert.equal(await plate.evaluate((c) => c.toDataURL()), original);
    await motion(false);
    await p.mouse.move(660, 425);
    await p.mouse.move(750, 395, { steps: 8 });
    await p.mouse.move(899, 400);
    await advance(1);
    const ripple = (await advance(46)).sample;
    assert.ok(ripple[0].visible > 100);
    assert.notEqual(idle[1].hash, ripple[1].hash);
    await p.evaluate(() => {
      window.contextLoss = document
        .querySelector("#aurora")
        .getContext("webgl")
        .getExtension("WEBGL_lose_context");
      window.contextLoss.loseContext();
    });
    await p.waitForFunction(
      () => !document.querySelector(".aurora-still").hidden,
      undefined,
      { polling: 10 },
    );
    assert.equal((await advance(8)).pending, 0);
    assert.equal(await plate.evaluate((c) => c.toDataURL()), original);
    await p.evaluate(() => window.contextLoss.restoreContext());
    await p.waitForFunction(
      () => document.querySelector(".aurora-still").hidden,
      undefined,
      { polling: 10 },
    );
    assert.ok((await advance(8)).sample[1].visible > 100);
  });
});
test("without WebGL the textured background and installer remain usable", async () => {
  await visit({ reducedMotion: "no-preference" }, async (p) => {
    await p.addInitScript(() => {
      const get = window.HTMLCanvasElement.prototype.getContext;
      window.HTMLCanvasElement.prototype.getContext = function (type, ...args) {
        return type === "webgl" ? null : get.call(this, type, ...args);
      };
    });
    await p.reload();
    await p.locator("main.ready").waitFor();
    const plate = p.locator(".aurora-still");
    assert.ok(await plate.isVisible());
    assert.ok(
      await plate.evaluate((c) =>
        c
          .getContext("2d")
          .getImageData(0, 0, c.width, c.height)
          .data.some((v) => v > 90),
      ),
    );
    assert.ok(
      await p.getByRole("button", { name: "Copy install command" }).isVisible(),
    );
  });
});

test("Eulerian grid projects velocity, transports passive dye and dissipates after release", () => {
  for (const [w, h] of [
    [8, 16],
    [32, 20],
    [64, 40],
    [20, 64],
  ]) {
    const f = new FlowField(w, h);
    assert.equal(f.step(1 / 60), false);
    const flux = () => {
      let squares = 0;
      for (let j = 0; j < h; j++)
        for (let i = 0; i < w; i++) {
          const k = j * w + i;
          squares +=
            (f.u[k] - (i ? f.u[k - 1] : 0) + f.v[k] - (j ? f.v[k - w] : 0)) **
            2;
        }
      return squares;
    };
    f.stir(0.25, 0.65, 0.77, 0.52);
    for (let row = 0; row < h; row++) {
      if (
        row / (h - 1) < 0.52 - 2.2 / (h - 1) - 0.075 ||
        row / (h - 1) > 0.65 + 2.2 / (h - 1) + 0.075
      )
        assert.ok(f.dye.slice(row * w, row * w + w).every((v) => v === 0));
    }
    const before = flux();
    f.project();
    assert.ok(before > 1 && flux() < before * 0.28);
    for (let i = 0; i < 16; i++) f.step(1 / 60);
    assert.ok(Math.max(...f.dye) > 0.1);
    assert.ok(f.dye.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
    // In a uniform rightward current a compact dye spot must move downstream.
    const tracer = new FlowField(32, 20);
    tracer.active = true;
    tracer.u.fill(7);
    tracer.dye[9 * 32 + 12] = 1;
    tracer.step(1 / 30);
    assert.ok(tracer.dye[9 * 32 + 13] > 0.1 && tracer.dye[9 * 32 + 11] === 0);
    for (let j = 0; j < h; j++) assert.equal(f.u[(j + 1) * w - 1], 0);
    for (let i = 0; i < w; i++) assert.equal(f.v[(h - 1) * w + i], 0);
    for (let step = 0; step < 750 && f.active; step++) f.step(1 / 60);
    assert.equal(f.active, false);
    assert.ok(f.dye.every((v) => v === 0));
    assert.deepEqual([...f.pixels.slice(0, 4)], [128, 128, 0, 255]);
  }
});

test("the light travels while ASCII changes shape on fixed Retina cells", async () => {
  await visit(
    {
      viewport: { width: 900, height: 600 },
      deviceScaleFactor: 2,
      reducedMotion: "no-preference",
    },
    async (p) => {
      // Seven alternating left/right bars make cell alignment and character changes observable.
      const fixtures = await p.evaluate(() => {
        const c = document.createElement("canvas");
        c.width = 140;
        c.height = 32;
        const ctx = c.getContext("2d");
        ctx.fillStyle = "white";
        for (let i = 0; i < 7; i++)
          ctx.fillRect(20 * i + (i % 2 ? 12 : 4), 8, 4, 16);
        const atlas = c.toDataURL().split(",")[1];
        c.width = 128;
        c.height = 256;
        ctx.fillStyle = "#f00000";
        ctx.fillRect(26, 125, 76, 6);
        ctx.globalAlpha = 0.59;
        ctx.fillRect(26, 164, 76, 65);
        return [atlas, c.toDataURL().split(",")[1]];
      });
      for (const [asset, source] of [
        ["ascii-glyphs.png", fixtures[0]],
        ["aurora-light.webp", fixtures[1]],
        ["aurora.webp", fixtures[1]],
      ]) {
        await p.route("**/assets/" + asset, (route) =>
          route.fulfill({
            contentType: "image/png",
            body: Buffer.from(source, "base64"),
          }),
        );
      }
      await p.addInitScript(() => {
        const proto = window.WebGLRenderingContext.prototype;
        const uniform = proto.uniform1f,
          draw = proto.drawArrays;
        proto.uniform1f = function (loc, value) {
          uniform.call(this, loc, window.testPhase ?? value);
        };
        proto.drawArrays = function (...args) {
          draw.apply(this, args);
          if (!window.captureCells) return;
          window.captureCells = false;
          const h = this.drawingBufferHeight,
            left = 1380,
            width = 80;
          const data = new Uint8Array(h * width * 4);
          this.readPixels(
            left,
            0,
            width,
            h,
            this.RGBA,
            this.UNSIGNED_BYTE,
            data,
          );
          const at = (x, y, ch) => data[((h - y - 1) * width + x) * 4 + ch];
          let ink = 0,
            escaped = 0;
          const alpha = [],
            signatures = [];
          for (let y = 0; y < h; y++) {
            alpha.push(at(40, y, 3));
            for (let x = 0; x < width; x++) {
              const g = at(x, y, 1);
              if (g <= 3) continue;
              ink++;
              const sx = x % 10,
                sy = y % 16;
              if (
                (sx !== 2 && sx !== 3 && sx !== 6 && sx !== 7) ||
                sy < 4 ||
                sy > 11
              )
                escaped++;
            }
          }
          for (let y = 8; y < h; y += 16)
            for (let x = 0; x < width; x += 10) {
              const a = at(x + 2, y, 1),
                b = at(x + 6, y, 1);
              signatures.push(a + b > 9 ? a / (a + b) : -1);
            }
          window.cellSample = { alpha, ink, escaped, signatures };
        };
      });
      await p.reload();
      await p.waitForFunction(
        () =>
          document.querySelector("main.ready") &&
          document.querySelector(".aurora-still").hidden,
      );
      async function sample(t) {
        await p.evaluate((phase) => {
          window.cellSample = null;
          window.testPhase = phase;
          window.captureCells = true;
        }, t);
        await p.waitForFunction(() => window.cellSample);
        return p.evaluate(() => window.cellSample);
      }
      const first = await sample(0),
        later = await sample(3);
      const crest = (a) => a.indexOf(Math.max(...a));
      assert.equal(first.alpha.length, 1200);
      const old = crest(first.alpha);
      assert.ok(
        first.alpha[old] > 60 && Math.abs(crest(later.alpha) - old) > 30,
      );
      assert.ok(
        later.alpha[old] < first.alpha[old] * 0.04,
        "the old crest should clear",
      );
      for (const sample of [first, later]) {
        assert.ok(
          sample.ink > 700,
          "render enough of the real grid to check its borders",
        );
        assert.equal(
          sample.escaped,
          0,
          "glyph coverage escaped its fixed atlas shape",
        );
      }
      const changed = first.signatures.filter(
        (v, i) =>
          v >= 0 &&
          later.signatures[i] >= 0 &&
          Math.abs(v - later.signatures[i]) > 0.7,
      ).length;
      assert.ok(
        changed > 12,
        "passing light must replace the glyph within the same cell",
      );
    },
  );
});
