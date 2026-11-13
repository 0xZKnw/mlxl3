import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { Buffer } from "node:buffer";
import {
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
} from "node:timers";
import {
  mkdtemp,
  mkdir,
  writeFile,
  readFile,
  rm,
  lstat,
  readlink,
  access,
  rename,
  readdir,
  symlink,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { test } from "node:test";

const installer = new URL("./install.sh", import.meta.url).pathname;
const archive = "official fixture disk image\n";
const digest = createHash("sha256").update(archive).digest("hex");
const copy = (value) => JSON.parse(JSON.stringify(value));
function release(tag, prerelease, date, build = 33) {
  const name = `MLXL3-Desktop-${tag}-b${build}-Apple-Silicon.dmg`;
  return {
    tag_name: tag,
    prerelease,
    draft: false,
    published_at: date,
    assets: [
      {
        name,
        size: Buffer.byteLength(archive),
        digest: `sha256:${digest}`,
        browser_download_url: `https://github.com/0xZKnw/mlxl3/releases/download/${tag}/${name}`,
      },
    ],
  };
}
const stable = release("v1.5.1", false, "2026-10-26T13:51:47Z");
const beta = release("v2.0.0-beta.4", true, "2026-10-27T09:34:33Z", 35);
const older = release("v1.4.0", false, "2026-10-01T09:00:00Z", 23);
const releases = [
  older,
  beta,
  {
    tag_name: "engine-v9.0.0",
    prerelease: false,
    draft: false,
    published_at: "2026-10-27T10:00:00Z",
    assets: [],
  },
  { ...release("v9.0.0", false, "2026-10-27T10:00:00Z"), draft: true },
  stable,
];

// Substitute only external OS/network commands; install.sh and file operations run unchanged.
const stub = `
const fs = require('node:fs');
const cp = require('node:child_process');
const path = require('node:path');
const config = JSON.parse(fs.readFileSync(process.env.MLXL3_TEST_CONFIG));
const name = path.basename(process.argv[1]);
const args = process.argv.slice(2);
fs.appendFileSync(config.log, JSON.stringify({ name, args }) + '\\n');
const fail = () => process.exit(1);
if (name === 'uname') console.log(args[0] === '-s' ? (config.os ?? 'Darwin') : (config.arch ?? 'arm64'));
else if (name === 'sw_vers') console.log(config.version ?? '26.2');
else if (name === 'find') { if (config.globalApp) console.log(config.globalApp); }
else if (name === 'pgrep') process.exit(config.running || fs.existsSync(config.state + '.running') ? 0 : 1);
else if (name === 'plutil') {
  try {
    let value = JSON.parse(fs.readFileSync(args.at(-1)));
    for (const key of args[args.indexOf('-extract') + 1].split('.')) value = value[key];
    const expected = args.includes('-expect') ? args[args.indexOf('-expect') + 1] : null;
    if (value === undefined || value === null || (expected === 'array' && !Array.isArray(value)) || (expected === 'string' && typeof value !== 'string') || (expected === 'bool' && typeof value !== 'boolean') || (expected === 'integer' && !Number.isInteger(value))) fail();
    console.log(Array.isArray(value) ? value.length : typeof value === 'object' ? JSON.stringify(value) : String(value));
  } catch { fail(); }
} else if (name === 'curl') {
  const out = args[args.indexOf('-o') + 1];
  if (args.some(arg => arg.startsWith('https://api.github.com/'))) {
    if (config.fail === 'api') process.exit(22);
    const url = args.find(arg => arg.startsWith('https://api.github.com/'));
    const page = Number(new URL(url).searchParams.get('page') ?? 1);
    fs.writeFileSync(out, config.response ?? JSON.stringify(config.releases.slice((page-1)*100, page*100)));
  } else {
    if (config.fail === 'download') process.exit(22);
    const selected = config.releases.find(item => item.assets?.some(asset => args.includes(asset.browser_download_url)));
    if (!selected) fail();
    fs.writeFileSync(config.state, JSON.stringify(selected));
    if (config.fail === 'silence') setTimeout(() => {}, 60000);
    else fs.writeFileSync(out, config.corrupt ? 'x'.repeat(Buffer.byteLength(config.archive)) : config.archive);
  }
} else if (name === 'hdiutil') {
  if (args[0] === 'detach') { if (config.fail === 'detach') fail(); }
  else {
    if (config.fail === 'mount') fail();
    const mount = args[args.indexOf('-mountpoint') + 1];
    const selected = JSON.parse(fs.readFileSync(config.state));
    const app = path.join(mount, 'MLXL3 Desktop.app/Contents');
    fs.mkdirSync(path.join(app, 'Resources/runtime'), { recursive: true });
    fs.mkdirSync(path.join(app, 'MacOS'), { recursive: true });
    fs.writeFileSync(path.join(app, 'Info.plist'), JSON.stringify({CFBundleIdentifier: config.bundleId ?? 'io.mlxl3.desktop', CFBundleShortVersionString: config.bundleVersion ?? selected.tag_name.slice(1).split('-')[0], CFBundleVersion: selected.assets[0].name.match(/-b([0-9]+)-/)[1]}));
    for (const file of ['MacOS/MLXL3Studio', 'Resources/runtime/mlxl3']) {
      if (config.fail === 'missing-cli' && file.endsWith('/mlxl3')) continue;
      fs.writeFileSync(path.join(app, file), config.fail === 'cli-smoke' && file.endsWith('/mlxl3') ? '#!/bin/sh\\nexit 1\\n' : '#!/bin/sh\\nprintf "fixture CLI\\\\n"\\n', {mode: 0o755});
    }
  }
} else if (name === 'ditto') {
  if (config.fail === 'copy') fail();
  fs.cpSync(args[0], args[1], {recursive: true});
} else if (name === 'codesign') { if (config.fail === 'signature') fail(); if (config.startDuringVerify) fs.writeFileSync(config.state + '.running', ''); }
else if (name === 'mv' || name === 'ln' || name === 'rm') {
  if(name==='rm'&&config.fail==='cleanup-stage'&&args.some(arg=>arg.includes('/Applications/.mlxl3-install.')))fail();
  const cli = path.join(process.env.HOME, '.local/bin/mlxl3');
  if (name === 'mv' && ((config.fail === 'cli-backup' && args[0] === cli) || (config.failRestoreApp && args[0].endsWith('/previous.app')) || (config.failRestoreCli && args[0].includes('/.mlxl3-cli-backup.') && args[0].endsWith('/mlxl3')))) fail();
  if ((name === 'mv' && config.fail === 'swap' && args[0].endsWith('/new.app')) || (name === 'mv' && config.fail === 'profile' && args[1].endsWith('/.zprofile')) || (name === 'ln' && config.fail === 'link')) fail();
  const result = cp.spawnSync('/bin/' + name, args, {stdio: 'inherit'});
  if (name === 'mv' && config.pauseAfterCliBackup && args[0] === cli && result.status === 0) { fs.writeFileSync(config.state + '.cli-backup', ''); setTimeout(() => process.exit(0), 60000); }
  else
  process.exit(result.status ?? 1);
}
`;

async function fixture(options = {}) {
  const root = await mkdtemp(join(tmpdir(), "mlxl3-installer-"));
  const home = join(root, "user home é");
  const bin = join(root, "tools");
  const temp = join(root, "temp");
  await Promise.all(
    [home, bin, temp].map((path) => mkdir(path, { recursive: true })),
  );
  const config = {
    archive,
    releases: copy(releases),
    log: join(root, "calls.jsonl"),
    state: join(root, "download.json"),
    ...options,
  };
  const configPath = join(root, "fixture.json");
  await writeFile(configPath, JSON.stringify(config));
  for (const name of [
    "uname",
    "sw_vers",
    "find",
    "pgrep",
    "plutil",
    "curl",
    "hdiutil",
    "ditto",
    "codesign",
    "mv",
    "ln",
    "rm",
  ]) {
    if (name === "plutil" && process.env.MLXL3_TEST_NATIVE_PLUTIL === "1")
      continue;
    await writeFile(join(bin, name), `#!${process.execPath}\n${stub}`, {
      mode: 0o755,
    });
  }
  const app = join(home, "Applications/MLXL3 Desktop.app");
  const profile = join(home, ".zprofile");
  await writeFile(profile, "# user's existing config\nexport KEEP_ME=yes\n");
  return {
    root,
    home,
    app,
    profile,
    config,
    async update(changes) {
      Object.assign(config, changes);
      await writeFile(configPath, JSON.stringify(config));
    },
    async existing(version = "1.4.0", build = "23") {
      await mkdir(join(app, "Contents/Resources/runtime"), { recursive: true });
      await writeFile(
        join(app, "Contents/Info.plist"),
        JSON.stringify({
          CFBundleIdentifier: "io.mlxl3.desktop",
          CFBundleShortVersionString: version,
          CFBundleVersion: build,
        }),
      );
      await writeFile(
        join(app, "Contents/Resources/runtime/mlxl3"),
        "old CLI",
        { mode: 0o755 },
      );
    },
    run(args = [], interrupt = false, input, tty = false) {
      return new Promise((resolve, reject) => {
        const bashArgs =
          input === undefined ? [installer, ...args] : ["-s", "--", ...args];
        const command = tty
          ? (process.env.MLXL3_TEST_PYTHON ?? "/usr/bin/python3")
          : "/bin/bash";
        const commandArgs = tty
          ? [
              "-I",
              "-c",
              `import errno, os, pty, subprocess, sys
master, slave = pty.openpty()
child = subprocess.Popen(sys.argv[1:], stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
try:
    while True:
        try:
            data = os.read(master, 65536)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    status = child.wait(timeout=3)
finally:
    os.close(master)
    if child.poll() is None:
        child.kill()
        child.wait(timeout=2)
sys.exit(status)`,
              "/bin/bash",
              ...bashArgs,
            ]
          : bashArgs;
        const environment = { ...process.env, TERM: "xterm-256color" };
        if (config.noColor) environment.NO_COLOR = "1";
        else delete environment.NO_COLOR;
        const child = spawn(command, commandArgs, {
          detached: true,
          env: {
            ...environment,
            HOME: home,
            SHELL: "/bin/zsh",
            TMPDIR: temp,
            PATH: `${bin}:/usr/bin:/bin:/usr/sbin:/sbin`,
            MLXL3_TEST_CONFIG: configPath,
          },
        });
        child.stdin.end(input ?? "");
        let output = "";
        child.stdout.on("data", (data) => {
          output += data;
        });
        child.stderr.on("data", (data) => {
          output += data;
        });
        let killTimer;
        const interruptTimer = interrupt
          ? setInterval(async () => {
              if (
                await exists(
                  config.pauseAfterCliBackup
                    ? config.state + ".cli-backup"
                    : config.state,
                )
              ) {
                clearInterval(interruptTimer);
                try {
                  process.kill(-child.pid, "SIGTERM");
                } catch {
                  /* already joined */
                }
              }
            }, 20)
          : undefined;
        const timeout = setTimeout(() => {
          process.kill(-child.pid, "SIGTERM");
          killTimer = setTimeout(() => {
            try {
              process.kill(-child.pid, "SIGKILL");
            } catch {
              /* joined */
            }
          }, 500);
        }, 15000);
        child.on("error", (error) => {
          clearTimeout(timeout);
          clearTimeout(killTimer);
          clearInterval(interruptTimer);
          reject(error);
        });
        child.on("close", (code, signal) => {
          clearTimeout(timeout);
          clearTimeout(killTimer);
          clearInterval(interruptTimer);
          resolve({ code, signal, output });
        });
      });
    },
    async calls() {
      return (await readFile(config.log, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
    },
    async close() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}
async function cliBackups(f) {
  const bin = join(f.home, ".local/bin");
  return (await readdir(bin))
    .filter((name) => name.startsWith(".mlxl3-cli-backup."))
    .map((name) => join(bin, name, "mlxl3"));
}
async function legacyCli(f, relative = true) {
  const cli = join(f.home, ".local/bin/mlxl3"),
    target = join(f.home, "build/mlxl3-rs");
  await mkdir(join(f.home, ".local/bin"), { recursive: true });
  await mkdir(join(f.home, "build"), { recursive: true });
  const contents = "#!/bin/sh\nexit 99\n";
  await writeFile(target, contents, { mode: 0o755 });
  const link = relative ? "../../build/mlxl3-rs" : target;
  await symlink(link, cli);
  return { cli, target, contents, link };
}
function scenario(name, callback, options = {}) {
  test(name, { timeout: 20000 }, async () => {
    const f = await fixture(options);
    try {
      await callback(f);
    } finally {
      await f.close();
    }
  });
}

scenario(
  "truncated bootstrap cannot silently lose the requested beta channel",
  async (f) => {
    const source = await readFile(installer, "utf8");
    const truncated = source.slice(0, source.lastIndexOf('main "$@"')) + "main";
    const result = await f.run(["--channel", "beta"], false, truncated);
    assert.notEqual(result.code, 0, result.output);
    assert.match(result.output, /Incomplete installer/);
    assert.equal(await exists(f.app), false);
  },
);

scenario(
  "fresh stable install: latest published Desktop, real SHA-256, CLI and PATH; rerun is idempotent",
  async (f) => {
    const result = await f.run();
    assert.equal(result.code, 0, result.output);
    assert.match(result.output, /Desktop: 1.5.1 \(stable\).*CLI included/);
    assert.equal(
      JSON.parse(await readFile(join(f.app, "Contents/Info.plist")))
        .CFBundleShortVersionString,
      "1.5.1",
    );
    assert.equal(
      await readlink(join(f.home, ".local/bin/mlxl3")),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
    assert.equal(
      (await lstat(join(f.home, ".local/bin/mlxl3"))).isSymbolicLink(),
      true,
    );
    const profile = await readFile(f.profile, "utf8");
    assert.match(profile, /KEEP_ME=yes/);
    assert.equal(
      profile.split('export PATH="$HOME/.local/bin:$PATH"').length - 1,
      1,
    );
    const again = await f.run();
    assert.equal(again.code, 0, again.output);
    assert.match(again.output, /already current/);
    assert.equal(await readFile(f.profile, "utf8"), profile);
    assert.equal(
      (await f.calls()).filter(
        (call) => call.name === "hdiutil" && call.args[0] === "attach",
      ).length,
      1,
    );
  },
);

scenario(
  "beta selection and update preserve models, conversations and unrelated files",
  async (f) => {
    await f.existing();
    const data = join(f.home, "Library/Application Support/io.mlxl3.desktop");
    await mkdir(data, { recursive: true });
    await writeFile(join(data, "sentinel"), "models and conversations");
    const result = await f.run(["--channel", "beta"]);
    assert.equal(result.code, 0, result.output);
    assert.equal(
      JSON.parse(await readFile(join(f.app, "Contents/Info.plist")))
        .CFBundleShortVersionString,
      "2.0.0",
    );
    assert.equal(
      await readFile(join(data, "sentinel"), "utf8"),
      "models and conversations",
    );
    const again = await f.run(["--channel", "beta"]);
    assert.equal(again.code, 0, again.output);
    assert.match(again.output, /already current/);
    assert.equal(
      (await f.calls()).filter(
        (call) => call.name === "hdiutil" && call.args[0] === "attach",
      ).length,
      1,
    );
  },
);

for (const fail of [
  "api",
  "download",
  "mount",
  "copy",
  "signature",
  "swap",
  "profile",
  "link",
  "missing-cli",
  "cli-smoke",
]) {
  scenario(
    `failure ${fail}: previous app and shell configuration survive`,
    async (f) => {
      await f.existing();
      const before = await readFile(f.profile, "utf8");
      const result = await f.run();
      assert.notEqual(result.code, 0, result.output);
      assert.match(result.output, /Installation stopped/);
      assert.match(result.output, /Next: \S/);
      const details = result.output.match(/Details: (.+)/);
      assert.ok(details);
      assert.ok((await readFile(details[1], "utf8")).includes("Cause:"));
      assert.equal(
        await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
        "old CLI",
      );
      assert.equal(await readFile(f.profile, "utf8"), before);
      assert.equal(
        await exists(join(f.home, "Library/Caches/io.mlxl3.install.lock")),
        false,
      );
      const calls = await f.calls();
      const expected = {
        api: "curl",
        download: "curl",
        mount: "hdiutil",
        copy: "ditto",
        signature: "codesign",
        swap: "mv",
        profile: "mv",
        link: "ln",
        "missing-cli": "hdiutil",
        "cli-smoke": "ln",
      }[fail];
      assert.ok(
        calls.some((call) => call.name === expected),
        `failure ${fail} was never reached`,
      );
    },
    { fail },
  );
}

for (const changes of [
  { corrupt: true },
  { bundleId: "other.app" },
  { bundleVersion: "9.0.0" },
  { response: "{" },
  { response: "[]" },
  { running: true },
]) {
  scenario(
    `reject invalid input ${JSON.stringify(changes)}`,
    async (f) => {
      await f.existing();
      const result = await f.run();
      assert.notEqual(result.code, 0, result.output);
      assert.equal(
        await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
        "old CLI",
      );
    },
    changes,
  );
}

scenario(
  "download metadata boundaries: foreign URL, missing digest, invalid size and duplicate DMG rejected",
  async (f) => {
    for (const change of [
      { browser_download_url: "https://evil.example/app.dmg" },
      { digest: null },
      { size: 0 },
      { size: 536870913 },
      { size: "12" },
    ]) {
      const bad = copy(stable);
      Object.assign(bad.assets[0], change);
      await f.update({ releases: [bad] });
      const result = await f.run();
      assert.notEqual(result.code, 0, result.output);
      assert.equal(await exists(f.app), false);
    }
    const bad = copy(stable);
    bad.assets.push({ ...bad.assets[0] });
    await f.update({ releases: [bad] });
    assert.notEqual((await f.run()).code, 0);
  },
);

scenario(
  "existing CLI file is backed up before installing the bundled CLI",
  async (f) => {
    await mkdir(join(f.home, ".local/bin"), { recursive: true });
    const cli = join(f.home, ".local/bin/mlxl3");
    await writeFile(cli, "unrelated");
    const result = await f.run();
    assert.equal(result.code, 0, result.output);
    assert.equal(
      await readlink(cli),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
    const backups = await cliBackups(f);
    assert.equal(backups.length, 1);
    assert.equal(await readFile(backups[0], "utf8"), "unrelated");
    assert.match(result.output, /CLI backup/);
  },
);

scenario(
  "existing development CLI link is migrated without executing or changing its target",
  async (f) => {
    await f.existing();
    const target = join(f.home, "local build/target/release/mlxl3-rs");
    await mkdir(join(f.home, "local build/target/release"), {
      recursive: true,
    });
    const old = "#!/bin/sh\nexit 99\n";
    await writeFile(target, old, { mode: 0o755 });
    await mkdir(join(f.home, ".local/bin"), { recursive: true });
    await symlink(target, join(f.home, ".local/bin/mlxl3"));
    const result = await f.run(["--channel", "beta"], false, undefined, true);
    assert.equal(result.code, 0, result.output);
    assert.equal(await readFile(target, "utf8"), old);
    const backups = await cliBackups(f);
    assert.equal(backups.length, 1);
    assert.equal(await readlink(backups[0]), target);
    assert.equal(
      await readlink(join(f.home, ".local/bin/mlxl3")),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
    for (const word of [
      "[1/5]",
      "[2/5]",
      "[3/5]",
      "[4/5]",
      "[5/5]",
      "Next steps",
    ])
      assert.ok(result.output.includes(word), word);
    assert.equal(result.output.includes(String.fromCharCode(27)), true);
    if (process.env.MLXL3_INSTALLER_EVIDENCE)
      await writeFile(
        join(process.env.MLXL3_INSTALLER_EVIDENCE, "terminal-update.ansi"),
        result.output,
      );
  },
);

for (const changes of [
  { os: "Linux" },
  { arch: "x86_64" },
  { version: "26.1" },
  { version: "bad" },
]) {
  scenario(
    `unsupported platform ${JSON.stringify(changes)} has no install side effects`,
    async (f) => {
      assert.notEqual((await f.run()).code, 0);
      assert.equal(await exists(join(f.home, "Applications")), false);
    },
    changes,
  );
}

for (const fail of ["cli-backup", "link", "cli-smoke", "profile"])
  scenario(
    `CLI migration failure ${fail} restores the relative link and previous app`,
    async (f) => {
      await f.existing();
      const old = await legacyCli(f);
      const result = await f.run();
      assert.notEqual(result.code, 0, result.output);
      assert.equal(await readlink(old.cli), old.link);
      assert.equal(await readFile(old.cli, "utf8"), old.contents);
      assert.equal(
        await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
        "old CLI",
      );
      assert.deepEqual(await cliBackups(f), []);
    },
    { fail },
  );

scenario("a CLI directory is preserved with an actionable error", async (f) => {
  const cli = join(f.home, ".local/bin/mlxl3");
  await mkdir(cli, { recursive: true });
  await writeFile(join(cli, "keep"), "data");
  const result = await f.run();
  assert.notEqual(result.code, 0);
  assert.match(result.output, /folder or another non-command item/);
  assert.match(result.output, /Next: Move/);
  assert.equal(await readFile(join(cli, "keep"), "utf8"), "data");
  assert.equal(await exists(f.app), false);
});

scenario(
  "a broken CLI symlink is preserved in backup and repaired",
  async (f) => {
    await mkdir(join(f.home, ".local/bin"), { recursive: true });
    const cli = join(f.home, ".local/bin/mlxl3");
    await symlink("../../missing/mlxl3-rs", cli);
    const result = await f.run();
    assert.equal(result.code, 0, result.output);
    assert.equal(
      await readlink((await cliBackups(f))[0]),
      "../../missing/mlxl3-rs",
    );
    assert.equal(
      await readlink(cli),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
  },
);

scenario(
  "a current Desktop repairs a legacy CLI without downloading again",
  async (f) => {
    await f.existing("1.5.1", "33");
    await writeFile(
      join(f.app, "Contents/Resources/runtime/mlxl3"),
      "#!/bin/sh\nexit 0\n",
      { mode: 0o755 },
    );
    const old = await legacyCli(f);
    const result = await f.run();
    assert.equal(result.code, 0, result.output);
    assert.match(result.output, /already current/);
    assert.equal((await cliBackups(f)).length, 1);
    assert.equal(
      await readlink(old.cli),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
    assert.equal(
      (await f.calls()).some((c) => c.name === "hdiutil"),
      false,
    );
  },
);

scenario(
  "interrupt after CLI backup restores the previous command and app",
  async (f) => {
    await f.existing();
    const old = await legacyCli(f);
    const result = await f.run([], true);
    assert.notEqual(result.code, 0);
    assert.equal(await readlink(old.cli), old.link);
    assert.equal(await readFile(old.cli, "utf8"), old.contents);
    assert.equal(
      await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
      "old CLI",
    );
    assert.deepEqual(await cliBackups(f), []);
    assert.match(result.output, /interrupted/);
  },
  { pauseAfterCliBackup: true },
);

scenario(
  "failed Desktop rollback retains its backup without claiming restoration",
  async (f) => {
    await f.existing();
    await legacyCli(f);
    const result = await f.run();
    assert.notEqual(result.code, 0);
    assert.doesNotMatch(
      result.output,
      /Your previous Desktop app was restored/,
    );
    const backup = result.output.match(/Restore the previous app from: (.+)/);
    assert.ok(backup);
    assert.equal(
      await readFile(
        join(backup[1], "Contents/Resources/runtime/mlxl3"),
        "utf8",
      ),
      "old CLI",
    );
  },
  { fail: "profile", failRestoreApp: true },
);

scenario(
  "failed CLI rollback retains the exact original link for manual recovery",
  async (f) => {
    await f.existing();
    const old = await legacyCli(f);
    const result = await f.run();
    assert.notEqual(result.code, 0);
    const backups = await cliBackups(f);
    assert.equal(backups.length, 1);
    assert.equal(await readlink(backups[0]), old.link);
    assert.match(result.output, /Previous CLI kept at:/);
    assert.match(result.output, /Restore it to:/);
    assert.equal(
      await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
      "old CLI",
    );
  },
  { fail: "profile", failRestoreCli: true },
);

for (const noColor of [false, true])
  scenario(
    `terminal output: color on TTY, NO_COLOR=${noColor}`,
    async (f) => {
      const result = await f.run(["--help"], false, undefined, true);
      assert.equal(
        result.code,
        0,
        `TTY helper must finish, not merely print output: ${result.signal}`,
      );
      assert.match(result.output, /Desktop \+ CLI installer/);
      assert.match(result.output, /Usage:/);
      assert.equal(result.output.includes(String.fromCharCode(27)), !noColor);
    },
    { noColor },
  );

scenario(
  "cleanup error distinguishes an installed app from an interrupted installation",
  async (f) => {
    const result = await f.run();
    assert.notEqual(result.code, 0);
    assert.match(result.output, /Installed; cleanup needs attention/);
    assert.match(result.output, /could not eject/);
    assert.equal(
      await readlink(join(f.home, ".local/bin/mlxl3")),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
  },
  { fail: "detach" },
);

scenario(
  "old app cleanup failure reports a usable installation and keeps the backup",
  async (f) => {
    await f.existing();
    const result = await f.run();
    assert.notEqual(result.code, 0);
    assert.match(result.output, /Installed; cleanup needs attention/);
    assert.match(result.output, /temporary app folder/);
    assert.equal(
      await readlink(join(f.home, ".local/bin/mlxl3")),
      join(f.app, "Contents/Resources/runtime/mlxl3"),
    );
    const folders = (await readdir(join(f.home, "Applications"))).filter(
      (name) => name.startsWith(".mlxl3-install."),
    );
    assert.equal(folders.length, 1);
    assert.equal(
      await readFile(
        join(
          f.home,
          "Applications",
          folders[0],
          "previous.app/Contents/Resources/runtime/mlxl3",
        ),
        "utf8",
      ),
      "old CLI",
    );
  },
  { fail: "cleanup-stage" },
);

scenario("unknown channel is refused before downloads", async (f) => {
  assert.notEqual((await f.run(["--channel", "nightly"])).code, 0);
  assert.equal(await exists(f.app), false);
});

scenario(
  "interrupted silent download leaves the old application usable and releases the lock",
  async (f) => {
    await f.existing();
    const result = await f.run([], true);
    assert.notEqual(result.code, 0);
    assert.equal(
      await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
      "old CLI",
    );
    assert.equal(
      await exists(join(f.home, "Library/Caches/io.mlxl3.install.lock")),
      false,
    );
  },
  { fail: "silence" },
);

for (const order of [
  [0, 1, 2],
  [0, 2, 1],
  [1, 0, 2],
  [1, 2, 0],
  [2, 0, 1],
  [2, 1, 0],
]) {
  for (const channel of ["stable", "beta"]) {
    scenario(
      `finite release order ${order.join("")} selects latest ${channel}`,
      async (f) => {
        const result = await f.run(["--channel", channel]);
        assert.equal(result.code, 0, result.output);
        const info = JSON.parse(
          await readFile(join(f.app, "Contents/Info.plist")),
        );
        assert.equal(
          info.CFBundleShortVersionString,
          channel === "stable" ? "1.5.1" : "2.0.0",
        );
      },
      { releases: order.map((index) => [older, stable, beta][index]) },
    );
  }
}

scenario(
  "pagination finds Desktop beyond 100 engine releases",
  async (f) => {
    const result = await f.run(["--channel", "beta"]);
    assert.equal(result.code, 0, result.output);
    assert.equal(
      JSON.parse(await readFile(join(f.app, "Contents/Info.plist")))
        .CFBundleShortVersionString,
      "2.0.0",
    );
    assert.ok(
      (await f.calls()).some(
        (call) =>
          call.name === "curl" &&
          call.args.some((arg) => arg.endsWith("page=2")),
      ),
    );
  },
  {
    releases: [
      ...Array.from({ length: 100 }, (_, index) => ({
        tag_name: `engine-v${index}`,
      })),
      stable,
      beta,
    ],
  },
);

for (const bad of [
  { ...stable, draft: "false" },
  { ...stable, prerelease: "false" },
  { ...stable, published_at: "bad" },
  { ...stable, assets: null },
]) {
  scenario(
    `malformed release metadata is rejected: ${JSON.stringify(bad).slice(0, 90)}`,
    async (f) => {
      assert.notEqual((await f.run()).code, 0);
      assert.equal(await exists(f.app), false);
    },
    { releases: [bad] },
  );
}

scenario(
  "app opened during verification is not replaced",
  async (f) => {
    await f.existing();
    const result = await f.run();
    assert.notEqual(result.code, 0);
    assert.match(result.output, /opened while the update was downloading/);
    assert.equal(
      await readFile(join(f.app, "Contents/Resources/runtime/mlxl3"), "utf8"),
      "old CLI",
    );
  },
  { startDuringVerify: true },
);

scenario("an existing system application is updated in place", async (f) => {
  await f.existing();
  const systemApp = join(f.root, "system applications/MLXL3 Desktop.app");
  await mkdir(join(f.root, "system applications"), { recursive: true });
  await rename(f.app, systemApp);
  await f.update({ globalApp: systemApp });
  const result = await f.run();
  assert.equal(result.code, 0, result.output);
  assert.equal(
    JSON.parse(await readFile(join(systemApp, "Contents/Info.plist")))
      .CFBundleShortVersionString,
    "1.5.1",
  );
  assert.equal(await exists(f.app), false);
  assert.equal(
    await readlink(join(f.home, ".local/bin/mlxl3")),
    join(systemApp, "Contents/Resources/runtime/mlxl3"),
  );
});

scenario("explicit stable channel can replace an installed beta", async (f) => {
  await f.existing("2.0.0", "35");
  const result = await f.run(["--channel", "stable"]);
  assert.equal(result.code, 0, result.output);
  assert.equal(
    JSON.parse(await readFile(join(f.app, "Contents/Info.plist")))
      .CFBundleShortVersionString,
    "1.5.1",
  );
});
