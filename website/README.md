# MLXL3 website

A single-screen, static landing page for MLXL3. Its ASCII texture is exported from the Desktop renderer. Travelling bands of light sweep through the ASCII field, synchronized to the display. The pointer stirs a fixed-grid 2D fluid. A pressure projection reduces divergence and semi-Lagrangian backtracing transports a separate dye concentration. Each fixed 5 × 8 CSS-pixel cell selects a Menlo glyph from a seven-character atlas. The passing light and the dye control which glyph appears. The cursor stirs a narrow band of the grid. Browsers without WebGL show the same four still prints. The animation pauses for Reduce Motion and hidden tabs. The copy button installs the latest stable Desktop and its bundled CLI using the installer’s default channel.

GitHub and Discord links point to the project repository and community invite.

Run from this directory with `python3 -m http.server 4187 --bind 127.0.0.1`. The browser uses local fonts and assets. GitHub and Discord marks are from Simple Icons.

The public files are `index.html`, `styles.css`, `main.js`, `aurora.js`, `install.sh`, `assets/`, `_headers`, `_redirects`, `indexnow.txt`, `robots.txt` and `sitemap.xml`. The former preview URL redirects to the landing page. CSS and script URL versions are the first 12 hex digits of their SHA-256 digest.

With dependencies installed, run `bun run check`, `bun run lint`, `bun run format:check`, `bun run lint:shell`, `bun run format:shell:check` and `bun run test`. The tests use the existing Node runner and Playwright Chromium. Installer fixtures use disposable paths.
