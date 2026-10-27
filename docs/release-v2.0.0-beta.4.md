# MLXL3 Desktop 2.0.0-beta.4 — build 35

This prerelease adds visible download progress, executable HTML previews in Chat and Agent, and safeguards for stalled Agent turns. It includes native engine **2.0.1**. The stable update channel is unchanged.

## New features

- **Updater:** a real progress bar, percentage, received/total bytes and download speed for both Desktop and engine downloads. SHA-256 verification is shown separately; a paused transfer stops reporting a stale speed.
- **HTML in Chat and Agent:** finished HTML answers expose **Run HTML**. Agent also offers previews of HTML files created inside its task folder. Inline JavaScript, ES modules, import maps and relative task-folder modules/CSS work. HTTPS dependencies can load from jsDelivr, unpkg, esm.sh, cdnjs, Skypack and threejs.org, including Three.js.
- **Isolated previews:** ephemeral WebKit storage, a response-header policy and independent resource rules restrict access. Previews cannot read arbitrary host files, reach LAN services, open new windows, navigate outside the document, or invoke a native app bridge. Each local resource is limited to 2 MiB. Imports outside the supported CDNs, workers and package installation are unavailable.

## Fixes

- Agent now displays tool preparation while the model writes hidden tool arguments, including file-creation commands after thinking.
- Agent turns share a **4,096-token budget** across tool rounds and stop after **10 minutes overall** or **2 minutes without matching activity**. Partial output and the stopping reason remain visible.
- Stop requests cooperative cancellation, then unloads an unresponsive engine after three seconds with a two-second kill fallback. Reloading waits for the prior engine to exit.
- Deleting the final Agent task restores the Chat tool registry. Stale process callbacks and watchdogs no longer affect a replacement request.
- HTML execution stays unavailable while the overall message is streaming, processing tools, or interrupted, even when earlier chunks are already complete.

## Performance

No numerical inference kernel, model weight or sampling algorithm changed. This release claims no measured throughput or RAM improvement. Existing GPU/MLX behavior is outside the two targeted CPU proofs.

## Installation and compatibility

Apple Silicon, macOS 26.2 or later. Select **Beta** in Settings → Updates, then search for updates, or install the DMG manually. Desktop remains version 2.0.0 internally and advances to build 35; the prerelease tag identifies beta 4. Bundled engine 2.0.1 takes precedence over an older managed engine. No standalone engine release is published here.

The application is ad-hoc signed and is not notarized. Existing models and conversations remain outside the app bundle. The Agent command sandbox remains native macOS Seatbelt without Docker; it has no hard aggregate RAM/disk quota. Remote JS dependencies need an internet connection, while Agent shell commands remain offline.

## Verification and limitations

The complete local Desktop suite passed. Python: **1,833 passed / 115 skipped**; native CPU: **155 passed / 5 ignored**. The MLX/chat release build and strict Clippy passed. The updater passed **22 real transfer checks**, including cancellation, truncation, unknown Content-Length and integrity failures, also under AddressSanitizer. WebKit passed **34 default checks**, plus two optional journeys exercising the actual Qwen-generated file and a real Three.js 0.179.1 WebGL cube.

A real Qwen3.6-35B-A3B EXL3 2.49 bpw run on an Apple M5 created and read back HTML, displayed tool preparation, stopped and resumed. Its original reported prompt/settings were unavailable; this does not establish the cause of every stalled or hot generation. Two targeted Kani properties passed over symbolic preparation-state and token-budget inputs. The full String-filter attempt was interrupted without a proof; Swift source proof was unavailable. Tests and bounded checks are not a complete proof. Unrelated physical Metal/optional conversion cases were skipped.

Source qualification and retained failures: [verification report](desktop-updater-html-agent-verification.md). Final package provenance and artifacts are recorded after packaging. Publication before GitHub CI completion was explicitly requested; CI is not claimed fully green.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the product direction and the Qwen3.6 bug report; [Apple WebKit](https://webkit.org/) for the isolated browser runtime; [MLX](https://github.com/ml-explore/mlx) for local inference; and [Three.js](https://github.com/mrdoob/three.js) for the real 3D dependency used in preview qualification. Existing native-sandbox and speculative-decoding credits remain in the preceding release notes.
