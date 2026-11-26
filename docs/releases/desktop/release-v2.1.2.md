# MLXL3 Desktop 2.1.2

**Agent turns on Inkling 2 no longer read the conversation again in the middle of a turn, and no longer read the model's own output again at each step.**

Desktop **2.1.2 / build 90** bundles MLXL3 engine **2.0.39**, built from this source. The engine is also published on its own as [`engine-v2.0.39`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.39).

## Changed

- **The prompt cache of Inkling 2 keeps its snapshot.** In an Agent turn, one step could read the whole conversation again from the first token (9,619 tokens in the turn this was found in, shown as `cache 68%`). The saved state sometimes exceeded the engine's cache budget and was dropped without a word; it is now compact and always fits. Nothing to set, and answers are unchanged.
- **Agent steps go on from where the model stopped.** When the engine announces it (Inkling 2 with its DFlash2 drafter), Agent turns ask for the live prompt cache: each step keeps the model's own output, reasoning included, and only tool results and your next message are read. In a two-turn trial with real tools, 92 to 139 tokens were read per step instead of 259 to 314.
  - The model now sees its reasoning from the earlier steps of the task. This mode is not identical to the bit to reading the prompt from scratch; details in the engine notes.
  - Chat requests without tools and other models are sent as before.

Everything else is [Desktop 2.1.1](https://github.com/0xZKnw/mlxl3/releases/tag/v2.1.1).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

**Windows:** the Windows app uses its own engine and is unchanged; its installer for this tag has the same content as 2.1.1.

## Verification and limits

- The engine checks are listed in the [engine 2.0.39 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.39): the snapshot fix is verified to the bit on the real model, the live prompt cache by an oracle and by two-turn Agent runs with real tools on the engine's bridge.
- **The app change was only compiled** (it sends one more request field in Agent turns when the engine announces support). The Studio modes check that covers this field, and the Desktop suite, were **not run**: the final verification pass was stopped at the maintainer's request to publish.
- **Not verified:** the app of this DMG was not launched; no Agent turn was run through the app itself with this engine, so the cache figure it shows was not read on screen. CI of `main` for this source had not finished when this release was published.

## Source and download

- Source: [`SOURCE_SHA`](https://github.com/0xZKnw/mlxl3/commit/SOURCE_SHA) (tag `v2.1.2`).
- `MLXL3-Desktop-v2.1.2-b90-Apple-Silicon.dmg`: DMG_SIZE bytes, SHA-256 `DMG_SHA`.
- `MLXL3-Desktop-v2.1.2-b90-Apple-Silicon.dmg.sha256`: DMGSUM_SIZE bytes.
- Bundled engine: MLXL3 engine 2.0.39 (`engine-v2.0.39`).
- DMG_CHECKS

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the report, the decisions and the request to publish.
- OpenAI's [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/), Anthropic's [prompt caching guide](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and DeepSeek's [thinking mode guide](https://api-docs.deepseek.com/guides/thinking_mode) for the practice this follows.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
