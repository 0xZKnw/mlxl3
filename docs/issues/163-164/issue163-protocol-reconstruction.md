# PR163 protocol reconstruction: T01–T07

## Scope and provenance

Reconstructed from issue [162](https://github.com/0xZKnw/mlxl3/issues/162) and its annex against local `f428945b40533d11b3cf10048954158fabeb4012` (published tree recovery), not from claims about the lost unpublished implementation. The parent preregistered this reconstruction in `opti.md` before execution. This report covers MCP, OpenAI capabilities, structured validation, and token/answer-context changes. Other workers changed Desktop and native runtime concurrently. The final CPU library total therefore includes their tests and is not attributed exclusively to this work.

No GPU, model-speed benchmark, installed-app change, push or merge was performed by this worker. Quality, performance, and integration statuses are separate: CPU tests passed; performance was not measured; code remains local pending parent publication. No full parser, concurrency, allocator, OS process-reaping, or tokenizer round-trip proof is claimed.

## Implemented contracts

- **T01:** Stdio now owns nonblocking stdin/stdout/stderr and the child process, rather than detached reader threads and a 128-message queue. A monotonic absolute 12-second deadline covers request writes, server-request replies, intermediate messages, and reads. The production bridge passes its cancellation flag into `Manager::call_cancellable`. Polling checks cancellation/deadline every bounded read/write iteration. Errors invalidate the client so cleanup runs immediately. Cleanup kills the owned process group and polls reaping for at most one second; an OS process stuck in uninterruptible kernel I/O is not guaranteed to be reaped by that bound. There are no transport reader threads left to detach or join.
- **T02:** HTTP uses an owned curl child with incremental stdout consumption, without body/header response files. SSE accepts fragmented UTF-8 lines, LF/CRLF, comments, empty events, and multiple events; notifications and unrelated response IDs do not consume the awaited response. Server-initiated requests receive JSON-RPC method-not-supported replies under the original deadline. A matching result is consumed without waiting for SSE EOF. Direct JSON remains supported and requires successful curl completion; incomplete bodies cannot be accepted merely because their prefix is valid JSON. Informational HTTP status blocks share the header budget.
- **T03:** Incoming chunks are 4 KiB, stderr drains into a fixed 4 KiB buffer without line allocation, aggregate incoming response bytes are limited to 8 MiB per request (including notifications), application response headers to 64 KiB, and retained paginated tool metadata to 8 MiB. The stdio serialized queue was removed. Curl's own transport/header implementation is not formally verified; these are application budgets, not measured RSS values. JSON representation and serialization overhead are not equal to wire size.
- **T04:** A file lock now encloses first creation, read, mutation, validation, atomic persist, and parent-directory fsync. Concurrent first-launch add/add and distinct add/remove updates survive. Read/list also takes this lock and preserves the built-in Exa entry.
- **T05:** The native ready event exports actual `images_supported`; the API retains it and backend metadata and derives `images`/`text_only` from that explicit capability, defaulting safely to text-only if absent. Image requests are refused early for text backends. `json_schema`/`validated_json` and accepted response formats are advertised separately from `constrained_decoding:false`. This is post-generation validation/retry, not token-by-token grammar-constrained generation. Existing unsupported-schema keyword rejection is unchanged.
- **T06:** `uniqueItems` uses exact canonical JSON keys in a balanced set instead of pairwise comparison. Integer values never round-trip through f64. Integral floats normalize only inside the exact signed/unsigned integer conversion range, preserving `1 == 1.0`, nested equivalence, negative zero, and distinct integers above 2^53. Validation has a shared one-million-node visit budget and cancellation checks, with exhaustion reported separately and checked after matching so `not` cannot turn exhaustion into acceptance. This budget does not claim to count every byte comparison or prove the entire schema parser's CPU complexity.
- **T07, implemented portion:** Production generation obtains channel controls from the tokenizer's declared special-token IDs, not a lexical search in arbitrary decoded text. Ordinary-token bytes spelling markers remain literal. Gemma's special channel introducer and following `thought` label are handled separately. The legacy `split` CLI intentionally retains its lexical input protocol. `assistant_context` now comes from the exact accumulated visible answer deltas (or validated JSON), and tool parsing receives channel-separated text instead of stripping the raw response after its last marker. Original raw `cache_context` remains available.

## T07 remaining acceptance limit and next change

`cache_context` is still a string-only, retokenizable text contract. It does **not** preserve the exact original token-ID sequence: ordinary tokens that spell a reserved marker can be re-encoded as a special token on later replay. This patch fixes the current delta/assistant-context truncation and distinguishes token controls during generation, but does not close exact cache replay provenance.

A complete token-preserving replay contract needs versioned replay metadata containing generated IDs and tokenizer/template/model identity, persisted by consumers and admitted only on identity/prefix match, with a safe text fallback. Test ordinary-piece marker generation, quoted/code markers, actual reasoning controls, all chunk boundaries, restart/history persistence, and mismatched identities through the actual Desktop/bridge/tokenizer path. Do not label T07 fully closed on the current CPU fixtures alone. Model-specific decoder behavior, including physical Flash/Qwen/LFM3 paths, remains unqualified on this Linux host.

## Verification

Toolchain: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, Linux CPU. Final commands used `PATH=/home/agent/.cargo/bin:$PATH`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, and `CARGO_TARGET_DIR=/tmp/pr163-protocol-target`.

- `cargo test --no-default-features --features chat --lib -- --nocapture`: **376 passed, 10 ignored, 0 failed**, 34.28 s. This includes ten MCP regressions, eight structured tests, five streaming tests, and seven successful tokenizer tests. The tokenizer-vs-local-Transformers fixture remains ignored because the physical model and local Transformers input are absent.
- `cargo test --no-default-features --features chat --test openai_transport -- --nocapture`: **37 passed, 0 ignored, 0 failed**, 0.83 s. Includes capability/valid response-format assertions and a three-backend image admission matrix with a fake resident bridge. This does not validate image decoding by a real model.
- MCP tests exercise a nonreading stdin and newline-free noisy stderr, live cancellation, notification floods, an absolute deadline under continuing notifications, late result rejection, aggregate byte overflow, first-creation/add/add/add/remove configuration concurrency, every two-way SSE byte split, CRLF notifications followed by unrelated then matching IDs, result-before-EOF, oversized body/header, truncated SSE, valid direct JSON, and a valid JSON prefix with an incomplete HTTP Content-Length body.
- `cargo clippy --no-default-features --features chat --all-targets -- -D warnings`: **passed**.
- `rustfmt --edition 2024 --check` for all seven exclusively owned Rust files: **passed**. Whole-tree `cargo fmt --check` observed concurrent native-worker formatting differences in `gpu/gate.rs` and `native_draft.rs`; preserved rather than overwriting their work. Parent must rerun whole-tree formatting after integration.
- `git diff --check` on owned files and shared main: **passed** at handoff.
- Numeric-normalization mutation (`float integer key + 1`) was **detected** by the production regression on `[1,1.0]`, exit 101. The mutation was restored in `finally`, then the full library and API suites above passed on restored code.
- Added `validation_budget_never_wraps_or_accepts_cancellation`, a Kani harness over the actual production budget step with symbolic `usize` and cancellation, no assumptions, and both outcome covers. Attempted `cargo kani --lib --no-default-features --harness validation_budget_never_wraps_or_accepts_cancellation`: **not verified**, `cargo kani` is not installed. Neither the requested formal-proof skill nor a checkout `.agents` copy was available. No proof result is inferred from tests.

## Failures retained

1. Initial unformatted implementation failed formatting, then owned files were formatted.
2. The first oversized-header fixture omitted its final CRLF; curl rejected the incomplete header before the application parser, so an over-specific error assertion failed. The fixture was completed and the final production HTTP test passed.
3. An OpenAI test command omitted the existing required `chat` feature and failed to resolve `vision_input`; corrected final command includes `--features chat`.
4. Debug-info builds filled the 4.9 GiB `/tmp` volume. Only this worker's disposable target directory was removed; reduced-debug/incremental-off builds then completed. No source or user files were removed.
5. First strict Clippy run found two needless path borrows and one collapsible conditional in new MCP code. Those diagnostics were fixed; final strict Clippy passed.
6. Kani unavailable, physical Apple/model tests unexecuted, ten existing library tests ignored, and exact cache token replay unresolved as described above.

## Evidence

Raw final results, mutation failure, proof-tool diagnostic, whole-tree format attempt, owned-file format check, and source hashes are in [`measurements/pr163-protocol-reconstruction/`](../../measurements/pr163-protocol-reconstruction/). The seven owned source hashes identify the tested protocol implementation; shared `main.rs` also contains parent/native-worker integration edits, so parent final commit CI and macOS qualification are still necessary.

## T07-R2 preregistration: declared delimiters with `special:false`

Independent review found that official Qwen3-8B declares `<think>` (151667) and
`</think>` (151668) with `special:false` in its tokenizer configuration:
https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json .
The flag controls decode skipping; it does not make the declared token ID
ordinary prose. Add a production ChatTokenizer fixture covering both flags,
declared open/close IDs, and literal marker bytes assembled from ordinary IDs.
Run against the previous filter to reproduce failure, then admit recognized
added-token delimiter IDs regardless of that flag and rerun scoped checks.
No physical model, performance, or exact cache replay claim is added.

### T07-R2 result and clarified card scope

The new fixture failed on the previous implementation exactly at the
`special:false` delimiter lookup (`None` instead of `<think>`). The correction
uses recognized **declared added-token IDs**, independent of the `special`
decode-skipping flag. Both false and true cases now pass through the real
`ChatTokenizer::load`, tokenizers incremental decoder, and production splitter.
Separate ordinary IDs decode to the same `<think>` bytes and remain answer text;
declared opening/closing IDs produce reasoning then answer correctly. This is a
representative tokenizer fixture, not a download or physical run of Qwen3-8B.

Scoped results: tokenizer **7 passed, 1 ignored** (external Transformers/model
fixture); streaming **5 passed**; `cargo clippy --no-default-features --features
chat --lib -- -D warnings` passed. The before-fix failure and final results are
preserved as `t07-r2-*.log`. No aggregate/model benchmark was repeated. Prior
source hashes remain historical; `t07-r2-source-sha256.txt` identifies this
correction. No Kani harness changed in R2.

The issue asks for an explicit channel/token/escaping and context contract, not
a new persisted token-ID replay protocol. The applicable contract is:

- `assistant_context` equals the accumulated visible answer deltas, except the
  already-explicit structured-output validated JSON replacement.
- `cache_context` retains the full raw decoded response, including declared
  control delimiters. It is **text for template rendering and retokenization**,
  not an assertion of exact token-ID replay. Legacy history uses the same text
  fallback. Quotes/backticks do not escape a declared control token; identical
  bytes produced by different ordinary IDs are literal during generation.
- Reuse is based on actual encoded-token prefix equality and existing
  conversation/mode checks, not on raw-text equality. See
  `contracts::reusable_prefix_len`, direct-native `native_prompt_cache::prefill`,
  and MLX prefill callers in `main.rs`. Different retokenization cannot silently
  reuse an unequal saved token prefix; it takes the existing cold-prefill path.

The earlier proposed versioned replay metadata is therefore an optional future
exact-token-continuation extension, **not an unimplemented requirement inferred
for T07**. The current card's truncation fix and declared-ID channel contract are
implemented and CPU-tested. Actual model tokenizer/decoder fixtures and macOS
end-to-end qualification remain unexecuted and are not claimed complete.
