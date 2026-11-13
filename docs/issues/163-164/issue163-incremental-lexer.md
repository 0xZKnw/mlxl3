# U03 incremental editor lexing, 2026-11-09

## Preregistered design and checks

Baseline: published b28d33b and the preserved concurrent recovery checkout.
Read AGENTS.md, relevant opti.md U03/U-COMPLETE entries, and responsiveness
reconstruction/completion reports. This extends the detached whole-source lexer;
it does not replace the historical reports or claim their missing tests passed.

Own StudioCodeEditor.swift, tests/responsiveness-check.swift and this report.
Parent recorded this U03 protocol under PR163-POST-PUBLISH-20261109 before
implementation/tests (the initially proposed separate identifier was not used).

Use Sendable line/checkpoint and token-chunk caches in the detached operation.
Compare old/new line text, propagate conservative multiline delimiter state only
through affected lines until the unchanged suffix has matching incoming state.
Re-tokenize affected chunks between safe checkpoints with the existing production
regex lexer. Reuse unchanged chunks, relocating their UTF-16 ranges on assembly.
Conservative checkpoints deliberately retain ambiguous/unclosed constructs in a
larger chunk rather than resetting context or changing regex precedence. An edit
that changes an unclosed delimiter can legitimately invalidate the full suffix.
Source splitting/comparison and output assembly are still linear off-main, not a
rope or a constant-time algorithm. Keep the 256 KiB plain-text fallback.

Publish the cache only alongside tokens after the existing revision, source and
IME guards. Reset it on document/language/font/diff reset. Preserve attribute-only
application, selection and undo behavior.

Planned deterministic Swift gates: incremental versus fresh whole-source lexer
over insertion/removal/replacement edits, CRLF, surrogate pairs, escaped quotes,
block comments, triple quotes, template literals, markup overlays, split/join and
multiline state convergence; explicit reused/relexed chunk counters; production
editor stale/IME/undo fixtures retained. Add a mutation-sensitive propagation
assertion. Inspect diff and existing macOS CI registration. Swift unavailable in
this Linux environment; attempt availability, record diagnostics, never substitute
source assertions for compile or behavior. No Mac, benchmark, commit or push.

The requested formal-proof-skill and Swift reference are not present in the local
skill catalog; no formal proof is claimed. Compiler/behavioral gates remain needed.

## Implemented candidate and verification limits

- `EditorLexicalCache.update` is now the actual detached editor path. It compares
  line text, retains unchanged prefix state, advances through modified lines and
  stops state propagation when the unchanged suffix has the same incoming state.
  Neutral chunks reuse the existing `SyntaxHighlighter.tokens` renderer. No changes
  to the legacy renderer or its regex precedence were made.
- State covers language-specific line/block comments, ordinary/triple/backtick
  strings, SQL doubled quotes, literal shell quotes and independent overlapping
  markup comment/string/DOCTYPE passes. Unclosed or ambiguous constructs retain
  the suffix as one chunk. Invalid escaped newlines and unusual ICU line endings
  disable later checkpoints rather than guessing failed-regex backtracking.
- A checkpoint cannot split a function/markup-attribute whitespace lookahead.
  Final review caught and fixed the additional `data-1` / `data-` markup case:
  punctuation in attribute names must not be mistaken for a safe numeric end.
- Tokens remain contiguous UTF-16 runs. Reused chunk offsets are rebased and
  adjacent equal colors merged. Only a current revision with unchanged source and
  no marked text publishes either colors or the new cache. Resets discard it.
- Added 800 deterministic edit/full-source comparisons over ten language/edge
  fixtures, exact UTF-16 coverage checks, and direct production-helper assertions
  for one-line reuse, convergence, unclosed suffixes, language changes and diffs.
  The convergence assertion would reject a mutant that blindly reuses unchanged
  suffix text without comparing incoming state. Mutation execution is **not run**.
- `git diff --check` for the two owned Swift files passes. `zsh -n
  scripts/check-desktop.sh` passes. Existing desktop-and-protocol macOS CI invokes
  the updated responsiveness fixture through check-e2e/check-desktop.
- `swiftc --version` fails with command not found (127). Swift compilation,
  behavioral tests and mutation execution are **not executed**, not passed.
  No local proof tool is available. No AppKit/IME physical qualification, latency
  measurement, Mac access, commit, push, release or installation was performed.

Quality: candidate source, hosted Swift gates pending. Performance: unmeasured;
state propagation and regex work are incremental, but line comparison, chunk
assembly, token-range assembly and attribute comparison remain linear. Worst-case
delimiter edits can invalidate the full suffix. Integration: shared local files
only. No experiment or process remains running in this worker.
