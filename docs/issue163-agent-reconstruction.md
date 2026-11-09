# Issue 162 D03–D05 reconstruction

Local changes reconstructed on PR163 recovery tree `f428945b` (tree identical to published `2ddde23`), with merged PR161 retained. This report does not qualify all 42 audit findings or claim a performance gain.

## Production changes

- D03: `AgentProjectStore` persists message-to-Git-checkpoint associations outside the model-writable workspace. `StudioModel.send` records the pre-user boundary before clearing the draft; successful completion and acknowledged cancellation record the assistant boundary. Branching copies the private repository into an independent workspace and resets that copy to the selected boundary. Current tool restrictions, source project, apply baseline and original hashes follow the branch. Later file edits cannot leak between copies. Missing historical associations (including old conversations and terminal failures without an acknowledged checkpoint) fail visibly instead of substituting the latest files. Unacknowledged termination is intentionally not snapshotted while tools might still write.
- Applying one branch does not silently update another branch's baseline. Overlapping source edits conflict using each branch's recorded source hashes and baseline executable modes. Nonoverlapping changes remain independently applicable.
- D04: regeneration creates a ticket tied to conversation, model-load identity, draft and attachments. It waits up to 30 seconds for readiness and the exact workspace's MCP acknowledgment, consumes the ticket before sending, and cancels on navigation, Stop, service shutdown, errors or changed inputs. The native `set_mcp` handler echoes `request_id`; the Swift bridge sends it. Stale correlated replies are ignored. Legacy uncorrelated replies retain manual Send compatibility, but never authorize automatic regeneration; the draft remains available with an update explanation.
- D05: apply rereads Git tree bytes plus executable mode, checks the source mode against its baseline before any writes, applies the approved mode, and records full source modes in rollback snapshots. Baseline metadata is saved inside the rollback scope. A failed write or failed metadata save cannot intentionally advance the baseline. Concurrent external filesystem changes after preflight remain outside an atomic cross-file transaction; this patch does not claim filesystem isolation against another writer.

## Regression coverage added

`tests/studios-check.swift` exercises real Git import/checkpoint/fork/apply entry points: historical bytes versus newer files, preserved denied run/write permissions, source association, executable mode, mutation isolation, absent checkpoint rejection, 644→755, 755→644, new executable, concurrent original chmod conflict, and a late parent-path collision requiring restoration of deleted executable bytes/mode and changed permissions. The failed apply must retain its prior baseline.

`tests/studio-modes-check.swift` exercises the real StudioModel/bridge fixture path: a delayed MCP handshake, a stale acknowledgment, generated response content, exactly one send after additional time, navigation cancellation, Stop while waiting, and rejection of an uncorrelated legacy acknowledgment. `tests/fake-mode-engine.py` delays MCP status by 200 ms and echoes the request ID.

These are behavioral tests, not source-text assertions. Existing `scripts/check-desktop.sh` runs the modes harness and `scripts/check-studios.sh`; the existing macOS `desktop-and-protocol` CI job invokes them via `scripts/check-e2e.sh`. No additional parallel CI compilation was added.

## Checks actually run in Linux

- `git diff --check`: passed for the integrated working tree at review time.
- Python AST parsing of the modified fake engine: passed.
- Disposable Python subprocess probe: ready → delayed MCP status → generated delta → complete passed, observed handshake delay 0.201 seconds. Child terminated and joined. This validates the fixture only, not Swift's handling.
- First fixture probe was blocked by the read-only shell sandbox's temporary-directory restriction, before creating its subprocess. The same disposable probe succeeded with approved write access.
- `swiftc --version`: command unavailable. No Swift typecheck, production Swift harness execution, mutation run, GUI session or end-to-end Agent generation was performed here. The new tests are **unexecuted**, pending Apple CI; original-bug detection is not claimed as executed.
- Default Python lacked Ruff; no Python lint success claimed. The default shell lacked Cargo, then `/home/agent/.cargo/bin/cargo kani --version` reported no such subcommand. Kani therefore remains unavailable. The native echo is included in the parent's integrated Rust validation rather than launching a competing build.
- The requested formal-proof skill was absent from the available skill catalog and selected local skill roots. No Swift source verifier was available, and no formal proof is claimed.

No Mac, model, GPU, push, merge or release was used by this worker. All subprocesses launched by the probe are closed. Apple compilation and the behavioral/mutation gates are the remaining validation limit, not a completed result.
