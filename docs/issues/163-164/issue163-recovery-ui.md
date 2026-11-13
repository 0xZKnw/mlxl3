# PR163 recovery: download phases

The Hub progress decoder now forwards the CLI phase to the production transfer
tracker. `download` (and absent phase from older engines) retains ordinary byte
progress. `verify`, `publication`/`publish`, and `install` display phase labels and
indeterminate progress. Unknown phases also suppress potentially stale transfer
statistics. Transfer percentage, volume and rate are hidden during non-transfer
work, and rate samples reset when returning to download. Only the existing
terminal `installed` result marks the download complete. No hash pass was removed.

Regression cases were added to the existing download-progress Swift harness,
already invoked by scripts/check-desktop.sh: transfer to verification/publication/
installation, unknown phase, return to download without counting hashing time,
and legacy/terminal events. Existing finite-rate tests remain unchanged.

Validation on cloud Linux: formal-proof-skill lookup failed (unavailable); Swift
compiler unavailable. `git diff --check` passed for the final changes. Swift tests and visual macOS validation are not executed,
not passed. No Mac access, runtime hash changes, performance claims or commit.
The parent owns opti PR163-RECOVERY results and hosted CI follow-up.
