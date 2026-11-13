# PR163 recovery and remaining scope

## Provenance

On 9 November the previous cloud checkout, temporary restoration files and execution stores were no longer available. The unpublished broad working tree recorded as 3cd484bc could not be recovered as a complete verified artifact. It was never the GitHub PR head. Old remote head818cc3ff remains the source of the original N04/N05 implementation and its historical evidence.

The current recovery merges main0fd9fbd1 (which includes PR161) into that head. Conflicts in opti.md, scripts/kani_obligations.json and tests/test_kani_ci.py preserve both journals and the union of proof names. No force update, merge of PR163, release, installed app change or user Mac access is authorized here.

## Reconstructed review fixes

- Final output size limits no longer reject large intermediate HTTP redirect bodies. Initial and resumed transfers retain hash verification and final-size enforcement.
- Completed staging is distinguished from resumable staging; a new attempt invalidates its previous completion marker before fallible work under the download lock. This is not full crash/registry reconciliation.
- CLI and Desktop expose download, verification, publication and installation phases. Both hash passes remain; there is no hash-byte progress counter.
- Swift sampling wiring guards tolerate formatting while retaining positive and negative checks. The RGB type-checker simplification is already in current main.

## Validation boundary

New Linux evidence is separate from the lost broad-lot logs. Kani orchestration/inventory tests:49 passed in64.26s; these do not execute proofs. Python source oracle checked39 formatting/wiring cases; this is not Swift compilation or semantic verification. Swift/compiler, AppKit, Metal, real model parity and speed tests were not executed. See the Hub and UI recovery reports and new logs for their exact results. CI must be inspected on the actual published commit, not an older PR run.

## Entire issue162 remains incomplete

This recovery does not restore or claim closure of all42 findings. N04/N05 and the review corrections above are the active scoped changes. N01–N03/N06–N09, P01–P08, D01–D10, U01–U05, T01–T07 and C01–C03 need reconciliation against current main before each can be closed. Some have changes already merged through PR161; that is not a blanket acceptance claim. The previous broad unpublished implementation and its test counts cannot be treated as delivered or as validation of this reconstructed source.

Known unfinished acceptance from the previous plan includes producer-bound MatrixView completion ownership, full GPU/compact-draft loading, immutable conversation snapshots, fully incremental editor lexing, tokenizer provenance for literal markers, transport process containment, and physical Apple qualification. Full installation crash recovery, malicious staging races, live authenticated Hub downloads and hardware throughput remain unverified. No40tokens/s or other speed gain is claimed.
