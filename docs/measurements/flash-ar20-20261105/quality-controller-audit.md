# AR20 quality controller audit

Scope: source audit and 39 deterministic synthetic Python checks of frozen root
`build/ar20/quality.py`, SHA-256
`0e495cb9107127b4fa28644e7da07c59ee005a201ad5ac7ce95bf0f9ea40d5de`.
No root files were modified. No model, Cargo, GPU, compiler or prover was run.

The checked source matches `native/src/gpu/ar20_capture_tests.rs`:

- `COUNTS`: 2 cases, 57,606,144 state words, 993,280 logits, 81,920 hidden
  bytes, 84 initial archived states, and readers joined. Reference legitimately
  has `parity=false/status=reference_captured`; candidate requires
  `parity=true/status=validated`.
- Reference uses all 12 new options OFF, including immutable kernel macros,
  spin, read-ahead and SLRU. Baseline 13 ALL12 flags remain ON. Default candidate
  uses 11 options, excluding the regressing B01.
- Per row, QSA chains/prefetch deltas are 12, resident bundle 36, finish/mix 48,
  and head gates zero by default. L1/MoE deltas are 34. All integer deltas reject
  booleans, negatives and decreases. Zero in-flight readers and exact compiled
  bits are required. Spin may legitimately use fallback only; read-ahead
  descriptors are absolute initialized counts. SLRU requires 48 layers and
  positive reservation activity.
- Rust's typed oracle parser rejects unknown/duplicate fields; its `validate`
  checks row tokens/offsets, 84 unique nonempty tensor names, finite nonempty
  shape extents, checked byte sizes, SHA fields and exact total state count.
  Candidate compares complete typed rows, including every state hash, and
  verifies oracle bytes did not change. Python independently checks the oracle
  SHA before/after comparison and CLI. Python does not independently reproduce
  every Rust oracle structural check; this is an explicit component boundary.
- Exactly three successful phases and joined process groups are required for
  global qualification. CLI uses the existing two-token golden/no-MLX/shutdown
  helper. The reference phase's false parity is not promoted individually.

One reproduced audit finding: the reused `OLD['marker']` checks one marker line
but parses with the older JSON parser, accepting duplicate object keys. For
example, `{"parity":false,"parity":true}` becomes true. Production Rust uses
`serde_json::json!` and does not emit duplicate keys, so this is not evidence of
incorrect physical output, but malformed evidence should still be rejected.
Proposed minimal correction, not applied to frozen root: keep the existing
unique-marker-line check, then parse its payload through `COMPONENTS['strict']`,
as the component controller already does. Root was notified before any edit.

Command:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/justin/Documents/mix-stq1_0/.venv/bin/python docs/measurements/flash-ar20-20261105/quality-controller-selfcheck.py > docs/measurements/flash-ar20-20261105/quality-controller-selfcheck.json
```

Result: 39/39 expected accepts/rejections passed, and the duplicate-key marker
false positive was retained explicitly. Invalid/duplicate selections,
reference/candidate/subset activations, missing/wrong/decreasing counters, OFF
activations, count type mismatches, incomplete/duplicate/nonfinite JSON, and
duplicate marker lines were exercised. Source SHA was unchanged after import.
These are adversarial sampled checks, not a proof. Full subprocess timeout,
cleanup and production CLI execution are left to root's physical campaign.
CrossHair was not run during the exclusive GPU grant (root already recorded it
unavailable); no formal Python verification is claimed.

## Correction and offline requalification

Parent authorized the minimal correction after the physical 3/3 campaign
completed. Root `quality.py` now preserves the OLD unique-line check and strictly
reparses the payload through the existing duplicate/nonfinite rejecting parser;
root `speed.py` uses that same strict helper in its nested numerical callback.
The immediate `delta` closure binds its row dictionaries explicitly to satisfy
Ruff B023. No Rust/binary changes or new physical run were required.

47/47 checks pass in `quality-controller-selfcheck-after.json`: previous 39,
the duplicate false/true regression against the new production helper, four
speed-arm invariants (8 unique arms, selected10 excluding B01/B12, subsets only,
frozen old-binary controls first/last), and strict reparsing of all three existing
native quality logs. Parsed evidence equals the saved reports, and the actual
reference/candidate counters also pass `paths`. The source and snapshot files
were byte-identical after the edit. Quality SHA is now
`488ad1754cf2ce92d0b3c633ab960866cb61fb38bc3f2510d33f375a89892ec6`;
the speed SHA is recorded in the receipt because root separately removed an
unused import. The old receipt and reproduced OLD-parser weakness remain intact.

The callback remains nested in `main` and was audited in source rather than
executing the speed controller. Its exact helpers and accepted counter domain
were exercised above; subprocess production qualification remains root-owned.
