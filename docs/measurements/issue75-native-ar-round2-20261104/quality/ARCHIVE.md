# Composed physical quality evidence

`report21-summary.json` joins the14 essential pre-pilot and7 deferred post-pilot
phases. Both historical reports remain separate. All21 phases pass; each
owned process was joined and its process group empty before the next phase.
Artifacts are the same789a CLI/d3f6 libtest throughout; root owns subsequent
CPU/compat checks and default-path changes.

Exact subprocess commands, bounds and per-phase setting overrides are in the
two reports, with the common controlled environment in the summary. Small
component counter reports and logs are gzip-compressed losslessly; the
manifest records compressed and uncompressed SHA256s. No raw trace or global
environment dump is included.

The independent stock corpus, staged packets and numerical binary outputs
remain in `build/ar-round2/quality-01` on the qualification Mac. Their paths
and hashes are recorded; the large arrays are not copied into this archive.
The original48 archived references were not regenerated. Staged JSON uses
new serialization and `fixed_key:null`; every original native field and all
stock integer arrays match exactly, as the provenance receipt documents.
Reproduction needs the real Flash checkpoint and retained local corpus.
The one-completion short wrapper is transient under `build/ar-round2`; its
completion validator and JSON transport come from existing repository code.

The successful pilot's performance is separate under the sibling
`performance/` directory. These quality runs supply no extra throughput gain
and no complete proof of Metal, allocation or concurrency.
