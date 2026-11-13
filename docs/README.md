# MLXL3 documentation

An entry point for navigating the repository and its archives. Reports are
organized by subject; their date, revision, and status matter more than their location.

## Start here

1. Read the [repository instructions](../AGENTS.md) and the [main README](../README.md).
2. Before optimization, read the [optimization journal `opti.md`](../opti.md),
   then the associated reports and evidence before proposing a new experiment.
3. Choose a section below or search for a model, issue, version, or date with
   `rg -n 'term' docs/ opti.md`.
4. Check each experiment's conditions, failures, limitations, and actual status:
   a prototype or historical result does not necessarily describe the product.

## Directory layout

- [Engine and configuration](engine/README.md)
- [Desktop and interface](desktop/README.md)
- Models: [Qwen](modeles/qwen/README.md), [LFM](modeles/lfm/README.md),
  [Ling](modeles/ling/README.md), [Gemma](modeles/gemma/README.md)
- Performance: [cross-cutting audits and kernels](performance/general/README.md),
  [Flash, DFlash, and SSD](performance/flash-dflash/README.md)
- Issues and reviews: [issues 163–164](issues/163-164/README.md),
  [other reviews](issues/reviews/README.md)
- Operations: [memory and hardware](operations/memory-hardware/README.md),
  [publishing](operations/release/README.md)
- Release notes: [Desktop](releases/desktop/README.md),
  [engine](releases/engine/README.md)
- [Measurements and campaigns](measurements/README.md): raw evidence organized
  by campaign, with previously loose files grouped into dedicated directories.
- [Native coverage](native-coverage/README.md): a historical register and
  evidence organized by component; check its date before using it.

Each report directory has its own index. Follow references to measurements
instead of treating an isolated raw file as validation. A few files at the root
redirect to reports cited by older releases. The `audit-runtime-2026-09-04.md`
link preserves a path used by tests.

## Add a document

Place the report in its primary subject directory and add it to that index.
Preserve campaign names, negative results, and limitations; update the
optimization journal and links when relevant.
