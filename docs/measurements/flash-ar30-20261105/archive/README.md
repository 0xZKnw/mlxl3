# AR30 lossless raw evidence

[index.json](index.json) records raw and gzip sizes/SHA256 for every archived log, event stream and output binary. Every stream passed gzip roundtrip equality; original files remain local and are not staged. The initial Clippy large-enum failure is preserved alongside the corrected run. JSON receipts and frozen stock metadata remain directly readable in the report directories.

Restore any entry by decompressing its `gzip` path to the corresponding `raw` path, then verify the raw SHA256. Historical input/checkpoint directories named by the freeze remain separate prerequisites; no model weights or installed app are included.
