# Qwen GPU state and MTP

[kv.rs](kv.rs) manages attention storage with an initialized prefix; [mtp.rs](mtp.rs) supplements the MTP path. These modules are connected through [qwen.rs](../qwen.rs) and used with the model configurations and state in [native/src/](../../README.md).
