# Qwen Flash model components

[memory.rs](memory.rs) applies memory policies at load and request boundaries; [profile.rs](profile.rs) provides related profiling. Oracle and test files check GDN, HC, prepared convolution, and MTP prefill. The main model is in [qwen4.rs](../qwen4.rs).
