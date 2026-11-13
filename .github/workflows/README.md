# GitHub Actions workflows

- `check.yml`: Python regression tests, native inventory, and Desktop/protocol checks.
- `rust.yml`: Rust tests and lint, Metal checks, and Kani proof harnesses split into shards.
- `apple-compatibility.yml`: Apple compatibility contracts.
- `profile.yml`: Flash profiling tool compatibility.
- `quantization.yml`: conversion and quantization contracts.
- `discord-release.yml`: release-related notifications.

Check each file for its exact triggers, environments, versions, and jobs. Hosted checks do not replace numerical validation on a physical GPU.
