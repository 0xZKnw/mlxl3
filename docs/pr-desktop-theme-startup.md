# Desktop themes, completion notifications and startup recovery

Updates PR #72 onto main `b9d5cbdce50aa98adfee560e06da75614e730bd3`. Adds persisted System/Light/Dark/Midnight Blue themes, optional startup loading with an explicit model picker, opt-in background completion notifications and same-model retry after a failed load.

The integration preserves Agent registry setup, Server/Lab modes, bounded cancellation/watchdogs, process retirement and current code-preview/scroll behavior. Model-load tasks are cancelled when ejecting or stopping services; a completion racing with user/watchdog cancellation does not notify.

The engine and Rust workflow are byte-identical to main. The older RAM commit remains in history with its measurements; its distinct BF16 codec is deferred for separate qualification against the current engine. The historical 806–867 MB reduction is not a measurement of this integrated PR.

Cloud validation and remaining macOS/physical-model limits: [integration report](pr72-cloud-integration-20261103.md). No merge, release or installed-app replacement.
