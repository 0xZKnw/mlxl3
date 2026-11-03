# Native Flash NT2 default on current main

NT2 is enabled when `MLXL3_NATIVE_FLASH_NT2` is absent. Explicit `0` selects NT1; `1` selects NT2. Eligibility remains M5 resident K3/K4 MUL1, output at least 2560 and a multiple of 128. The kernels and excluded paths are unchanged.

This follow-up starts from main `17c66327` after PR112 was merged with the cache, packaging and HEADONLY stack. The earlier [isolated default controls](../default/verification.json) remain separate evidence. Current-main integration was rebuilt in its own package-clean target and requalified with the absent setting.

- [Final verification and limits](verification.json)
- [Frozen source/binary receipt](candidate.json)
- [Preregistered plan](plan.json) and [completed physical results](physical-result.json)
- [Actual AR report](ar.json), [artifact hashes](artifacts.json) and [raw logs](logs.tar.gz)

`candidate.json` and `plan.json` preserve their pre-launch status. The completed result and verification files carry the final outcome. Quality elapsed times are diagnostics, not a new throughput comparison. The earlier explicit-flag AR32 short screen is preserved separately; no installed-app or release update is claimed.
