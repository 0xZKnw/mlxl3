#!/bin/zsh
# REL-ENGINE-2.0.11-BETA2: real Desktop EngineRuntimeStore install of the archive into a disposable store.
# Temporary driver = tests/updater-check.swift with the release tag read from MLXL3_TEST_ENGINE_TAG (updater-driver.diff).
set -eu
setopt pipefail
S=/private/tmp/claude-501/-Users-justin--claude-spares-spare-Pqz2cz/a7b58a63-1981-431b-af7f-59575a979a8a/scratchpad/updater-rel-b2
W=/Users/justin/.codex/worktrees/rel-2.0.11-beta2/mix-stq1_0
cd $W
source build/env.sh
# The default SDK 27.2 lacks the SwiftUIMacros plugin with Command Line Tools (attempt 1); use 26.5 like build-engine.sh.
export MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
swift_args=(--package-path apps/MLXL3Studio --scratch-path $S/swiftbuild)
swiftc_args=(-warnings-as-errors -swift-version 6 -parse-as-library -module-cache-path $S/modules)
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
  swift_args+=(--build-system native --sdk "$MLXL3_MACOS_SDK")
  swiftc_args+=(-sdk "$MLXL3_MACOS_SDK")
fi
swift build "${swift_args[@]}"
bin="$(swift build "${swift_args[@]}" --show-bin-path)"
swiftc_args+=(-I "$bin")
if [[ -d "$bin/Modules" ]]; then swiftc_args+=(-I "$bin/Modules"); fi
if [[ -f "$bin/SwiftMath.o" ]]; then math=("$bin/SwiftMath.o"); else math=("$bin"/SwiftMath.build/*.swift.o); fi
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
mkdir -p $S/archives
python3 tests/update-archive-fixtures.py $S/archives
swiftc "${swiftc_args[@]}" "${sources[@]}" "${math[@]}" $S/updater-check-tag.swift -o $S/updater-check
MLXL3_TEST_ENGINE_TAG=engine-v2.0.11-beta.2 $S/updater-check $S/archives $W/dist/MLXL3-Engine-v2.0.11-beta.2-arm64.tar.gz 2.0.11
echo updater_status=0
