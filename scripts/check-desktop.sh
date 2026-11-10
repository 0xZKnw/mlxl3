#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
preferences_only=false
if [[ "${1:-}" == --preferences-only && $# == 1 ]]; then
    preferences_only=true
elif (( $# != 0 )); then
    print -u2 "Usage: $0 [--preferences-only]"
    exit 2
fi
test_python="${MLXL3_TEST_PYTHON:-}"
if [[ -z "${test_python}" ]]; then
    test_python="$(command -v python3)"
fi
test_dir="$(mktemp -d -t mlxl3-check)"
trap 'rm -rf "${test_dir}"' EXIT

# The new I/O fixtures must also be reaped if a Swift assertion aborts the check.
run_fixture_check() {
    "${test_python}" - "$@" <<'PY'
import os
import signal
import subprocess
import sys

process = subprocess.Popen(sys.argv[1:], start_new_session=True)
try:
    code = process.wait(timeout=90)
except subprocess.TimeoutExpired:
    print("Desktop fixture exceeded its 90-second limit", file=sys.stderr)
    code = 1
finally:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)
sys.exit(code)
PY
}

# CI normally selects the active Xcode toolchain. A local Command Line Tools
# install can opt into an installed SDK explicitly (for example 26.5).
swift_args=(--package-path apps/MLXL3Studio)
swiftc_args=(-warnings-as-errors -swift-version 6 -parse-as-library -module-cache-path "${test_dir}/modules")
if [[ -n "${MLXL3_MACOS_SDK:-}" ]]; then
    # shortcut: native honors this SDK override; revisit when Swift Build handles cached modules correctly.
    swift_args+=(--build-system native --sdk "${MLXL3_MACOS_SDK}")
    swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
fi
swift build "${swift_args[@]}"
binary_dir="$(swift build "${swift_args[@]}" --show-bin-path)"
sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
swiftc_args+=(-I "${binary_dir}")
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    apps/MLXL3Studio/Sources/MLXL3Studio/ContextMemoryProfile.swift \
    tests/context-memory-check.swift -o "${test_dir}/context-memory-check"
"${test_dir}/context-memory-check"
[[ -d "${binary_dir}/Modules" ]] && swiftc_args+=(-I "${binary_dir}/Modules")
if [[ -f "${binary_dir}/SwiftMath.o" ]]; then
    math_objects=("${binary_dir}/SwiftMath.o")
else
    math_objects=("${binary_dir}/SwiftMath.build"/*.swift.o)
fi

# Both CI paths compile and run the same registered response app and fixtures.
check_response_ui() {
    local -a response_sources
    response_sources=("${(@)sources:#*/MarkdownResponseView.swift}")
    local response_app="${test_dir}/MLXL3 Response Check.app"
    awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/MarkdownResponseView.swift \
        tests/table-rendering-check.swift \
        tests/response-ui-check.swift > "${test_dir}/ResponseUICombined.swift"
    mkdir -p "${response_app}/Contents/MacOS"
    cp tests/response-check-Info.plist "${response_app}/Contents/Info.plist"
    swiftc "${swiftc_args[@]}" -O \
        "${response_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/ResponseUICombined.swift" -o "${response_app}/Contents/MacOS/response-check"
    print "Response UI Swift 6 compilation passed"
    run_fixture_check "${response_app}/Contents/MacOS/response-check" "$PWD/tests/fake-desktop-engine.py"
    # A fresh process runs the fixed-size native bitmap/OCR checks.
    run_fixture_check "${response_app}/Contents/MacOS/response-check" "$PWD/tests/fake-desktop-engine.py" --check-long-visibility
}

# The focused PR job and the full suite execute the same production fixtures.
check_preferences() {
    local -a idle_sources
    idle_sources=("${(@)sources:#*/StudioModel.swift}")
    # Appearance persistence, startup gate and actual subprocess lifecycle.
    awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
        tests/desktop-preferences-check.swift > "${test_dir}/DesktopPreferencesCombined.swift"
    swiftc "${swiftc_args[@]}" -warnings-as-errors \
        "${idle_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/DesktopPreferencesCombined.swift" -o "${test_dir}/preferences-check"
    run_fixture_check "${test_dir}/preferences-check" "$PWD/tests/fake-desktop-engine.py"

    awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
        tests/completion-notifications-check.swift > "${test_dir}/CompletionNotificationsCombined.swift"
    swiftc "${swiftc_args[@]}" -warnings-as-errors \
        "${idle_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/CompletionNotificationsCombined.swift" -o "${test_dir}/notification-check"
    run_fixture_check "${test_dir}/notification-check" "$PWD/tests/fake-desktop-engine.py"
    swiftc "${swiftc_args[@]}" "${sources[@]}" "${math_objects[@]}" \
        tests/logo-appearance-check.swift -o "${test_dir}/logo-appearance-check"
    run_fixture_check "${test_dir}/logo-appearance-check"
    if ${preferences_only}; then
        check_response_ui
    fi
}

if ${preferences_only}; then
    check_preferences
    print "Desktop preferences, notifications, logo and response UI checks passed"
    exit 0
fi

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/UpdateManager.swift \
    tests/studio-hardening-check.swift > "${test_dir}/StudioHardeningCombined.swift"
hardening_sources=("${(@)sources:#*/UpdateManager.swift}")
swiftc "${swiftc_args[@]}" \
    "${hardening_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/StudioHardeningCombined.swift" -o "${test_dir}/check"
"${test_dir}/check" "$PWD/tests/fake-desktop-engine.py"

# Check the production idle deadline and lifecycle without waiting minutes or
# loading a model. File concatenation gives the fixture private event access.
awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/model-idle-unload-check.swift > "${test_dir}/ModelIdleUnloadCombined.swift"
idle_sources=("${sources[@]}")
idle_sources=("${(@)idle_sources:#*/StudioModel.swift}")

swiftc "${swiftc_args[@]}" -O \
    apps/MLXL3Studio/Sources/MLXL3Studio/AgentSandbox.swift \
    tests/agent-sandbox-check.swift -o "${test_dir}/agent-sandbox-check"
"${test_dir}/agent-sandbox-check"
"${test_python}" tests/check-agent-protocol.py "${test_dir}/agent-sandbox-check"

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/studio-modes-check.swift > "${test_dir}/StudioModesCombined.swift"
swiftc "${swiftc_args[@]}" \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/StudioModesCombined.swift" -o "${test_dir}/studio-modes-check"
"${test_dir}/studio-modes-check" "$PWD/tests/fake-mode-engine.py"

swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/ModelIdleUnloadCombined.swift" -o "${test_dir}/idle-unload-check"
"${test_dir}/idle-unload-check" "$PWD/tests/fake-desktop-engine.py"

check_preferences

# Keep global speculative preferences independent of current-model sampling.
awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/speculative-sampling-check.swift > "${test_dir}/SpeculativeSamplingCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/SpeculativeSamplingCombined.swift" -o "${test_dir}/sampling-check"
"${test_dir}/sampling-check" "$PWD/tests/fake-desktop-engine.py" \
    "$PWD/apps/MLXL3Studio/Sources/MLXL3Studio/GenerationInspector.swift"

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/desktop-experience-check.swift > "${test_dir}/DesktopExperienceCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/DesktopExperienceCombined.swift" -o "${test_dir}/experience-check"
run_fixture_check "${test_dir}/experience-check" "$PWD/tests/fake-desktop-engine.py"

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/response-text-size-check.swift > "${test_dir}/ResponseTextSizeCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/ResponseTextSizeCombined.swift" -o "${test_dir}/text-size-check"
run_fixture_check "${test_dir}/text-size-check"

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/mtp-panel-layout-check.swift > "${test_dir}/MTPPanelLayoutCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/MTPPanelLayoutCombined.swift" -o "${test_dir}/mtp-layout-check"
run_fixture_check "${test_dir}/mtp-layout-check"

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/StudioModel.swift \
    tests/issue163-runtime-contract-check.swift > "${test_dir}/RuntimeContractCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    "${idle_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/RuntimeContractCombined.swift" -o "${test_dir}/runtime-contract-check"
run_fixture_check "${test_dir}/runtime-contract-check"

"${test_python}" tests/update-archive-fixtures.py "${test_dir}/archives"
swiftc "${swiftc_args[@]}" \
    "${sources[@]}" "${math_objects[@]}" \
    tests/updater-check.swift -o "${test_dir}/updater-check"
"${test_dir}/updater-check" "${test_dir}/archives" ${MLXL3_TEST_ENGINE_ARCHIVE:+"${MLXL3_TEST_ENGINE_ARCHIVE}"} ${MLXL3_TEST_ENGINE_VERSION:+"${MLXL3_TEST_ENGINE_VERSION}"}

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/UpdateManager.swift \
    tests/updater-application-check.swift > "${test_dir}/UpdaterApplicationCombined.swift"
swiftc "${swiftc_args[@]}" -warnings-as-errors \
    apps/MLXL3Studio/Sources/MLXL3Studio/EngineRuntimeStore.swift \
    apps/MLXL3Studio/Sources/MLXL3Studio/Localization.swift \
    apps/MLXL3Studio/Sources/MLXL3Studio/ModelDownloadProgress.swift \
    "${test_dir}/UpdaterApplicationCombined.swift" -o "${test_dir}/updater-application-check"
"${test_dir}/updater-application-check" "${test_dir}/archives" ${MLXL3_TEST_APP_DMG:+"${MLXL3_TEST_APP_DMG}"} ${MLXL3_TEST_APP_VERSION:+"${MLXL3_TEST_APP_VERSION}"}

awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/UpdateManager.swift \
    tests/updater-progress-check.swift > "${test_dir}/UpdaterProgressCombined.swift"
swiftc "${swiftc_args[@]}" \
    apps/MLXL3Studio/Sources/MLXL3Studio/EngineRuntimeStore.swift \
    apps/MLXL3Studio/Sources/MLXL3Studio/Localization.swift \
    apps/MLXL3Studio/Sources/MLXL3Studio/ModelDownloadProgress.swift \
    "${test_dir}/UpdaterProgressCombined.swift" -o "${test_dir}/updater-progress-check"
run_fixture_check "${test_dir}/updater-progress-check" "${test_python}" "$PWD/tests/desktop-transfer-fixture.py"

swiftc "${swiftc_args[@]}" "${sources[@]}" "${math_objects[@]}" \
    tests/html-preview-check.swift -o "${test_dir}/html-preview-check"
run_fixture_check "${test_dir}/html-preview-check" "${test_python}" "$PWD/tests/desktop-transfer-fixture.py"

swiftc "${swiftc_args[@]}" \
    "${sources[@]}" "${math_objects[@]}" \
    tests/file-import-check.swift -o "${test_dir}/file-import-check"
"${test_dir}/file-import-check" "$PWD/tests/fake-desktop-engine.py"

swiftc "${swiftc_args[@]}" \
    "${sources[@]}" "${math_objects[@]}" \
    tests/image-input-check.swift -o "${test_dir}/image-input-check"
run_fixture_check "${test_dir}/image-input-check" "$PWD/tests/fake-desktop-engine.py" "$PWD/tests/fixtures/image-input"

swiftc "${swiftc_args[@]}" \
    "${sources[@]}" "${math_objects[@]}" \
    tests/workspace-features-check.swift -o "${test_dir}/workspace-features-check"
"${test_dir}/workspace-features-check" "$PWD/tests/fake-desktop-engine.py"

swiftc "${swiftc_args[@]}" "${sources[@]}" "${math_objects[@]}" \
    tests/responsiveness-check.swift -o "${test_dir}/responsiveness-check"
"${test_dir}/responsiveness-check"

"${binary_dir}/MLXL3Studio" --check-chat-timeline
"${binary_dir}/MLXL3Studio" --check-mcp-preferences

# Exercise the real Swift bridge against a subprocess: callback ordering,
# stale-generation suppression, reload, cancellation and crash handling.
awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift \
    tests/bridge-dispatch-check.swift > "${test_dir}/BridgeDispatchCombined.swift"
bridge_sources=("${sources[@]}")
bridge_sources=("${(@)bridge_sources:#*/MLXL3Bridge.swift}")
swiftc "${swiftc_args[@]}" -O \
    "${bridge_sources[@]}" "${math_objects[@]}" \
    "${test_dir}/BridgeDispatchCombined.swift" -o "${test_dir}/bridge-check"
"${test_dir}/bridge-check" "$PWD/tests/fake-desktop-engine.py"

# Exercise incremental Markdown/code rendering and Unicode chunk boundaries
# through AppKit/SwiftUI, not only the pure Python parser tests.
swiftc "${swiftc_args[@]}" -O \
    "${sources[@]}" "${math_objects[@]}" tests/streaming-render-check.swift \
    -o "${test_dir}/streaming-check"
"${test_dir}/streaming-check"

check_response_ui

# Exercise the cancellable CLICommand transport independently of the model
# bridge (large pipe output, line progress, cancellation and stderr errors).
swiftc "${swiftc_args[@]}" -O apps/MLXL3Studio/Sources/MLXL3Studio/CLICommand.swift \
    tests/cli-command-check.swift -o "${test_dir}/cli-command-check"
"${test_dir}/cli-command-check" "${test_python}"

swiftc "${swiftc_args[@]}" -O apps/MLXL3Studio/Sources/MLXL3Studio/ModelDownloadProgress.swift \
    tests/download-progress-check.swift -o "${test_dir}/download-progress-check"
"${test_dir}/download-progress-check"

swiftc "${swiftc_args[@]}" -O apps/MLXL3Studio/Sources/MLXL3Studio/MTPTuning.swift \
    tests/mtp-tuning-check.swift -o "${test_dir}/mtp-tuning-check"
"${test_dir}/mtp-tuning-check"

swiftc "${swiftc_args[@]}" -O apps/MLXL3Studio/Sources/MLXL3Studio/MCPConfiguration.swift \
    apps/MLXL3Studio/Sources/MLXL3Studio/Localization.swift tests/mcp-configuration-check.swift \
    -o "${test_dir}/mcp-configuration-check"
"${test_dir}/mcp-configuration-check"

# Beta 5 project/artifact integration shares the already-built app sources.
MLXL3_STUDIOS_TMP="${test_dir}" zsh scripts/check-studios.sh

# CPU-only full-window Studio navigation, editing and reviewed snapshot safety.
zsh scripts/check-studio-workspace.sh
print "Desktop E2E checks passed: lifecycle, file imports, bridge, streaming/rendering, CLI transport and Studios"
