#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
# Usage: check-desktop.sh [--preferences-only | --skip-preferences]
#   --preferences-only  only the icon, context memory, voice, preferences,
#                       notifications, logo and response UI units (focused CI job).
#   --skip-preferences  every other unit; CI runs it beside --preferences-only so
#                       the two jobs together execute the full suite exactly once.
# MLXL3_CHECK_JOBS sets how many swiftc compilations run at the same time.
mode=full
if [[ "${1:-}" == --preferences-only && $# == 1 ]]; then
    mode=preferences
elif [[ "${1:-}" == --skip-preferences && $# == 1 ]]; then
    mode=skip-preferences
elif (( $# != 0 )); then
    print -u2 "Usage: $0 [--preferences-only | --skip-preferences]"
    exit 2
fi
test_python="${MLXL3_TEST_PYTHON:-}"
if [[ -z "${test_python}" ]]; then
    test_python="$(command -v python3)"
fi
test_dir="$(mktemp -d -t mlxl3-check)"
compile_dir="${test_dir}/compile"
mkdir -p "${compile_dir}"
compile_pool=""
process_tree() {
    local child
    print -r -- "$1"
    for child in $(pgrep -P "$1"); do
        process_tree "${child}"
    done
}
cleanup() {
    # Freeze the whole compile pool until no new process appears, then kill it,
    # so neither xargs nor a swift-driver can start another compilation.
    if [[ -n "${compile_pool}" ]]; then
        local -a frozen current
        current=($(process_tree "${compile_pool}"))
        while (( ${#current} > ${#frozen} )); do
            frozen=("${current[@]}")
            kill -STOP "${frozen[@]}" 2>/dev/null
            current=($(process_tree "${compile_pool}"))
        done
        kill -KILL "${current[@]}" 2>/dev/null
    fi
    rm -rf "${test_dir}"
}
trap cleanup EXIT
# zsh does not run the EXIT trap after exiting from a signal trap.
trap 'cleanup; trap - EXIT; exit 130' INT TERM HUP

# The new I/O fixtures must also be reaped if a Swift assertion aborts the check.
run_fixture_check() {
    "${test_python}" - "${fixture_timeout:-90}" "$@" <<'PY'
import os
import signal
import subprocess
import sys

timeout = int(sys.argv[1])
process = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    code = process.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    print(f"Desktop fixture exceeded its {timeout}-second limit", file=sys.stderr)
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

# Compilations are independent of each other and of the checks, so they are all
# queued first and run in parallel while the checks execute one at a time, in
# the same order and with the same binaries as a sequential run.
typeset -a compile_jobs
queue_compile() {
    local name=$1
    shift
    local job="${compile_dir}/$(printf '%03d' $(( ${#compile_jobs} + 1 )))-${name}.zsh"
    local log="${compile_dir}/${name}.log" result="${compile_dir}/${name}.status"
    [[ ! -e "${log}" ]] || { print -u2 "Duplicate compile job ${name}"; exit 1; }
    : > "${log}"
    {
        print -r -- "zmodload zsh/datetime; cd ${(qq)PWD} || exit 0"
        print -r -- "start=\${EPOCHREALTIME}"
        print -r -- "swiftc ${(j: :)${(qq)swiftc_args[@]}} ${(j: :)${(qq)@}} > ${(qq)log} 2>&1"
        print -r -- "code=\$?"
        print -r -- "printf '%d %.1f\\n' \${code} \$(( EPOCHREALTIME - start )) > ${(qq)result}.tmp"
        print -r -- "mv ${(qq)result}.tmp ${(qq)result}"
        print -r -- "exit 0"
    } > "${job}"
    compile_jobs+=("${job}")
}

compile_job_count() {
    if [[ -n "${MLXL3_CHECK_JOBS:-}" ]]; then
        [[ "${MLXL3_CHECK_JOBS}" == <1-64> ]] || { print -u2 "MLXL3_CHECK_JOBS must be 1-64"; exit 2; }
        print -r -- "${MLXL3_CHECK_JOBS}"
        return
    fi
    # A full-app swiftc uses several cores and peaked at 1.7 GB (MacOSX26.5 SDK),
    # so allow one job per two cores (3-core CI runner: 2) and 2 GB per job.
    local cores memory jobs
    cores=$(sysctl -n hw.ncpu)
    memory=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    jobs=$(( (cores + 1) / 2 ))
    (( jobs <= memory / 2 )) || jobs=$(( memory / 2 ))
    (( jobs >= 1 )) || jobs=1
    print -r -- "${jobs}"
}

start_compiles() {
    local jobs
    jobs=$(compile_job_count)
    print "Compiling ${#compile_jobs} Swift check binaries, ${jobs} at a time"
    # Disowned (&!): zsh can crash reaping a stopped job while cleanup kills the pool.
    print -rl -- "${compile_jobs[@]}" | xargs -P "${jobs}" -n 1 zsh &!
    compile_pool=$!
}

# Waits for one binary; a failed, missing or stalled compilation fails the suite.
await_compile() {
    local name=$1
    local result="${compile_dir}/${name}.status" log="${compile_dir}/${name}.log"
    local deadline=$(( SECONDS + 1800 ))
    [[ -e "${log}" ]] || { print -u2 "No compile job named ${name}"; exit 1; }
    while [[ ! -s "${result}" ]]; do
        if ! kill -0 "${compile_pool}" 2>/dev/null; then
            [[ -s "${result}" ]] && break
            print -u2 "Compilation ${name} never finished"
            exit 1
        fi
        if (( SECONDS > deadline )); then
            print -u2 "Compilation ${name} exceeded 30 minutes"
            exit 1
        fi
        sleep 0.2
    done
    local code seconds
    read -r code seconds < "${result}"
    [[ -s "${log}" ]] && cat "${log}"
    if [[ "${code}" != 0 ]]; then
        print -u2 "Compilation ${name} failed with status ${code}"
        exit 1
    fi
    print "Compiled ${name} in ${seconds} s"
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
app_sources=apps/MLXL3Studio/Sources/MLXL3Studio

# Combined sources give a fixture private access to one production file.
combine() {
    local output="${test_dir}/$1"
    shift
    awk '{ print }' "$@" > "${output}"
}

# ----- Queue every compilation (cheap), in the order the checks need them -----

queue_early() {
    queue_compile context-memory -warnings-as-errors \
        "${app_sources}/ContextMemoryProfile.swift" \
        tests/context-memory-check.swift -o "${test_dir}/context-memory-check"
    combine VoiceCallbackCombined.swift "${app_sources}/VoiceController.swift" tests/voice-callback-check.swift
    queue_compile voice-callback "${app_sources}/Localization.swift" \
        "${test_dir}/VoiceCallbackCombined.swift" -o "${test_dir}/voice-callback-check"
}

queue_response_ui() {
    local -a response_sources
    response_sources=("${(@)sources:#*/MarkdownResponseView.swift}")
    response_app="${test_dir}/MLXL3 Response Check.app"
    combine ResponseUICombined.swift "${app_sources}/MarkdownResponseView.swift" \
        tests/table-rendering-check.swift tests/response-ui-check.swift
    mkdir -p "${response_app}/Contents/MacOS"
    cp tests/response-check-Info.plist "${response_app}/Contents/Info.plist"
    queue_compile response-ui -O \
        "${response_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/ResponseUICombined.swift" -o "${response_app}/Contents/MacOS/response-check"
}

queue_preferences() {
    combine DesktopPreferencesCombined.swift "${app_sources}/StudioModel.swift" tests/desktop-preferences-check.swift
    queue_compile preferences -warnings-as-errors \
        "${idle_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/DesktopPreferencesCombined.swift" -o "${test_dir}/preferences-check"
    combine CompletionNotificationsCombined.swift "${app_sources}/StudioModel.swift" tests/completion-notifications-check.swift
    queue_compile notifications -warnings-as-errors \
        "${idle_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/CompletionNotificationsCombined.swift" -o "${test_dir}/notification-check"
    queue_compile logo-appearance "${sources[@]}" "${math_objects[@]}" \
        tests/logo-appearance-check.swift -o "${test_dir}/logo-appearance-check"
    queue_compile update-release-notes "${sources[@]}" "${math_objects[@]}" \
        tests/update-release-notes-check.swift -o "${test_dir}/update-release-notes-check"
}

# Fixture name, production file it replaces, test file, extra swiftc flags.
queue_studio_model_check() {
    local name=$1 combined=$2 test=$3
    shift 3
    combine "${combined}" "${app_sources}/StudioModel.swift" "${test}"
    queue_compile "${name}" "$@" "${idle_sources[@]}" "${math_objects[@]}" \
        "${test_dir}/${combined}" -o "${test_dir}/${name}-check"
}

queue_app_check() {
    local name=$1 test=$2
    shift 2
    queue_compile "${name}" "$@" "${sources[@]}" "${math_objects[@]}" "${test}" -o "${test_dir}/${name}-check"
}

queue_rest() {
    combine StudioHardeningCombined.swift "${app_sources}/UpdateManager.swift" tests/studio-hardening-check.swift
    queue_compile hardening "${(@)sources:#*/UpdateManager.swift}" "${math_objects[@]}" \
        "${test_dir}/StudioHardeningCombined.swift" -o "${test_dir}/check"
    queue_compile agent-sandbox -O "${app_sources}/AgentSandbox.swift" "${app_sources}/AgentJavaScript.swift" "${app_sources}/AgentNetwork.swift" \
        tests/agent-sandbox-check.swift -o "${test_dir}/agent-sandbox-check"
    queue_studio_model_check studio-modes StudioModesCombined.swift tests/studio-modes-check.swift
    queue_app_check model-storage tests/model-storage-check.swift
    queue_app_check decision tests/decision-studio-check.swift
    queue_studio_model_check idle-unload ModelIdleUnloadCombined.swift tests/model-idle-unload-check.swift -warnings-as-errors
    queue_studio_model_check unload-control ModelUnloadControlCombined.swift tests/model-unload-control-check.swift -warnings-as-errors
    queue_studio_model_check mtp-autotune MTPAutoTuneCombined.swift tests/mtp-autotune-check.swift -warnings-as-errors
}

queue_after_preferences() {
    queue_studio_model_check sampling SpeculativeSamplingCombined.swift tests/speculative-sampling-check.swift -warnings-as-errors
    queue_studio_model_check experience DesktopExperienceCombined.swift tests/desktop-experience-check.swift tests/model-storage-ui-check.swift -warnings-as-errors
    queue_studio_model_check text-size ResponseTextSizeCombined.swift tests/response-text-size-check.swift -warnings-as-errors
    queue_studio_model_check mtp-layout MTPPanelLayoutCombined.swift tests/mtp-panel-layout-check.swift -warnings-as-errors
    queue_studio_model_check runtime-contract RuntimeContractCombined.swift tests/issue163-runtime-contract-check.swift -warnings-as-errors
    queue_app_check updater tests/updater-check.swift
    local -a updater_sources
    updater_sources=("${app_sources}/EngineRuntimeStore.swift" "${app_sources}/Localization.swift"
        "${app_sources}/ModelDownloadProgress.swift")
    combine UpdaterApplicationCombined.swift "${app_sources}/UpdateManager.swift" tests/updater-application-check.swift
    queue_compile updater-application -warnings-as-errors "${updater_sources[@]}" \
        "${test_dir}/UpdaterApplicationCombined.swift" -o "${test_dir}/updater-application-check"
    combine UpdaterProgressCombined.swift "${app_sources}/UpdateManager.swift" tests/updater-progress-check.swift
    queue_compile updater-progress "${updater_sources[@]}" \
        "${test_dir}/UpdaterProgressCombined.swift" -o "${test_dir}/updater-progress-check"
    queue_app_check html-preview tests/html-preview-check.swift
    queue_app_check model-card tests/model-card-check.swift
    queue_app_check file-import tests/file-import-check.swift
    queue_app_check image-input tests/image-input-check.swift
    queue_app_check workspace-features tests/workspace-features-check.swift
    combine RecommendedModelsCombined.swift "${app_sources}/ModelManager.swift" tests/recommended-models-ui-check.swift
    queue_compile recommendations "${(@)sources:#*/ModelManager.swift}" "${math_objects[@]}" \
        "${test_dir}/RecommendedModelsCombined.swift" -o "${test_dir}/recommendations-check"
    queue_app_check responsiveness tests/responsiveness-check.swift
    combine BridgeDispatchCombined.swift "${app_sources}/MLXL3Bridge.swift" tests/bridge-dispatch-check.swift
    queue_compile bridge -O "${(@)sources:#*/MLXL3Bridge.swift}" "${math_objects[@]}" \
        "${test_dir}/BridgeDispatchCombined.swift" -o "${test_dir}/bridge-check"
    queue_compile streaming -O "${sources[@]}" "${math_objects[@]}" tests/streaming-render-check.swift \
        -o "${test_dir}/streaming-check"
}

queue_tail() {
    # StudioAurora.swift resolves its colours and style through Theme.swift and the app's
    # environment, so the shader check builds with every app source (no MLXL3StudioApp.swift).
    queue_compile aurora-shader -O "${sources[@]}" "${math_objects[@]}" \
        tests/aurora-shader-check.swift -o "${test_dir}/aurora-shader-check"
    queue_compile cli-command -O "${app_sources}/CLICommand.swift" \
        tests/cli-command-check.swift -o "${test_dir}/cli-command-check"
    queue_compile download-progress -O "${app_sources}/ModelDownloadProgress.swift" \
        tests/download-progress-check.swift -o "${test_dir}/download-progress-check"
    queue_compile mtp-tuning -O "${app_sources}/MTPTuning.swift" \
        tests/mtp-tuning-check.swift -o "${test_dir}/mtp-tuning-check"
    queue_compile mcp-configuration -O "${app_sources}/MCPConfiguration.swift" \
        "${app_sources}/Localization.swift" tests/mcp-configuration-check.swift \
        -o "${test_dir}/mcp-configuration-check"
    # Binaries of check-studios.sh and check-studio-workspace.sh (same flags as there).
    local -a studio_args
    studio_args=("${swiftc_args[@]}")
    swiftc_args=(-swift-version 6 -warnings-as-errors -parse-as-library -module-cache-path "${test_dir}/modules")
    [[ -n "${MLXL3_MACOS_SDK:-}" ]] && swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
    swiftc_args+=(-I "${binary_dir}" -I "${binary_dir}/Modules")
    queue_compile studios "${sources[@]}" "${math_objects[@]}" tests/studios-check.swift -o "${test_dir}/studios-check"
    queue_compile studios-agent "${app_sources}/AgentSandbox.swift" "${app_sources}/AgentJavaScript.swift" "${app_sources}/AgentNetwork.swift" tests/agent-sandbox-check.swift \
        -o "${test_dir}/studios-agent-check"
    swiftc_args=(-swift-version 6 -warnings-as-errors -parse-as-library)
    [[ -n "${MLXL3_MACOS_SDK:-}" ]] && swiftc_args+=(-sdk "${MLXL3_MACOS_SDK}")
    queue_compile studio-workspace -I "${binary_dir}" -I "${binary_dir}/Modules" "${sources[@]}" \
        "${math_objects[@]}" tests/studio-workspace-check.swift -o "${test_dir}/studio-workspace-check"
    swiftc_args=("${studio_args[@]}")
}

# ----- Checks, in the original order -----

run_early() {
    await_compile context-memory
    "${test_dir}/context-memory-check"
    await_compile voice-callback
    run_fixture_check "${test_dir}/voice-callback-check"
}

# Both CI paths compile and run the same registered response app and fixtures.
run_response_ui() {
    await_compile response-ui
    print "Response UI Swift 6 compilation passed"
    run_fixture_check "${response_app}/Contents/MacOS/response-check" "$PWD/tests/fake-desktop-engine.py"
    # A fresh process runs the fixed-size native bitmap/OCR checks.
    run_fixture_check "${response_app}/Contents/MacOS/response-check" "$PWD/tests/fake-desktop-engine.py" --check-long-visibility
    # Long replies streamed below long histories must keep the view pinned (own time budget).
    run_fixture_check "${response_app}/Contents/MacOS/response-check" "$PWD/tests/fake-desktop-engine.py" --check-scroll-follow
}

# The focused PR job and the full suite execute the same production fixtures.
run_preferences() {
    local logo_app
    # Appearance persistence, startup gate and actual subprocess lifecycle.
    await_compile preferences
    run_fixture_check "${test_dir}/preferences-check" "$PWD/tests/fake-desktop-engine.py"
    await_compile notifications
    run_fixture_check "${test_dir}/notification-check" "$PWD/tests/fake-desktop-engine.py"
    await_compile logo-appearance
    run_fixture_check "${test_dir}/logo-appearance-check"
    logo_app="${test_dir}/Logo Check.app"
    mkdir -p "${logo_app}/Contents/MacOS" "${logo_app}/Contents/Resources"
    cp "${test_dir}/logo-appearance-check" "${logo_app}/Contents/MacOS/MLXL3Studio"
    cp apps/MLXL3Studio/Resources/Info.plist "${logo_app}/Contents/Info.plist"
    cp apps/MLXL3Studio/Resources/AppIcon.png "${logo_app}/Contents/Resources/AppIcon.png"
    scripts/build-app-icon.sh apps/MLXL3Studio/Resources/AppDockIcon.png "${logo_app}/Contents/Resources/MLXL3DockV2.icns"
    run_fixture_check /usr/bin/env MLXL3_TEST_PACKAGED_ICON=1 "${logo_app}/Contents/MacOS/MLXL3Studio"
    await_compile update-release-notes
    run_fixture_check "${test_dir}/update-release-notes-check"
}

run_rest() {
    await_compile hardening
    "${test_dir}/check" "$PWD/tests/fake-desktop-engine.py"
    await_compile agent-sandbox
    "${test_dir}/agent-sandbox-check"
    "${test_python}" tests/check-agent-protocol.py "${test_dir}/agent-sandbox-check"
    await_compile studio-modes
    fixture_timeout=180 run_fixture_check "${test_dir}/studio-modes-check" "$PWD/tests/fake-mode-engine.py" | tee "${test_dir}/studio-modes.log"
    grep -q "^Studio modes checks passed:" "${test_dir}/studio-modes.log"
    await_compile model-storage
    run_fixture_check "${test_dir}/model-storage-check"
    # Decision mode: drafts, discovery, decide-bridge lifecycle and decide-serve launch on a fake engine.
    await_compile decision
    "${test_dir}/decision-check" "$PWD/tests/fake-decision-engine.py"
    # Check the production idle deadline and lifecycle without waiting minutes or
    # loading a model. File concatenation gives the fixture private event access.
    await_compile idle-unload
    "${test_dir}/idle-unload-check" "$PWD/tests/fake-desktop-engine.py"
    # Toolbar unload beside the model selector: state table, hosted capsule and the
    # real bridge lifecycle (loading, generating, tuning, unload and reload).
    await_compile unload-control
    run_fixture_check "${test_dir}/unload-control-check" "$PWD/tests/fake-desktop-engine.py"
    # Automatic MTP Tune through the real bridge/CLI/fixture engine (MTP-AUTOTUNE-20261115).
    await_compile mtp-autotune
    run_fixture_check "${test_dir}/mtp-autotune-check" "$PWD/tests/fake-desktop-engine.py"
}

run_after_preferences() {
    # Keep global speculative preferences independent of current-model sampling.
    await_compile sampling
    "${test_dir}/sampling-check" "$PWD/tests/fake-desktop-engine.py" \
        "$PWD/apps/MLXL3Studio/Sources/MLXL3Studio/GenerationInspector.swift"
    await_compile experience
    run_fixture_check "${test_dir}/experience-check" "$PWD/tests/fake-desktop-engine.py" | tee "${test_dir}/experience.log"
    grep -q "^Desktop experience checks passed:" "${test_dir}/experience.log"
    await_compile text-size
    run_fixture_check "${test_dir}/text-size-check"
    await_compile mtp-layout
    run_fixture_check "${test_dir}/mtp-layout-check"
    await_compile runtime-contract
    run_fixture_check "${test_dir}/runtime-contract-check"

    "${test_python}" tests/update-archive-fixtures.py "${test_dir}/archives"
    await_compile updater
    "${test_dir}/updater-check" "${test_dir}/archives" ${MLXL3_TEST_ENGINE_ARCHIVE:+"${MLXL3_TEST_ENGINE_ARCHIVE}"} ${MLXL3_TEST_ENGINE_VERSION:+"${MLXL3_TEST_ENGINE_VERSION}"}
    await_compile updater-application
    "${test_dir}/updater-application-check" "${test_dir}/archives" ${MLXL3_TEST_APP_DMG:+"${MLXL3_TEST_APP_DMG}"} ${MLXL3_TEST_APP_VERSION:+"${MLXL3_TEST_APP_VERSION}"}
    await_compile updater-progress
    run_fixture_check "${test_dir}/updater-progress-check" "${test_python}" "$PWD/tests/desktop-transfer-fixture.py"
    await_compile html-preview
    run_fixture_check "${test_dir}/html-preview-check" "${test_python}" "$PWD/tests/desktop-transfer-fixture.py"
    await_compile model-card
    run_fixture_check "${test_dir}/model-card-check" "${test_python}" "$PWD/tests/desktop-transfer-fixture.py"
    run_fixture_check "${test_dir}/model-card-check" "${test_python}" "$PWD/tests/slow-model-card-fixture.py"
    await_compile file-import
    "${test_dir}/file-import-check" "$PWD/tests/fake-desktop-engine.py"
    await_compile image-input
    run_fixture_check "${test_dir}/image-input-check" "$PWD/tests/fake-desktop-engine.py" "$PWD/tests/fixtures/image-input"
    await_compile workspace-features
    "${test_dir}/workspace-features-check" "$PWD/tests/fake-desktop-engine.py"
    await_compile recommendations
    run_fixture_check "${test_dir}/recommendations-check"
    await_compile responsiveness
    "${test_dir}/responsiveness-check"

    "${binary_dir}/MLXL3Studio" --check-chat-timeline
    "${binary_dir}/MLXL3Studio" --check-mcp-preferences

    # Exercise the real Swift bridge against a subprocess: callback ordering,
    # stale-generation suppression, reload, cancellation and crash handling.
    await_compile bridge
    "${test_dir}/bridge-check" "$PWD/tests/fake-desktop-engine.py"
    # Exercise incremental Markdown/code rendering and Unicode chunk boundaries
    # through AppKit/SwiftUI, not only the pure Python parser tests.
    await_compile streaming
    "${test_dir}/streaming-check"
}

run_tail() {
    # The animated background must keep the v2.0.2 look pixel for pixel: render the
    # real AuroraScene off screen with the frozen shader and with the shipped one.
    await_compile aurora-shader
    "${test_dir}/aurora-shader-check" tests/fixtures/aurora-shader-v2.0.2.fsh
    # Exercise the cancellable CLICommand transport independently of the model
    # bridge (large pipe output, line progress, cancellation and stderr errors).
    await_compile cli-command
    "${test_dir}/cli-command-check" "${test_python}"
    await_compile download-progress
    "${test_dir}/download-progress-check"
    await_compile mtp-tuning
    "${test_dir}/mtp-tuning-check"
    await_compile mcp-configuration
    "${test_dir}/mcp-configuration-check"

    # Beta 5 project/artifact integration shares the already-built app sources.
    await_compile studios
    await_compile studios-agent
    MLXL3_STUDIOS_TMP="${test_dir}" MLXL3_STUDIOS_CHECK_BIN="${test_dir}/studios-check" \
        MLXL3_STUDIOS_AGENT_BIN="${test_dir}/studios-agent-check" zsh scripts/check-studios.sh

    # CPU-only full-window Studio navigation, editing and reviewed snapshot safety.
    await_compile studio-workspace
    MLXL3_STUDIO_WORKSPACE_BIN="${test_dir}/studio-workspace-check" zsh scripts/check-studio-workspace.sh
}

if [[ "${mode}" != skip-preferences ]]; then
    queue_early
fi
[[ -d "${binary_dir}/Modules" ]] && swiftc_args+=(-I "${binary_dir}/Modules")
if [[ -f "${binary_dir}/SwiftMath.o" ]]; then
    math_objects=("${binary_dir}/SwiftMath.o")
else
    math_objects=("${binary_dir}/SwiftMath.build"/*.swift.o)
fi
idle_sources=("${(@)sources:#*/StudioModel.swift}")

case "${mode}" in
    preferences)
        queue_preferences
        queue_response_ui
        ;;
    full)
        queue_rest
        queue_preferences
        queue_after_preferences
        queue_response_ui
        queue_tail
        ;;
    skip-preferences)
        queue_rest
        queue_after_preferences
        queue_tail
        ;;
esac
start_compiles
if [[ "${mode}" != skip-preferences ]]; then
    zsh tests/app-icon-check.zsh
    run_early
fi
if [[ "${mode}" == preferences ]]; then
    run_preferences
    run_response_ui
    print "Desktop preferences, notifications, logo and response UI checks passed in ${SECONDS} s"
    exit 0
fi
run_rest
[[ "${mode}" == full ]] && run_preferences
run_after_preferences
[[ "${mode}" == full ]] && run_response_ui
run_tail
if [[ "${mode}" == full ]]; then
    print "Desktop E2E checks passed in ${SECONDS} s: lifecycle, file imports, bridge, streaming/rendering, CLI transport and Studios"
else
    print "Desktop E2E checks passed in ${SECONDS} s without the units of --preferences-only (icon, context memory, voice, preferences, response UI)"
fi
