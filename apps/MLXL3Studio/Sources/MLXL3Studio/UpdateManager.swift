import AppKit
import CryptoKit
import Darwin
import Foundation

struct AppUpdateRelease: Sendable, Equatable {
    let version: String
    let tag: String
    let title: String
    let notes: String
    let pageURL: URL
    let asset: AppUpdateAsset
    var build: Int {
        guard let range = asset.name.range(of: "-b[0-9]+-", options: .regularExpression) else { return 0 }
        return Int(asset.name[range].dropFirst(2).dropLast()) ?? 0
    }
}

struct AppUpdateAsset: Sendable, Equatable {
    let name: String
    let downloadURL: URL
    let size: Int64
    let digest: String?
}

enum AppUpdateState: Equatable {
    case idle
    case checking
    case upToDate(checkedAt: Date)
    case downloading(AppUpdateRelease)
    case ready(release: AppUpdateRelease, diskImage: URL)
    case installing(AppUpdateRelease)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .checking, .downloading, .installing: true
        default: false
        }
    }

    var readyRelease: AppUpdateRelease? {
        guard case let .ready(release, _) = self else { return nil }
        return release
    }
}

struct SemanticVersion: Comparable, Sendable {
    let components: [Int]

    init?(_ rawValue: String) {
        var version = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if version.hasPrefix("v") || version.hasPrefix("V") { version.removeFirst() }
        let pieces = version.split(separator: ".", omittingEmptySubsequences: false)
        guard (1...3).contains(pieces.count) else { return nil }
        var parsed: [Int] = []
        for piece in pieces {
            guard !piece.isEmpty, piece.utf8.allSatisfy({ (48...57).contains($0) }),
                  piece.count == 1 || !piece.hasPrefix("0"),
                  let number = Int(piece), number >= 0 else { return nil }
            parsed.append(number)
        }
        while parsed.count > 1 && parsed.last == 0 { parsed.removeLast() }
        components = parsed
    }

    static func < (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return left < right }
        }
        return false
    }
}

@MainActor
final class UpdateManager: ObservableObject {
    nonisolated static let repository = "0xZKnw/mlxl3"

    @Published private(set) var state: AppUpdateState = .idle
    @Published private(set) var latestRelease: AppUpdateRelease?
    @Published private(set) var engineState: AppUpdateState = .idle
    @Published private(set) var engineRelease: AppUpdateRelease?
    @Published private(set) var currentEngineVersion: String

    let currentVersion: String
    let currentBuild: Int
    private let releasesURL: URL
    private var updateTask: Task<Void, Never>?
    private var didRunAutomaticCheck = false

    init(
        currentVersion: String = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0.0.0",
        releasesURL: URL = URL(
            string: "https://api.github.com/repos/0xZKnw/mlxl3/releases?per_page=100"
        )!
    ) {
        self.currentVersion = currentVersion
        self.currentBuild = Int(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0") ?? 0
        self.releasesURL = releasesURL
        self.currentEngineVersion = EngineRuntimeStore.installedVersion()
    }

    deinit {
        updateTask?.cancel()
    }

    var hasAppUpdate: Bool {
        switch state {
        case .downloading, .ready, .installing:
            return true
        default:
            guard let latestRelease,
                  let current = SemanticVersion(currentVersion),
                  let latest = SemanticVersion(latestRelease.version)
            else { return false }
            return latest > current || (latest == current && latestRelease.build > currentBuild)
        }
    }

    var hasAvailableUpdate: Bool { hasAppUpdate || hasEngineUpdate }
    var hasEngineUpdate: Bool {
        switch engineState {
        case .downloading, .ready, .installing: true
        default:
            if let release = engineRelease, let current = SemanticVersion(currentEngineVersion),
               let latest = SemanticVersion(release.version) { latest > current } else { false }
        }
    }
    var hasReadyUpdate: Bool { state.readyRelease != nil || engineState.readyRelease != nil }
    var isBusy: Bool { state.isBusy || engineState.isBusy }

    func startAutomaticCheck() {
        guard !didRunAutomaticCheck else { return }
        didRunAutomaticCheck = true
        checkForUpdates()
    }

    func checkForUpdates() {
        guard !isBusy else { return }
        updateTask?.cancel()
        state = .checking
        engineState = .checking
        updateTask = Task { [weak self] in
            guard let self else { return }
            do {
                let data = try await Self.fetchReleases(from: releasesURL, currentVersion: currentVersion)
                try Task.checkCancellation()
                let release = try Self.selectRelease(data, channel: .app)
                latestRelease = release
                if let release, let current = SemanticVersion(currentVersion),
                   let latest = SemanticVersion(release.version),
                   latest > current || (latest == current && release.build > currentBuild) {
                    state = .downloading(release)
                    do {
                        let diskImage = try await Self.downloadAndVerify(release: release)
                        try Task.checkCancellation()
                        state = .ready(release: release, diskImage: diskImage)
                    } catch { state = .failed(error.localizedDescription) }
                } else {
                    state = .upToDate(checkedAt: Date())
                }
                let engine = try Self.selectRelease(data, channel: .engine)
                engineRelease = engine
                if let engine, let current = SemanticVersion(currentEngineVersion),
                   let latest = SemanticVersion(engine.version), latest > current {
                    engineState = .downloading(engine)
                    do {
                        let archive = try await Self.downloadAndVerify(release: engine)
                        try Task.checkCancellation()
                        engineState = .ready(release: engine, diskImage: archive)
                    } catch { engineState = .failed(error.localizedDescription) }
                } else { engineState = .upToDate(checkedAt: Date()) }
                try Task.checkCancellation()
            } catch is CancellationError {
                state = .idle
                engineState = .idle
            } catch {
                if state == .checking { state = .failed(error.localizedDescription) }
                engineState = .failed(error.localizedDescription)
            }
        }
    }

    enum Installation { case restartApp, reloadEngine }

    func beginInstallation() async -> Installation? {
        guard !isBusy, hasReadyUpdate else { return nil }
        var engineInstalled = false
        if case let .ready(release, archive) = engineState {
            engineState = .installing(release)
            let version = state.readyRelease?.version ?? currentVersion
            do {
                try await Task.detached(priority: .userInitiated) {
                    try EngineRuntimeStore.install(archive: archive, release: release, appVersion: version)
                }.value
                currentEngineVersion = release.version
                engineState = .upToDate(checkedAt: Date())
                engineInstalled = true
            } catch {
                engineState = .failed(error.localizedDescription)
            }
        }
        guard case let .ready(release, diskImage) = state else {
            return engineInstalled ? .reloadEngine : nil
        }
        state = .installing(release)
        let currentApp = Bundle.main.bundleURL
        let currentVersion = currentVersion
        let currentBuild = currentBuild
        do {
            try await Task.detached(priority: .userInitiated) {
                try Self.stageInstaller(diskImage: diskImage, currentApp: currentApp,
                                        currentVersion: currentVersion, currentBuild: currentBuild, release: release)
            }.value
            return .restartApp
        } catch {
            state = .failed(error.localizedDescription)
            return engineInstalled ? .reloadEngine : nil
        }
    }

    func openReleasePage() {
        guard let pageURL = latestRelease?.pageURL else { return }
        NSWorkspace.shared.open(pageURL)
    }

    private static func fetchReleases(
        from url: URL,
        currentVersion: String
    ) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        request.setValue("MLXL3-Desktop/\(currentVersion)", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw UpdateError.releaseLookupFailed
        }
        guard data.count <= 16 * 1_024 * 1_024 else { throw UpdateError.invalidRelease }
        // Frequent engine releases must never push the latest app out of the
        // first GitHub page. Engine tags are published with latest=false.
        if url.host == "api.github.com", url.path == "/repos/\(repository)/releases" {
            request.url = URL(string: "https://api.github.com/repos/\(repository)/releases/latest")!
            let (latestData, latestResponse) = try await URLSession.shared.data(for: request)
            guard let latestHTTP = latestResponse as? HTTPURLResponse, latestHTTP.statusCode == 200,
                  latestData.count <= 16 * 1_024 * 1_024,
                  var releases = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
                  let latest = try JSONSerialization.jsonObject(with: latestData) as? [String: Any] else {
                throw UpdateError.releaseLookupFailed
            }
            releases.append(latest)
            return try JSONSerialization.data(withJSONObject: releases)
        }
        return data
    }

    enum ReleaseChannel { case app, engine }

    nonisolated static func selectRelease(_ data: Data, channel: ReleaseChannel) throws -> AppUpdateRelease? {
        let payloads: [GitHubRelease]
        if let list = try? JSONDecoder().decode([GitHubRelease].self, from: data) { payloads = list }
        else { payloads = [try JSONDecoder().decode(GitHubRelease.self, from: data)] }
        let prefix = channel == .app ? "v" : "engine-v"
        let releases = payloads.compactMap { payload -> AppUpdateRelease? in
            guard !payload.draft, !payload.prerelease, payload.tagName.hasPrefix(prefix) else { return nil }
            let version = String(payload.tagName.dropFirst(prefix.count))
            guard version.split(separator: ".", omittingEmptySubsequences: false).count == 3,
                  version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 }),
                  SemanticVersion(version) != nil,
                  let page = URL(string: payload.htmlURL),
                  trustedURL(page, path: "/\(repository)/releases/tag/\(payload.tagName)") else { return nil }
            let pattern = channel == .app
                ? "^MLXL3-Desktop-v\(NSRegularExpression.escapedPattern(for: version))-b[0-9]+-Apple-Silicon\\.dmg$"
                : "^MLXL3-Engine-v\(NSRegularExpression.escapedPattern(for: version))-arm64\\.tar\\.gz$"
            let assets = payload.assets.compactMap { asset -> AppUpdateAsset? in
                guard asset.name.range(of: pattern, options: .regularExpression) != nil,
                      asset.size > 0, asset.size <= 1_500_000_000,
                      let digest = asset.digest, validDigest(digest),
                      let url = URL(string: asset.browserDownloadURL),
                      trustedURL(url, path: "/\(repository)/releases/download/\(payload.tagName)/\(asset.name)") else { return nil }
                return AppUpdateAsset(name: asset.name, downloadURL: url, size: asset.size, digest: digest)
            }
            guard let asset = assets.max(by: { $0.name.localizedStandardCompare($1.name) == .orderedAscending }) else { return nil }
            return AppUpdateRelease(version: version, tag: payload.tagName, title: payload.name ?? payload.tagName,
                                    notes: payload.body ?? "", pageURL: page, asset: asset)
        }
        return releases.max {
            let left = SemanticVersion($0.version)!, right = SemanticVersion($1.version)!
            return left == right ? $0.build < $1.build : left < right
        }
    }

    nonisolated static func trustedURL(_ url: URL, path: String) -> Bool {
        url.scheme == "https" && url.host == "github.com" && url.port == nil
            && url.user == nil && url.password == nil && url.query == nil && url.fragment == nil && url.path == path
    }

    nonisolated static func validDigest(_ digest: String) -> Bool {
        digest.hasPrefix("sha256:") && digest.count == 71 && digest.dropFirst(7).utf8.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }

    private static func downloadAndVerify(release: AppUpdateRelease) async throws -> URL {
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "io.mlxl3.desktop/Updates", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let destination = cacheRoot.appending(path: release.asset.name)

        if FileManager.default.fileExists(atPath: destination.path),
           try await verifyAsset(at: destination, asset: release.asset) {
            return destination
        }

        let (temporaryURL, response) = try await URLSession.shared.download(
            from: release.asset.downloadURL
        )
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode)
        else { throw UpdateError.downloadFailed }

        let staged = cacheRoot.appending(path: UUID().uuidString + ".download")
        try FileManager.default.moveItem(at: temporaryURL, to: staged)
        do {
            guard try await verifyAsset(at: staged, asset: release.asset) else {
                throw UpdateError.integrityCheckFailed
            }
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: staged, to: destination)
            return destination
        } catch {
            try? FileManager.default.removeItem(at: staged)
            throw error
        }
    }

    static func verifyAsset(at url: URL, asset: AppUpdateAsset) async throws -> Bool {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { return false }
        if asset.size > 0, Int64(values.fileSize ?? -1) != asset.size { return false }
        guard let digest = asset.digest, validDigest(digest) else {
            throw UpdateError.missingDigest
        }
        let expected = String(digest.dropFirst("sha256:".count)).lowercased()
        let actual = try await Task.detached(priority: .utility) {
            try sha256(at: url)
        }.value
        return actual == expected
    }

    nonisolated static func sha256(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try handle.read(upToCount: 4 * 1_024 * 1_024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func stageInstaller(
        diskImage: URL,
        currentApp: URL,
        currentVersion: String,
        currentBuild: Int,
        release: AppUpdateRelease
    ) throws {
        guard currentApp.pathExtension == "app" else { throw UpdateError.notRunningFromApp }
        let parent = currentApp.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: parent.path),
              try currentApp.resourceValues(forKeys: [.volumeIsReadOnlyKey]).volumeIsReadOnly != true
        else { throw UpdateError.applicationNotWritable }

        guard try verifyFile(source: diskImage, asset: release.asset) else { throw UpdateError.integrityCheckFailed }
        let mount = try mountDiskImage(diskImage)
        var shouldDetach = true
        defer {
            if shouldDetach { try? detachDiskImage(mount) }
        }

        let entries = try FileManager.default.contentsOfDirectory(
            at: mount,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        guard let sourceApp = entries.first(where: {
            $0.pathExtension == "app" && $0.lastPathComponent == "MLXL3 Desktop.app"
        }) ?? entries.first(where: { $0.pathExtension == "app" }) else {
            throw UpdateError.invalidDiskImage
        }
        try validateApplication(sourceApp, newerThan: currentVersion, currentBuild: currentBuild, release: release)
        // Validate the bundled Rust/Metal runtime before stopping the working app.
        _ = try runProcess(sourceApp.appending(path: "Contents/Resources/runtime/mlxl3").path,
                           arguments: ["list", "--json"])

        let helperRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "io.mlxl3.desktop/Installers", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: helperRoot, withIntermediateDirectories: true)
        let logRoot = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appending(path: "Logs/MLXL3", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: logRoot, withIntermediateDirectories: true)
        let logURL = logRoot.appending(path: "updater.log")
        let helper = helperRoot.appending(path: "install-\(UUID().uuidString).zsh")
        try installerScript.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: helper.path
        )
        _ = try runProcess("/bin/zsh", arguments: ["-n", helper.path])

        try spawnDetached(
            "/bin/zsh",
            arguments: [
            "/bin/zsh",
            helper.path,
            String(ProcessInfo.processInfo.processIdentifier),
            sourceApp.path,
            currentApp.path,
            mount.path,
            logURL.path,
            release.version,
            String(release.build),
            ]
        )
        shouldDetach = false
    }

    nonisolated private static func spawnDetached(
        _ executable: String,
        arguments: [String]
    ) throws {
        var processID: pid_t = 0
        var argv = arguments.map { strdup($0) }
        argv.append(nil)
        defer {
            for pointer in argv where pointer != nil { free(pointer) }
        }
        let result = executable.withCString { executablePath in
            argv.withUnsafeMutableBufferPointer { buffer in
                posix_spawn(
                    &processID,
                    executablePath,
                    nil,
                    nil,
                    buffer.baseAddress!,
                    environ
                )
            }
        }
        guard result == 0 else {
            throw UpdateError.commandFailed(String(cString: strerror(result)))
        }
    }

    nonisolated private static func validateApplication(
        _ app: URL,
        newerThan currentVersion: String,
        currentBuild: Int,
        release: AppUpdateRelease
    ) throws {
        _ = try runProcess(
            "/usr/bin/codesign",
            arguments: ["--verify", "--deep", "--strict", app.path]
        )
        let infoURL = app.appending(path: "Contents/Info.plist")
        let data = try Data(contentsOf: infoURL)
        guard let info = try PropertyListSerialization.propertyList(
            from: data,
            format: nil
        ) as? [String: Any],
              info["CFBundleIdentifier"] as? String == "io.mlxl3.desktop",
              let version = info["CFBundleShortVersionString"] as? String,
              let current = SemanticVersion(currentVersion),
              let incoming = SemanticVersion(version),
              version == release.version,
              Int(info["CFBundleVersion"] as? String ?? "0") == release.build,
              let minimumOS = info["LSMinimumSystemVersion"] as? String,
              let required = SemanticVersion(minimumOS),
              let installedOS = SemanticVersion(EngineRuntimeStore.osVersion), installedOS >= required,
              incoming > current || (incoming == current && (Int(info["CFBundleVersion"] as? String ?? "0") ?? 0) > currentBuild)
        else { throw UpdateError.invalidApplication }
        _ = try runProcess("/usr/bin/lipo", arguments: ["-verify_arch", "arm64", app.appending(path: "Contents/MacOS/MLXL3Studio").path])
        try EngineRuntimeStore.verify(app.appending(path: "Contents/Resources/runtime"), appVersion: version, expectedVersion: version)
    }

    nonisolated private static func verifyFile(source: URL, asset: AppUpdateAsset) throws -> Bool {
        let values = try source.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              values.fileSize.map(Int64.init) == asset.size,
              let digest = asset.digest, validDigest(digest) else { return false }
        return try sha256(at: source) == digest.dropFirst(7).lowercased()
    }

    nonisolated private static func mountDiskImage(_ diskImage: URL) throws -> URL {
        let data = try runProcess(
            "/usr/bin/hdiutil",
            arguments: ["attach", "-nobrowse", "-readonly", "-plist", diskImage.path]
        )
        guard let plist = try PropertyListSerialization.propertyList(
            from: data,
            format: nil
        ) as? [String: Any],
              let entities = plist["system-entities"] as? [[String: Any]],
              let mountPath = entities.compactMap({ $0["mount-point"] as? String }).first
        else { throw UpdateError.invalidDiskImage }
        return URL(fileURLWithPath: mountPath, isDirectory: true)
    }

    nonisolated private static func detachDiskImage(_ mount: URL) throws {
        _ = try runProcess("/usr/bin/hdiutil", arguments: ["detach", mount.path, "-quiet"])
    }

    nonisolated static func runProcess(
        _ executable: String,
        arguments: [String],
        outputLimit: Int = 256 * 1_024 * 1_024
    ) throws -> Data {
        let process = Process()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-updater-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let outURL = root.appendingPathComponent("stdout")
        let errURL = root.appendingPathComponent("stderr")
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        FileManager.default.createFile(atPath: errURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: outURL)
        let errors = try FileHandle(forWritingTo: errURL)
        defer { try? output.close(); try? errors.close() }
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        // Pass metadata as positional arguments and bound output on disk.
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", "ulimit -f \"$1\" || exit 77; shift; exec \"$@\"", "mlxl3-helper",
                             String(max(1, (outputLimit + 511) / 512)), executable] + arguments
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        if finished.wait(timeout: .now() + 60) == .timedOut {
            process.terminate()
            if finished.wait(timeout: .now() + 2) == .timedOut { Darwin.kill(process.processIdentifier, SIGKILL) }
            throw UpdateError.commandFailed("Update helper timed out")
        }
        guard (try outURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= outputLimit,
              (try errURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= outputLimit else {
            throw UpdateError.commandFailed("Update helper output limit exceeded")
        }
        let data = try Data(contentsOf: outURL)
        let errorData = try Data(contentsOf: errURL)
        guard process.terminationStatus == 0 else {
            let message = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw UpdateError.commandFailed(message ?? executable)
        }
        return data
    }

    nonisolated static let installerScript = #"""
#!/bin/zsh
set -u
parent_pid="$1"
source_app="$2"
destination_app="$3"
mount_path="$4"
log_path="$5"
backup_app="${destination_app}.mlxl3-backup"
staged_app="${destination_app}.mlxl3-stage-${$}"
expected_version="$6"
expected_build="$7"
replacing=0
cleanup() {
    local status="$?"
    if (( replacing )); then
        /bin/rm -rf "$destination_app"
        [[ ! -e "$backup_app" ]] || /bin/mv "$backup_app" "$destination_app"
    fi
    /bin/rm -rf "$staged_app"
    /usr/bin/hdiutil detach "$mount_path" -quiet >/dev/null 2>&1 || true
    return "$status"
}
trap cleanup EXIT
trap 'exit 23' HUP INT TERM

exec >> "$log_path" 2>&1
echo "[$(/bin/date -u +%Y-%m-%dT%H:%M:%SZ)] update start"
echo "source=$source_app"
echo "destination=$destination_app"

parent_is_alive() {
    local state
    state="$(/bin/ps -o stat= -p "$parent_pid" 2>/dev/null | /usr/bin/tr -d ' ')"
    [[ -n "$state" && "$state" != Z* ]]
}

for _ in {1..300}; do
    if ! parent_is_alive; then
        break
    fi
    /bin/sleep 0.1
done

if parent_is_alive; then
    echo "parent process did not terminate"
    /usr/bin/hdiutil detach "$mount_path" -quiet >/dev/null 2>&1 || true
    exit 22
fi

# Copy and verify before moving the working application. A failed copy or
# signature/version check leaves both the installed app and its backup intact.
/usr/bin/ditto "$source_app" "$staged_app" || exit 21
/usr/bin/codesign --verify --deep --strict "$staged_app" || exit 24
/usr/bin/lipo -verify_arch arm64 "$staged_app/Contents/MacOS/MLXL3Studio" || exit 24
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$staged_app/Contents/Info.plist")" == "$expected_version" ]] || exit 24
[[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$staged_app/Contents/Info.plist")" == "$expected_build" ]] || exit 24
"$staged_app/Contents/Resources/runtime/mlxl3" runtime-info || exit 24

if [[ -e "$backup_app" ]]; then
    /bin/mv "$backup_app" "${backup_app}.previous-${$}" || exit 20
fi
if [[ -e "$destination_app" ]]; then
    /bin/mv "$destination_app" "$backup_app" || exit 20
    replacing=1
fi
/bin/mv "$staged_app" "$destination_app" || exit 21
/usr/bin/open -n "$destination_app" || exit 25
replacing=0
echo "new application installed; previous application retained"

/bin/rm -f "$0"
"""#
}

private struct GitHubRelease: Decodable, Sendable {
    let tagName: String
    let name: String?
    let body: String?
    let htmlURL: String
    let draft: Bool
    let prerelease: Bool
    let assets: [GitHubAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case body
        case htmlURL = "html_url"
        case draft
        case prerelease
        case assets
    }
}

private struct GitHubAsset: Decodable, Sendable {
    let name: String
    let browserDownloadURL: String
    let size: Int64
    let digest: String?

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
        case size
        case digest
    }
}

enum UpdateError: LocalizedError {
    case releaseLookupFailed
    case invalidRelease
    case invalidVersion(String)
    case missingDiskImage
    case downloadFailed
    case missingDigest
    case integrityCheckFailed
    case notRunningFromApp
    case applicationNotWritable
    case invalidDiskImage
    case invalidApplication
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .releaseLookupFailed:
            L("GitHub ne répond pas correctement. Réessaie dans quelques instants.", "GitHub is not responding correctly. Try again shortly.")
        case .invalidRelease:
            L("La dernière release GitHub n’est pas valide.", "The latest GitHub release is invalid.")
        case let .invalidVersion(version):
            L("Version de release invalide : \(version).", "Invalid release version: \(version).")
        case .missingDiskImage:
            L("Cette release ne contient pas de DMG Apple Silicon.", "This release has no Apple Silicon DMG.")
        case .downloadFailed:
            L("Le téléchargement de la mise à jour a échoué.", "The update download failed.")
        case .missingDigest:
            L("GitHub n’a pas fourni l’empreinte SHA-256 de la release.", "GitHub did not provide a SHA-256 checksum for the release.")
        case .integrityCheckFailed:
            L("L’empreinte SHA-256 du DMG ne correspond pas à la release GitHub.", "The DMG checksum does not match the GitHub release.")
        case .notRunningFromApp:
            L("L’installation automatique fonctionne depuis MLXL3 Desktop.app.", "Automatic installation requires MLXL3 Desktop.app.")
        case .applicationNotWritable:
            L("MLXL3 Desktop doit être placé dans un dossier modifiable, par exemple Applications.", "MLXL3 Desktop must be in a writable folder, such as Applications.")
        case .invalidDiskImage:
            L("Le DMG téléchargé ne contient pas une application MLXL3 valide.", "The downloaded DMG does not contain a valid MLXL3 app.")
        case .invalidApplication:
            L("La nouvelle application a une identité ou une version invalide.", "The new app has an invalid identity or version.")
        case let .commandFailed(message):
            L("La préparation de la mise à jour a échoué : \(message)", "Update preparation failed: \(message)")
        }
    }
}
