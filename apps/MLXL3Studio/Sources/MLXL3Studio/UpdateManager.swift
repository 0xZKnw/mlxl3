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
        let version = rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "vV"))
            .split(separator: "-", maxSplits: 1)
            .first
            .map(String.init) ?? ""
        let pieces = version.split(separator: ".", omittingEmptySubsequences: false)
        guard !pieces.isEmpty else { return nil }
        var parsed: [Int] = []
        for piece in pieces {
            guard !piece.isEmpty, let number = Int(piece), number >= 0 else { return nil }
            parsed.append(number)
        }
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
    static let repository = "0xZKnw/mlxl3"

    @Published private(set) var state: AppUpdateState = .idle
    @Published private(set) var latestRelease: AppUpdateRelease?

    let currentVersion: String
    private let releasesURL: URL
    private var updateTask: Task<Void, Never>?
    private var didRunAutomaticCheck = false

    init(
        currentVersion: String = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0.0.0",
        releasesURL: URL = URL(
            string: "https://api.github.com/repos/0xZKnw/mlxl3/releases/latest"
        )!
    ) {
        self.currentVersion = currentVersion
        self.releasesURL = releasesURL
    }

    deinit {
        updateTask?.cancel()
    }

    var hasAvailableUpdate: Bool {
        switch state {
        case .downloading, .ready, .installing:
            return true
        default:
            guard let latestRelease,
                  let current = SemanticVersion(currentVersion),
                  let latest = SemanticVersion(latestRelease.version)
            else { return false }
            return latest > current
        }
    }

    func startAutomaticCheck() {
        guard !didRunAutomaticCheck else { return }
        didRunAutomaticCheck = true
        checkForUpdates()
    }

    func checkForUpdates() {
        guard !state.isBusy else { return }
        updateTask?.cancel()
        state = .checking
        updateTask = Task { [weak self] in
            guard let self else { return }
            do {
                let release = try await Self.fetchLatestRelease(
                    from: releasesURL,
                    currentVersion: currentVersion
                )
                try Task.checkCancellation()
                latestRelease = release

                guard let current = SemanticVersion(currentVersion),
                      let latest = SemanticVersion(release.version)
                else {
                    throw UpdateError.invalidVersion(release.version)
                }
                guard latest > current else {
                    state = .upToDate(checkedAt: Date())
                    return
                }

                state = .downloading(release)
                let diskImage = try await Self.downloadAndVerify(release: release)
                try Task.checkCancellation()
                state = .ready(release: release, diskImage: diskImage)
            } catch is CancellationError {
                state = .idle
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func beginInstallation() -> Bool {
        guard case let .ready(release, diskImage) = state else { return false }
        state = .installing(release)
        let currentApp = Bundle.main.bundleURL
        let currentVersion = currentVersion
        do {
            try Self.stageInstaller(
                diskImage: diskImage,
                currentApp: currentApp,
                currentVersion: currentVersion
            )
            return true
        } catch {
            state = .failed(error.localizedDescription)
            return false
        }
    }

    func openReleasePage() {
        guard let pageURL = latestRelease?.pageURL else { return }
        NSWorkspace.shared.open(pageURL)
    }

    private static func fetchLatestRelease(
        from url: URL,
        currentVersion: String
    ) async throws -> AppUpdateRelease {
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        request.setValue("MLXL3-Desktop/\(currentVersion)", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw UpdateError.releaseLookupFailed
        }
        let payload = try JSONDecoder().decode(GitHubRelease.self, from: data)
        guard !payload.draft,
              !payload.prerelease,
              let pageURL = URL(string: payload.htmlURL),
              let releaseVersion = SemanticVersion(payload.tagName)
        else { throw UpdateError.invalidRelease }

        let candidates = payload.assets.compactMap { asset -> (GitHubAsset, Int)? in
            guard asset.name.lowercased().hasSuffix(".dmg"),
                  let url = URL(string: asset.browserDownloadURL)
            else { return nil }
            var score = 0
            let lower = asset.name.lowercased()
            if lower.contains("mlxl3-desktop") { score += 4 }
            if lower.contains("apple-silicon") || lower.contains("arm64") { score += 2 }
            if url.host == "github.com" || url.host?.hasSuffix("githubusercontent.com") == true {
                score += 1
            }
            return (asset, score)
        }
        guard let selected = candidates.max(by: { $0.1 < $1.1 })?.0,
              let downloadURL = URL(string: selected.browserDownloadURL)
        else { throw UpdateError.missingDiskImage }

        let cleanVersion = payload.tagName.trimmingCharacters(
            in: CharacterSet(charactersIn: "vV")
        )
        _ = releaseVersion
        return AppUpdateRelease(
            version: cleanVersion,
            tag: payload.tagName,
            title: payload.name?.isEmpty == false
                ? payload.name!
                : "MLXL3 Desktop \(cleanVersion)",
            notes: payload.body ?? "",
            pageURL: pageURL,
            asset: AppUpdateAsset(
                name: selected.name,
                downloadURL: downloadURL,
                size: selected.size,
                digest: selected.digest
            )
        )
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

    private static func verifyAsset(at url: URL, asset: AppUpdateAsset) async throws -> Bool {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        if asset.size > 0, Int64(values.fileSize ?? -1) != asset.size { return false }
        guard let digest = asset.digest, digest.lowercased().hasPrefix("sha256:") else {
            throw UpdateError.missingDigest
        }
        let expected = String(digest.dropFirst("sha256:".count)).lowercased()
        let actual = try await Task.detached(priority: .utility) {
            try sha256(at: url)
        }.value
        return actual == expected
    }

    nonisolated private static func sha256(at url: URL) throws -> String {
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
        currentVersion: String
    ) throws {
        guard currentApp.pathExtension == "app" else { throw UpdateError.notRunningFromApp }
        let parent = currentApp.deletingLastPathComponent()
        guard FileManager.default.isWritableFile(atPath: parent.path),
              try currentApp.resourceValues(forKeys: [.volumeIsReadOnlyKey]).volumeIsReadOnly != true
        else { throw UpdateError.applicationNotWritable }

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
        try validateApplication(sourceApp, newerThan: currentVersion)

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
        newerThan currentVersion: String
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
              incoming > current
        else { throw UpdateError.invalidApplication }
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

    nonisolated private static func runProcess(
        _ executable: String,
        arguments: [String]
    ) throws -> Data {
        let process = Process()
        let output = Pipe()
        let errors = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let errorData = errors.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0 else {
            let message = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw UpdateError.commandFailed(message ?? executable)
        }
        return data
    }

    nonisolated private static let installerScript = #"""
#!/bin/zsh
set -u
parent_pid="$1"
source_app="$2"
destination_app="$3"
mount_path="$4"
log_path="$5"
backup_app="${destination_app}.mlxl3-backup"

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

/bin/rm -rf "$backup_app"
if [[ -e "$destination_app" ]]; then
    echo "moving current application to backup"
    /bin/mv "$destination_app" "$backup_app" || exit 20
fi

if /usr/bin/ditto "$source_app" "$destination_app"; then
    echo "new application copied"
    /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$destination_app/Contents/Info.plist" || true
    /bin/rm -rf "$backup_app"
    /usr/bin/hdiutil detach "$mount_path" -quiet >/dev/null 2>&1 || true
    /usr/bin/open -n "$destination_app"
else
    echo "copy failed; restoring backup"
    /bin/rm -rf "$destination_app"
    if [[ -e "$backup_app" ]]; then
        /bin/mv "$backup_app" "$destination_app"
    fi
    /usr/bin/hdiutil detach "$mount_path" -quiet >/dev/null 2>&1 || true
    exit 21
fi

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

private enum UpdateError: LocalizedError {
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
