import Foundation

// Compiled with UpdateManager.swift so the real private preflight is exercised.
extension UpdateManager {
    nonisolated static func checkApplicationPreflight(
        _ app: URL, release: AppUpdateRelease, currentVersion: String = "1.4.3",
        currentBuild: Int = 25
    ) throws {
        try validateApplication(
            app, newerThan: currentVersion, currentBuild: currentBuild, release: release)
    }

    nonisolated static func checkDiskImagePreflight(_ image: URL, version: String) throws {
        let mount = try mountDiskImage(image)
        do {
            let app = mount.appendingPathComponent("MLXL3 Desktop.app")
            let release = AppUpdateRelease(
                version: version, tag: "v" + version, title: "DMG preflight", notes: "",
                pageURL: image,
                asset: AppUpdateAsset(name: image.lastPathComponent, downloadURL: image,
                    size: Int64(try image.resourceValues(forKeys: [.fileSizeKey]).fileSize!), digest: nil))
            try checkApplicationPreflight(app, release: release)
            _ = try runProcess(app.appendingPathComponent("Contents/Resources/runtime/mlxl3").path,
                arguments: ["list", "--json"])
        } catch {
            try detachDiskImage(mount)
            throw error
        }
        try detachDiskImage(mount)
    }
}

@main struct UpdaterApplicationCheck {
    static func main() throws {
        // Signed Mach-O doubles exercise the real installer, without MLX or a user app.
        let executable = URL(fileURLWithPath: CommandLine.arguments[0])
        if executable.lastPathComponent == "mlxl3" {
            precondition(CommandLine.arguments.dropFirst() == ["runtime-info"]
                || CommandLine.arguments.dropFirst() == ["list", "--json"])
            print(CommandLine.arguments[1] == "runtime-info"
                ? #"{"bridge_protocol":1,"version":"1.4.5","mlx_enabled":true}"# : "[]")
            return
        }
        if executable.lastPathComponent == "MLXL3Studio" { return }
        precondition(CommandLine.arguments.count == 2 || CommandLine.arguments.count == 4)
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("mlxl3-app-preflight-" + UUID().uuidString)
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? files.removeItem(at: root) }
        let app = root.appendingPathComponent("MLXL3 Desktop.app")
        let runtime = app.appendingPathComponent("Contents/Resources/runtime")
        try files.createDirectory(at: app.appendingPathComponent("Contents/MacOS"), withIntermediateDirectories: true)
        try files.createDirectory(at: runtime.deletingLastPathComponent(), withIntermediateDirectories: true)
        try files.copyItem(at: URL(fileURLWithPath: CommandLine.arguments[0]), to: app.appendingPathComponent("Contents/MacOS/MLXL3Studio"))
        try EngineRuntimeStore.stage(
            archive: URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent("valid.tar.gz"),
            directory: runtime, releaseVersion: "1.2.0", appVersion: "1.4.4")
        let original = try JSONSerialization.jsonObject(
            with: Data(contentsOf: runtime.appendingPathComponent("engine.json"))) as! [String: Any]
        func prepare(appVersion: String = "1.4.4", engineVersion: String = "1.4.5",
                     change: (inout [String: Any]) -> Void = { _ in }) throws -> AppUpdateRelease {
            let info: [String: String] = [
                "CFBundleIdentifier": "io.mlxl3.desktop", "CFBundleExecutable": "MLXL3Studio",
                "CFBundlePackageType": "APPL", "CFBundleShortVersionString": appVersion,
                "CFBundleVersion": "27", "LSMinimumSystemVersion": "26.2"]
            try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
                .write(to: app.appendingPathComponent("Contents/Info.plist"))
            var manifest = original
            manifest["version"] = engineVersion
            manifest["minimum_app_version"] = "1.4.0"
            change(&manifest)
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: runtime.appendingPathComponent("engine.json"))
            _ = try UpdateManager.runProcess("/usr/bin/codesign", arguments: ["--force", "--sign", "-", app.path])
            return AppUpdateRelease(version: appVersion, tag: "v" + appVersion,
                title: "Fixture", notes: "", pageURL: app,
                asset: AppUpdateAsset(name: "MLXL3-Desktop-v\(appVersion)-b27-Apple-Silicon.dmg",
                    downloadURL: app, size: 1, digest: nil))
        }
        var checks = 0
        for (appVersion, engineVersion) in [("1.4.4", "1.4.5"), ("1.4.5", "1.4.5"),
                                            ("1.4.6", "1.4.5"), ("1.4.5", "1.4.6")] {
            try UpdateManager.checkApplicationPreflight(app,
                release: prepare(appVersion: appVersion, engineVersion: engineVersion))
            checks += 1
        }
        let sameVersion = try prepare()
        try UpdateManager.checkApplicationPreflight(app, release: sameVersion,
            currentVersion: "1.4.4", currentBuild: 26)
        checks += 1
        do {
            try UpdateManager.checkApplicationPreflight(app, release: sameVersion,
                currentVersion: "1.4.4", currentBuild: 27)
            preconditionFailure("Accepted an identical app build")
        } catch UpdateError.invalidApplication { checks += 1 }
        let wrongRelease = AppUpdateRelease(version: "1.4.5", tag: "v1.4.5",
            title: "Wrong version", notes: "", pageURL: app, asset: sameVersion.asset)
        do {
            try UpdateManager.checkApplicationPreflight(app, release: wrongRelease)
            preconditionFailure("Accepted app/release mismatch")
        } catch UpdateError.invalidApplication { checks += 1 }
        for key in ["minimum_app_version", "minimum_macos", "architecture", "bridge_protocol", "schema", "version"] {
            let release = try prepare { manifest in
                switch key {
                case "architecture": manifest[key] = "x86_64"
                case "bridge_protocol", "schema": manifest[key] = 2
                case "version": manifest[key] = "../1.4.5"
                case "minimum_app_version": manifest[key] = "1.4.5"
                default: manifest[key] = "999.0.0"
                }
            }
            do {
                try UpdateManager.checkApplicationPreflight(app, release: release)
                preconditionFailure("Accepted incompatible manifest: " + key)
            } catch EngineError.incompatible { checks += 1 }
        }
        let release = try prepare { manifest in
            var records = manifest["files"] as! [String: [String: Any]]
            records["mlxl3"]!["sha256"] = String(repeating: "0", count: 64)
            manifest["files"] = records
        }
        do {
            try UpdateManager.checkApplicationPreflight(app, release: release)
            preconditionFailure("Accepted corrupt engine")
        } catch EngineError.integrity { checks += 1 }
        let signed = try prepare()
        try Data("tampered".utf8).write(to: runtime.appendingPathComponent("mlx.metallib"))
        do {
            try UpdateManager.checkApplicationPreflight(app, release: signed)
            preconditionFailure("Accepted broken app signature")
        } catch UpdateError.commandFailed { checks += 1 }
        precondition(checks == 15)
        try checkInstallerBoundaries(root: root, executable: executable)
        if CommandLine.arguments.count == 4 {
            try UpdateManager.checkDiskImagePreflight(
                URL(fileURLWithPath: CommandLine.arguments[2]), version: CommandLine.arguments[3])
            print("Real DMG preflight passed: signature, versions, compatibility, runtime hashes/list; detached")
        }
        print("App updater preflight passed: \(checks) signed independent-version, compatibility and integrity cases")
    }

    static func checkInstallerBoundaries(root: URL, executable: URL) throws {
        let files = FileManager.default
        let source = root.appendingPathComponent("Fixture.app")
        let runtime = source.appendingPathComponent("Contents/Resources/runtime")
        let macOS = source.appendingPathComponent("Contents/MacOS")
        try files.createDirectory(at: runtime, withIntermediateDirectories: true)
        try files.createDirectory(at: macOS, withIntermediateDirectories: true)
        try files.copyItem(at: executable, to: macOS.appendingPathComponent("MLXL3Studio"))
        var records: [String: [String: Any]] = [:]
        for name in EngineRuntimeStore.runtimeFiles.sorted() {
            let file = runtime.appendingPathComponent(name)
            if name == "mlx.metallib" {
                try Data("test double; no Metal library".utf8).write(to: file)
            } else {
                try files.copyItem(at: executable, to: file)
                _ = try UpdateManager.runProcess("/usr/bin/codesign", arguments: ["--force", "--sign", "-", file.path])
            }
            records[name] = ["size": try file.resourceValues(forKeys: [.fileSizeKey]).fileSize!,
                             "sha256": try UpdateManager.sha256(at: file)]
        }
        let manifest: [String: Any] = ["schema": 1, "bridge_protocol": 1,
            "version": "1.4.5", "architecture": "arm64", "minimum_app_version": "1.4.0",
            "minimum_macos": "26.2.0", "files": records]
        try JSONSerialization.data(withJSONObject: manifest).write(to: runtime.appendingPathComponent("engine.json"))
        let archive = root.appendingPathComponent("signed-fixture.tar.gz")
        _ = try UpdateManager.runProcess("/usr/bin/tar", arguments: ["-czf", archive.path, "-C", runtime.path]
            + EngineRuntimeStore.runtimeFiles.sorted() + ["engine.json"])
        let release = AppUpdateRelease(version: "1.4.5", tag: "engine-v1.4.5", title: "Fixture",
            notes: "", pageURL: archive, asset: AppUpdateAsset(name: archive.lastPathComponent,
                downloadURL: archive, size: Int64(try archive.resourceValues(forKeys: [.fileSizeKey]).fileSize!),
                digest: "sha256:" + (try UpdateManager.sha256(at: archive))))
        let store = root.appendingPathComponent("engine-store")
        try EngineRuntimeStore.install(archive: archive, release: release, appVersion: "1.4.5", root: store)
        precondition(EngineRuntimeStore.resolve(appVersion: "1.4.5", root: store, bundledVersion: "1.4.5") == nil)
        precondition(EngineRuntimeStore.resolve(appVersion: "1.4.5", root: store, bundledVersion: "1.4.4") != nil)

        // A distinct identity and no-op executable prevent launching the real Desktop.
        let info: [String: Any] = ["CFBundleIdentifier": "io.mlxl3.updater-fixture",
            "CFBundleExecutable": "MLXL3Studio", "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.4.5", "CFBundleVersion": "27", "LSUIElement": true]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            .write(to: source.appendingPathComponent("Contents/Info.plist"))
        _ = try UpdateManager.runProcess("/usr/bin/codesign", arguments: ["--force", "--sign", "-", source.path])
        let destination = root.appendingPathComponent("InstalledFixture.app")
        try files.createDirectory(at: destination, withIntermediateDirectories: true)
        try Data("previous app".utf8).write(to: destination.appendingPathComponent("sentinel"))
        let script = root.appendingPathComponent("install-fixture.zsh")
        try UpdateManager.installerScript.write(to: script, atomically: true, encoding: .utf8)
        do {
            _ = try UpdateManager.runProcess("/bin/zsh", arguments: [script.path, "99999999", source.path,
                destination.path, root.path, root.appendingPathComponent("installer.log").path, "1.4.5", "27"])
        } catch {
            print(try String(contentsOf: root.appendingPathComponent("installer.log"), encoding: .utf8))
            throw error
        }
        let installedInfo = try Data(contentsOf: destination.appendingPathComponent("Contents/Info.plist"))
        let sourceInfo = try Data(contentsOf: source.appendingPathComponent("Contents/Info.plist"))
        precondition(installedInfo == sourceInfo)
        let previous = try String(contentsOf: root.appendingPathComponent("InstalledFixture.app.mlxl3-backup/sentinel"),
            encoding: .utf8)
        precondition(previous == "previous app")
        precondition(!files.fileExists(atPath: script.path))
        let remaining = try files.contentsOfDirectory(atPath: root.path)
        precondition(remaining.allSatisfy { !$0.contains(".mlxl3-stage-") })
        print("Installer boundaries passed: signed Mach-O engine activation and disposable app replacement/backup")
    }
}
