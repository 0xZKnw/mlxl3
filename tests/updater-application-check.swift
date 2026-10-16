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
        if CommandLine.arguments.count == 4 {
            try UpdateManager.checkDiskImagePreflight(
                URL(fileURLWithPath: CommandLine.arguments[2]), version: CommandLine.arguments[3])
            print("Real DMG preflight passed: signature, versions, compatibility, runtime hashes/list; detached")
        }
        print("App updater preflight passed: \(checks) signed independent-version, compatibility and integrity cases")
    }
}
