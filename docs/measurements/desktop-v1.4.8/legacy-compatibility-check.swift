
extension UpdateManager {
    nonisolated static func verifyLegacyApp(_ app: URL, version: String) throws {
        let release = AppUpdateRelease(version: version, tag: "v" + version,
            title: "Legacy compatibility", notes: "", pageURL: app,
            asset: AppUpdateAsset(name: "MLXL3-Desktop-v\(version)-b29-Apple-Silicon.dmg",
                downloadURL: app, size: 1, digest: nil))
        try validateApplication(app, newerThan: "1.3.0", currentBuild: 22, release: release)
    }
}
@main struct LegacyAppCheck {
    static func main() throws {
        precondition(CommandLine.arguments.count == 4)
        let app = URL(fileURLWithPath: CommandLine.arguments[1])
        let version = CommandLine.arguments[2]
        let accept = CommandLine.arguments[3] == "accept"
        do {
            try UpdateManager.verifyLegacyApp(app, version: version)
            precondition(accept, "Legacy preflight accepted the mismatched bundle")
            print("Legacy v1.3.0 preflight accepted app/runtime " + version)
        } catch EngineError.incompatible {
            precondition(!accept, "Legacy preflight rejected the aligned final bundle")
            print("Legacy v1.3.0 preflight rejected the mismatched app/runtime as expected")
        }
    }
}
