import Darwin
import Foundation

@main
struct ModelStorageCheck {
    struct Failure: Error { let message: String }

    @MainActor static func main() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxl3-storage-" + UUID().uuidString)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let managed = root.appendingPathComponent("Models"), hub = root.appendingPathComponent("hub")
        let external = root.appendingPathComponent("import")
        let first = managed.appendingPathComponent("first"), second = managed.appendingPathComponent("second")
        let stage = managed.appendingPathComponent(".downloads/abcdef123456")
        var count = 0
        func check(_ value: Bool, _ message: String) throws {
            guard value else { throw Failure(message: message) }
            count += 1
        }
        func dir(_ url: URL) throws { try fm.createDirectory(at: url, withIntermediateDirectories: true) }
        func file(_ url: URL, _ value: UInt8, count: Int = 32_768) throws {
            try Data(repeating: value, count: count).write(to: url)
        }
        func allocated(_ url: URL) throws -> Int64 {
            var data = stat()
            guard lstat(url.path, &data) == 0 else { throw Failure(message: url.path) }
            return Int64(data.st_blocks) * 512
        }
        func local(_ url: URL, name: String? = nil, id: String? = nil) -> LocalModel {
            LocalModel(name: name ?? url.lastPathComponent, path: url.path,
                       modelType: "qwen", format: "EXL3", bits: 4, sizeBytes: 1, modules: 1,
                       addedAt: "", size: "", installationID: id)
        }
        func install(_ url: URL, id: String) throws -> LocalModel {
            try dir(url)
            for name in ["config.json","quantization_config.json"] {
                try Data("{}".utf8).write(to: url.appendingPathComponent(name))
            }
            let record = ["attempt": id, "request": ["name": url.lastPathComponent, "destination": url.path]] as [String: Any]
            try JSONSerialization.data(withJSONObject: record).write(to: url.appendingPathComponent(".mlxl3-install.json"))
            return local(url, id: id)
        }
        let a = try install(first, id: "install-A"), b = try install(second, id: "install-B")
        try dir(external)
        try dir(stage)
        let ext = local(external)
        let aOnly = first.appendingPathComponent("a.weights"), shared = first.appendingPathComponent("shared.weights")
        let extOnly = external.appendingPathComponent("original.weights")
        let staged = stage.appendingPathComponent("retained.part")
        try file(aOnly, 1)
        try file(shared, 2, count: 65_536)
        try file(extOnly, 3)
        try file(staged, 4, count: 12_288)
        try fm.linkItem(at: shared, to: second.appendingPathComponent("shared.weights"))
        // Multiple names owned by the same folder still release this allocation only once.
        try fm.linkItem(at: aOnly, to: first.appendingPathComponent("a-hardlink.weights"))
        try fm.linkItem(at: extOnly,
                        to: first.appendingPathComponent("outside.weights"))
        let job = PendingDownload(id: stage.lastPathComponent, repo: "example/pending", size_bytes: 90_000, retained_bytes: 12_288)
        func scan(_ models: [LocalModel]? = nil, _ jobs: [PendingDownload]? = nil) throws -> ModelStorageInventory {
            try ModelStorageInventory.scan(models: models ?? [a,b,ext], pending: jobs ?? [job], managedRoot: managed, hubCache: hub)
        }
        func item(_ snapshot: ModelStorageInventory, _ id: String) throws -> ModelStorageItem {
            guard let row = snapshot.items.first(where: { $0.id == id }) else { throw Failure(message: id) }
            return row
        }
        let inventory = try scan()
        let ra = try item(inventory, "model:first"), rb = try item(inventory, "model:second")
        let incoming = try item(inventory, "download:abcdef123456")
        let regular = try allocated(aOnly), overlap = try allocated(shared), linked = try allocated(extOnly)
        let aMetadata = try [".mlxl3-install.json","config.json","quantization_config.json"].map { try allocated(first.appendingPathComponent($0)) }.reduce(0,+)
        let bMetadata = try [".mlxl3-install.json","config.json","quantization_config.json"].map { try allocated(second.appendingPathComponent($0)) }.reduce(0,+)
        try check(regular > 0 && overlap > 0 && linked > 0, "fixture must allocate blocks")
        try check(ra.kind == .managedModel && ra.canRemoveFiles && rb.canRemoveFiles, "eligible managed model")
        try check(ra.reclaimableBytes == aMetadata + regular, "exclude hardlinks in other folders from recovery")
        try check(rb.reclaimableBytes == bMetadata, "shared-only weights cannot be recovered")
        try check(ra.allocatedBytes == aMetadata + regular + overlap + linked, "duplicate within one folder")
        try check(Set(ra.otherOwners) == Set([b.name,ext.name]), "associated model names")
        try check(inventory.sharedBytes == overlap + linked && ra.sharedBytes == overlap + linked, "deduplicated shared bytes")
        try check(inventory.totalBytes == aMetadata + bMetadata + regular + overlap + linked + (try allocated(staged)), "unique inventory total")
        try check(incoming.kind == .partialDownload && incoming.canRemoveFiles && incoming.reclaimableBytes == (try allocated(staged)), "pending physical bytes")
        let re = try item(inventory, "model:import")
        try check(re.kind == .externalModel && !re.canRemoveFiles && re.reclaimableBytes == 0, "external registration protected")
        try inventory.validateRemoval(ra)

        let snapshotWithAlias = try scan([a,b,ext,local(first, name: "alias")])
        try check(!(try item(snapshotWithAlias,"model:first")).canRemoveFiles, "two registry names, same folder")
        let nested = first.appendingPathComponent("nested")
        try dir(nested)
        try check(!(try item(scan([a,local(nested)]),"model:first")).canRemoveFiles, "nested model protected")
        let aliasLink = external.appendingPathComponent("a-reference")
        try fm.createSymbolicLink(at: aliasLink, withDestinationURL: aOnly)
        let referenced = try item(scan(), "model:first")
        try check(!referenced.canRemoveFiles, "symlink from other model protects its live referent")
        try fm.removeItem(at: aliasLink)
        // An unregistered hardlink elsewhere is visible through the link count.
        let foreign = root.appendingPathComponent("outside-all-models")
        try fm.linkItem(at: aOnly, to: foreign)
        let afterChange = try scan()
        try check(try item(afterChange,"model:first").reclaimableBytes == aMetadata, "unknown hardlink owner")
        do { try afterChange.validateRemoval(ra); throw Failure(message: "stale estimate accepted") }
        catch is MLXL3BridgeError { count += 1 }
        try check(!(try item(scan([local(first, id: "stale-installation")]),"model:first")).canRemoveFiles, "receipt mismatch")

        let legacy = managed.appendingPathComponent("Qwen3.6-exl3-1bde3c20")
        try dir(legacy)
        try Data(#"{"model_type":"qwen"}"#.utf8).write(to: legacy.appendingPathComponent("config.json"))
        try Data(#"{"quant_method":"exl3"}"#.utf8).write(to: legacy.appendingPathComponent("quantization_config.json"))
        let old = local(legacy)
        try check(!(try item(scan([old]),"model:" + old.name)).canRemoveFiles, "legacy folder requires actual weights")
        try file(legacy.appendingPathComponent("model.safetensors"), 6)
        try check(try item(scan([old]),"model:" + old.name).canRemoveFiles, "legacy hub path")
        try check(!(try item(scan([local(legacy,name:"renamed")]),"model:renamed")).canRemoveFiles, "no legacy ownership by alias")
        try Data("{}".utf8).write(to: legacy.appendingPathComponent(".mlxl3-install.json"))
        try check(!(try item(scan([old]),"model:" + old.name)).canRemoveFiles, "legacy candidates cannot bypass a conflicting receipt")

        let aux = hub.appendingPathComponent("models--mlx-community--Qwen-Image-2.1-mflux-q4")
        let snapshots = aux.appendingPathComponent("snapshots/revision"), blobs = aux.appendingPathComponent("blobs")
        try dir(snapshots); try dir(blobs)
        let blob = blobs.appendingPathComponent("hash")
        try file(blob, 5)
        try fm.createSymbolicLink(at: snapshots.appendingPathComponent("weights"), withDestinationURL: blob)
        let hf = try scan([local(snapshots)], [])
        try check(hf.items.count == 2 && hf.totalBytes == (try allocated(blob)), "HF cache and snapshot deduplicated")
        try check(hf.items.allSatisfy { !$0.canRemoveFiles } && hf.items.contains { $0.kind == .sharedCache }, "HF cache read-only")

        count += try await checkActions(root: root, managed: managed, hub: hub, stage: stage, model: a, job: job)

        let bad = PendingDownload(id: "../escape", repo: "x", size_bytes: 1, retained_bytes: 1)
        try check(try scan([], [bad]).items.allSatisfy { $0.kind != .partialDownload }, "invalid pending identifier")
        try fm.moveItem(at: stage, to: root.appendingPathComponent("old-stage"))
        try fm.createSymbolicLink(at: stage, withDestinationURL: external)
        try check(!(try item(scan(),"download:" + job.id)).canRemoveFiles, "symlinked download stage")
        let vanished = try item(scan([local(root.appendingPathComponent("missing"))]),"model:missing")
        try check(!vanished.complete && !vanished.canRemoveFiles, "missing directory")
        let clone = second.appendingPathComponent("clone")
        if clonefile(aOnly.path, clone.path, 0) == 0 {
            let cloned = try scan()
            // A clone outside the selected folder owns the blocks even after the hardlink is removed.
            try fm.removeItem(at: foreign)
            let final = try scan()
            try check(try item(final,"model:first").reclaimableBytes == aMetadata, "APFS clone shared private blocks excluded")
            try check(cloned.totalBytes == final.totalBytes, "removing an external name cannot allocate more space")
        } else if errno != ENOTSUP { throw Failure(message: "unexpected clonefile failure") }

        print("Model storage: \(count) checks passed")
    }

    @MainActor
    static func checkActions(root: URL, managed: URL, hub: URL, stage: URL,
                             model: LocalModel, job: PendingDownload) async throws -> Int {
        let fm = FileManager.default
        let script = root.appendingPathComponent("fixture-cli")
        let models = root.appendingPathComponent("models.json")
        try JSONEncoder().encode([model]).write(to: models)
        let list = root.appendingPathComponent("jobs.json")
        func writeJob() throws {
            try JSONSerialization.data(withJSONObject: [["id":job.id, "repo":job.repo, "size_bytes":job.size_bytes,
                                                          "retained_bytes":job.retained_bytes]]).write(to: list)
        }
        try writeJob()
        // Only the isolated fixture's state changes when the CLI receives the exact expected request.
        try """
        #!/bin/sh
        set -eu
        case "$1 $2" in
          "list --json") cat "$MLXL3_STORAGE_TEST/models.json" ;;
          "hub pending") cat "$MLXL3_STORAGE_TEST/jobs.json" ;;
          "hub discard")
            [ "$3" = abcdef123456 ] || exit 7
            printf '[]' > "$MLXL3_STORAGE_TEST/jobs.json"
            printf 'discarded' > "$MLXL3_STORAGE_TEST/discarded"
            ;;
          *) exit 5 ;;
        esac
        """.write(to: script, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        let overrides = ["MLXL3_EXECUTABLE":script.path, "MLXL3_STORAGE_TEST":root.path,
                         "MLXL3_MODELS_DIR":managed.path, "HF_HUB_CACHE":hub.path]
        let prior = ProcessInfo.processInfo.environment
        for (key, val) in overrides { setenv(key, val, 1) }
        defer {
            for key in overrides.keys {
                if let old = prior[key] { setenv(key, old, 1) } else { unsetenv(key) }
            }
        }
        let suite = "mlxl3-storage-fixture-" + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("fixture-workspace.json"),
                                 isPreview: true, preferences: preferences)
        defer { studio.stopServices() }
        let library = studio.modelLibrary
        let inventory = try ModelStorageInventory.scan(models: [model], pending: [job])
        guard let preview = inventory.items.first(where: { $0.downloadID == job.id }) else {
            throw Failure(message: "partial preview missing")
        }
        func waitForDiscard() async throws {
            for _ in 0..<350 {
                if library.discardingStorageID == nil { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw Failure(message: "discard did not finish")
        }
        try Data(repeating: 99, count: 131072).write(to: stage.appendingPathComponent("new.part"))
        library.discard(job, storagePreview: preview, studio: studio)
        try await waitForDiscard()
        let received = root.appendingPathComponent("discarded")
        guard library.error != nil, !fm.fileExists(atPath: received.path) else {
            throw Failure(message: "stale preview should never reach discard")
        }
        let current = try ModelStorageInventory.scan(models: [model], pending: [job])
        guard let valid = current.items.first(where: { $0.downloadID == job.id }) else {
            throw Failure(message: "refreshed partial preview missing")
        }
        library.discard(job, storagePreview: valid, studio: studio)
        try await waitForDiscard()
        guard library.error == nil, library.pending.isEmpty, fm.fileExists(atPath: received.path),
              fm.fileExists(atPath: model.path) else { throw Failure(message: "confirmed discard did not refresh the inventory") }
        return 2
    }
}
