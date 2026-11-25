import Darwin
import Foundation

enum ModelStorageKind: Hashable, Sendable { case managedModel, externalModel, partialDownload, sharedCache }

struct ModelStorageItem: Identifiable, Equatable, Sendable {
    let id: String
    let name: String
    let path: String
    let kind: ModelStorageKind
    let registryName: String?
    let downloadID: String?
    fileprivate(set) var allocatedBytes: Int64 = 0
    fileprivate(set) var reclaimableBytes: Int64 = 0
    fileprivate(set) var sharedBytes: Int64 = 0
    fileprivate(set) var otherOwners: [String] = []
    fileprivate(set) var canRemoveFiles: Bool
    fileprivate(set) var complete = true
    // Distinguish a fresh installation or staging directory at the same path.
    fileprivate var identity: StorageFileID?
    fileprivate var installationID: String?
}

fileprivate struct StorageFileID: Hashable, Sendable {
    let device: dev_t
    let inode: ino_t

    init(_ data: stat) { device = data.st_dev; inode = data.st_ino }
    init(device: dev_t, inode: ino_t) { self.device = device; self.inode = inode }
}

private struct StorageFile {
    let bytes: Int64
    let privateBytes: Int64
    let cloneGroup: StorageFileID?
    let links: Int
    var owners = Set<String>()
    // An inode allocated once can have several removable directory entries, even in one model.
    var directPaths = Set<String>()
}

private struct StorageReceipt: Decodable {
    let attempt: String
    let request: Request
    struct Request: Decodable { let name: String; let destination: String }
}

struct ModelStorageInventory: Sendable {
    let items: [ModelStorageItem]
    let totalBytes: Int64
    let sharedBytes: Int64

    private static func environmentPath(_ key: String) -> URL? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else { return nil }
        return URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
    }

    static var defaultManagedRoot: URL {
        environmentPath("MLXL3_MODELS_DIR") ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/io.mlxl3.desktop/Models")
    }

    static var defaultHubCache: URL {
        if let root = environmentPath("HF_HUB_CACHE") ?? environmentPath("HUGGINGFACE_HUB_CACHE") { return root }
        if let root = environmentPath("HF_HOME") { return root.appendingPathComponent("hub") }
        let cache = environmentPath("XDG_CACHE_HOME") ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache")
        return cache.appendingPathComponent("huggingface/hub")
    }

    private static func metadata(_ path: String) -> stat? {
        var data = stat()
        return lstat(path, &data) == 0 ? data : nil
    }

    private static func type(_ data: stat) -> mode_t { data.st_mode & S_IFMT }
    private static func contains(_ root: String, _ path: String) -> Bool {
        root == path || path.hasPrefix(root == "/" ? "/" : root + "/")
    }

    private static func safeToScan(_ root: String) -> Bool {
        let home = FileManager.default.homeDirectoryForCurrentUser.resolvingSymlinksInPath().path
        guard !contains(root, home) else { return false }
        let reserved = ["/Applications", "/Library", "/System", "/Volumes", "/usr", "/bin", "/etc", "/private",
                        home + "/Documents", home + "/Desktop", home + "/Downloads", home + "/Library"]
        return !reserved.contains(root) && URL(fileURLWithPath: root).pathComponents.count >= 3
    }

    private static func ownedInstallation(_ model: LocalModel, at path: URL, root: String) -> Bool {
        guard model.path.hasPrefix("/"), path.deletingLastPathComponent().path == root,
              !path.lastPathComponent.hasPrefix("."), let data = metadata(model.path), type(data) == S_IFDIR else { return false }
        guard metadata(path.appendingPathComponent(".git").path) == nil, errno == ENOENT,
              ["config.json", "quantization_config.json"].allSatisfy({ name in
                  metadata(path.appendingPathComponent(name).path).map { type($0) == S_IFREG } == true
              }) else { return false }
        let receipt = path.appendingPathComponent(".mlxl3-install.json")
        if model.installationID == nil {
            // Older hub installations used the repository name and eight hex characters from the download ID.
            guard metadata(receipt.path) == nil, errno == ENOENT, model.name == path.lastPathComponent,
                  let suffix = model.name.split(separator: "-").last, suffix.utf8.count == 8,
                  model.name.hasSuffix("-" + suffix),
                  suffix.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { return false }
            // The registry checked the checkpoint on import. Also confirm the model type and
            // the presence of at least one locally owned shard when inferring older ownership.
            struct Config: Decodable { let model_type: String }
            let config = path.appendingPathComponent("config.json")
            guard let attributes = metadata(config.path), attributes.st_size <= 1_000_000,
                  let bytes = try? Data(contentsOf: config), let decoded = try? JSONDecoder().decode(Config.self, from: bytes),
                  decoded.model_type == model.modelType, let entries = try? FileManager.default.contentsOfDirectory(atPath: path.path) else { return false }
            return entries.contains { name in
                name.hasSuffix(".safetensors") && metadata(path.appendingPathComponent(name).path).map {
                    type($0) == S_IFREG && $0.st_size > 8
                } == true
            }
        }
        guard let id = model.installationID, !id.isEmpty,
              let attr = metadata(receipt.path), type(attr) == S_IFREG, attr.st_size <= 1_000_000,
              let bytes = try? Data(contentsOf: receipt),
              let record = try? JSONDecoder().decode(StorageReceipt.self, from: bytes) else { return false }
        return record.attempt == id && record.request.name == model.name && record.request.destination == model.path
    }

    private static func allocation(_ path: String, _ data: stat) -> StorageFile {
        let bytes = Int64(data.st_blocks) * 512
        var file = StorageFile(bytes: bytes, privateBytes: bytes, cloneGroup: nil, links: Int(data.st_nlink))
        guard bytes > 0 else { return file }
        // Darwin packs results to 4-byte boundaries: length, attribute_set_t, then the two 64-bit fields.
        var request = attrlist()
        request.bitmapcount = UInt16(ATTR_BIT_MAP_COUNT)
        request.commonattr = attrgroup_t(ATTR_CMN_RETURNED_ATTRS)
        request.forkattr = attrgroup_t(ATTR_CMNEXT_PRIVATESIZE | ATTR_CMNEXT_CLONEID)
        var buffer = [UInt32](repeating: 0, count: 10)
        buffer.withUnsafeMutableBytes { result in
            guard getattrlist(path, &request, result.baseAddress, result.count,
                              UInt32(FSOPT_ATTR_CMN_EXTENDED | FSOPT_NOFOLLOW)) == 0,
                  result.load(as: UInt32.self) == result.count,
                  result.loadUnaligned(fromByteOffset: 4, as: attribute_set_t.self).forkattr == request.forkattr else { return }
            let exclusive = result.loadUnaligned(fromByteOffset: 24, as: Int64.self)
            let clone = result.loadUnaligned(fromByteOffset: 32, as: UInt64.self)
            guard (0...bytes).contains(exclusive), exclusive < bytes, clone > 0 else { return }
            file = StorageFile(bytes: bytes, privateBytes: exclusive,
                               cloneGroup: StorageFileID(device: data.st_dev, inode: clone), links: Int(data.st_nlink))
        }
        return file
    }

    func validateRemoval(_ preview: ModelStorageItem) throws {
        guard preview.canRemoveFiles, let item = items.first(where: { $0.id == preview.id }),
              item == preview, let current = Self.metadata(item.path), item.identity == StorageFileID(current),
              Self.type(current) == S_IFDIR else {
            throw MLXL3BridgeError.commandFailed(L("Le stockage a changé. Actualise la liste avant de continuer.",
                                                  "Storage changed. Refresh the list before continuing."))
        }
    }

    static func scan(models: [LocalModel], pending: [PendingDownload], imageModelPath: String? = nil,
                     managedRoot: URL = defaultManagedRoot, hubCache: URL = defaultHubCache) throws -> Self {
        let managed = managedRoot.resolvingSymlinksInPath().standardizedFileURL
        let validRoot = metadata(managedRoot.path).map { type($0) == S_IFDIR } == true
        let hub = hubCache.resolvingSymlinksInPath().standardizedFileURL
        let downloads = managed.appendingPathComponent(".downloads")
        var rows: [ModelStorageItem] = []
        var cacheRepos = Set<String>()
        let imageAux = "models--mlx-community--Qwen-Image-2.1-mflux-q4"
        if metadata(hub.appendingPathComponent(imageAux).path) != nil { cacheRepos.insert(imageAux) }

        for model in models {
            let url = URL(fileURLWithPath: model.path).resolvingSymlinksInPath().standardizedFileURL
            let owned = validRoot && ownedInstallation(model, at: url, root: managed.path)
            rows.append(.init(id: "model:" + model.name, name: model.name, path: url.path,
                              kind: owned ? .managedModel : .externalModel, registryName: model.name, downloadID: nil,
                              canRemoveFiles: owned, installationID: model.installationID))
        }
        if let imageModelPath, imageModelPath.hasPrefix("/") {
            let path = URL(fileURLWithPath: imageModelPath).resolvingSymlinksInPath().standardizedFileURL.path
            if FileManager.default.fileExists(atPath: path) {
                rows.append(.init(id: "image:" + path, name: "Qwen-Image-2.1", path: path, kind: .externalModel,
                                  registryName: nil, downloadID: nil, canRemoveFiles: false))
            }
        }
        for row in rows where row.path.hasPrefix(hub.path + "/") {
            let component = row.path.dropFirst(hub.path.count + 1).split(separator: "/").first.map(String.init)
            if let component, component.hasPrefix("models--") { cacheRepos.insert(component) }
        }
        for repo in cacheRepos.sorted() {
            let root = hub.appendingPathComponent(repo)
            guard let data = metadata(root.path), type(data) == S_IFDIR else { continue }
            rows.append(.init(id: "cache:" + repo, name: String(repo.dropFirst("models--".count)).replacingOccurrences(of: "--", with: "/"),
                              path: root.path, kind: .sharedCache, registryName: nil, downloadID: nil, canRemoveFiles: false))
        }
        for job in pending {
            guard job.id.utf8.count == 12, job.id.utf8.allSatisfy({ (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) }) else { continue }
            let stage = downloads.appendingPathComponent(job.id)
            let regular = validRoot && metadata(downloads.path).map { type($0) == S_IFDIR } == true
                && metadata(stage.path).map { type($0) == S_IFDIR } == true
            rows.append(.init(id: "download:" + job.id, name: job.repo, path: stage.path, kind: .partialDownload,
                              registryName: nil, downloadID: job.id, canRemoveFiles: regular, complete: regular))
        }

        var files: [StorageFileID: StorageFile] = [:]
        // A symlink into another model must survive removing its owner's folder.
        var references: [String: Set<String>] = [:]
        for index in rows.indices {
            try Task.checkCancellation()
            let row = rows[index]
            guard row.complete, safeToScan(row.path), let rootData = metadata(row.path), type(rootData) == S_IFDIR else {
                rows[index].complete = false
                rows[index].canRemoveFiles = false
                continue
            }
            rows[index].identity = StorageFileID(rootData)
            var complete = true
            let root = URL(fileURLWithPath: row.path)
            guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [],
                                                                   errorHandler: { _, _ in complete = false; return true }) else {
                rows[index].complete = false
                rows[index].canRemoveFiles = false
                continue
            }
            for case let entry as URL in enumerator {
                try Task.checkCancellation()
                guard var data = metadata(entry.path) else { complete = false; continue }
                let direct = type(data) != S_IFLNK
                var allocationPath = entry.path
                if !direct {
                    let target = entry.resolvingSymlinksInPath().standardizedFileURL.path
                    references[row.id, default: []].insert(target)
                    guard let targetData = metadata(target) else { complete = false; continue }
                    data = targetData
                    allocationPath = target
                    // Do not walk directory links or cycles.
                    if type(data) == S_IFDIR { complete = false; continue }
                }
                guard type(data) == S_IFREG else { continue }
                let id = StorageFileID(data)
                if files[id] == nil {
                    files[id] = allocation(allocationPath, data)
                }
                files[id]!.owners.insert(row.id)
                if direct { files[id]!.directPaths.insert(entry.standardizedFileURL.path) }
            }
            rows[index].complete = complete
            if !complete { rows[index].canRemoveFiles = false }
        }
        // Detect both directions of nesting, same-directory aliases and symlinks from another registration.
        let roots = rows
        var peers: [String: Set<String>] = [:]
        for index in rows.indices {
            let root = rows[index]
            for other in roots where root.id != other.id {
                let overlaps = (root.identity != nil && root.identity == other.identity)
                    || contains(root.path, other.path) || contains(other.path, root.path)
                let referenced = references[other.id]?.contains(where: { contains(root.path, $0) }) == true
                if overlaps || referenced {
                    rows[index].canRemoveFiles = false
                    peers[root.id, default: []].insert(other.name)
                }
            }
        }
        guard Set(rows.map(\.id)).count == rows.count else { throw CocoaError(.fileReadCorruptFile) }
        let positions = Dictionary(uniqueKeysWithValues: rows.indices.map { (rows[$0].id, $0) })
        var total: Int64 = 0, shared: Int64 = 0
        // Partially diverged clones may share different extents. Count each group's largest shared portion
        // as an estimate, and offer only its known private blocks as recoverable on removal.
        var cloneShares: [StorageFileID: [String: Int64]] = [:]
        for file in files.values {
            try Task.checkCancellation()
            total += file.privateBytes
            if let group = file.cloneGroup {
                let attributed = file.bytes - file.privateBytes
                for owner in file.owners {
                    cloneShares[group, default: [:]][owner] = max(cloneShares[group]?[owner] ?? 0, attributed)
                }
            }
            let isShared = file.owners.count > 1 || file.links > file.directPaths.count
            if isShared { shared += file.privateBytes }
            for owner in file.owners {
                guard let i = positions[owner] else { continue }
                rows[i].allocatedBytes += file.privateBytes
                if isShared {
                    rows[i].sharedBytes += file.privateBytes
                    for peer in file.owners where peer != owner {
                        if let j = positions[peer] { peers[owner, default: []].insert(roots[j].name) }
                    }
                }
                if rows[i].canRemoveFiles && file.directPaths.count == file.links
                    && file.directPaths.allSatisfy({ contains(rows[i].path, $0) }) {
                    rows[i].reclaimableBytes += file.privateBytes
                }
            }
        }
        for owners in cloneShares.values {
            let sharedExtentEstimate = owners.values.max() ?? 0
            total += sharedExtentEstimate
            shared += sharedExtentEstimate
            for (owner, bytes) in owners {
                guard let i = positions[owner] else { continue }
                rows[i].allocatedBytes += bytes
                rows[i].sharedBytes += bytes
                for peer in owners.keys where peer != owner {
                    if let j = positions[peer] { peers[owner, default: []].insert(roots[j].name) }
                }
            }
        }
        for i in rows.indices { rows[i].otherOwners = peers[rows[i].id, default: []].sorted() }
        return Self(items: rows, totalBytes: total, sharedBytes: shared)
    }
}
