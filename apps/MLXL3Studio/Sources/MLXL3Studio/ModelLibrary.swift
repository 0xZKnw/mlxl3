import AppKit
import SwiftUI

struct HubModel: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let downloads: Int
    let likes: Int
    let gated: Bool
}

struct HubVariant: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let label: String
    let sizeBytes: Int64
    enum CodingKeys: String, CodingKey { case id, label; case sizeBytes = "size_bytes" }
}

struct HubDetails: Decodable, Sendable {
    let id: String
    let revision: String
    let commit: String
    let branches: [String]
    let variants: [HubVariant]
    let readme: String
    let gated: Bool
    let downloads: Int
    let likes: Int
}

private struct HubDownloadEvent: Decodable {
    let type: String
    let completed: Double?
    let total: Double?
    let model: LocalModel?
}

@MainActor
final class ModelLibrary: ObservableObject {
    @Published var query = ""
    @Published private(set) var results: [HubModel] = []
    @Published private(set) var detail: HubDetails?
    @Published private(set) var searching = false
    @Published private(set) var loadingDetail = false
    @Published private(set) var error: String?
    @Published private(set) var downloading: String?
    @Published private(set) var completed = 0.0
    @Published private(set) var total = 0.0
    @Published private(set) var downloadMessage: String?
    @Published var variantID = ""
    private var searchTask: Task<Void, Never>?
    private var detailTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?
    private var searchCache: [String: [HubModel]] = [:]
    private var detailCache: [String: HubDetails] = [:]

    var selectedVariant: HubVariant? { detail?.variants.first { $0.id == variantID } }

    func search() {
        searchTask?.cancel()
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        searching = true
        error = nil
        searchTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(300))
                if let cached = searchCache[text] { results = cached; searching = false; return }
                let data = try await CLICommand().output(["hub", "search", text])
                try Task.checkCancellation()
                let values = try JSONDecoder().decode([HubModel].self, from: data)
                if searchCache.count >= 30 { searchCache.removeAll() }
                searchCache[text] = values
                results = values
                searching = false
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription; searching = false }
        }
    }

    func open(_ repo: String, revision: String? = nil) {
        detailTask?.cancel()
        loadingDetail = true
        error = nil
        let key = repo + "@" + (revision ?? "auto")
        detailTask = Task {
            do {
                let value: HubDetails
                if let cached = detailCache[key] { value = cached }
                else {
                    var arguments = ["hub", "details", repo]
                    if let revision { arguments += ["--revision", revision] }
                    let data = try await CLICommand().output(arguments)
                    value = try JSONDecoder().decode(HubDetails.self, from: data)
                    if detailCache.count >= 20 { detailCache.removeAll() }
                    detailCache[key] = value
                }
                try Task.checkCancellation()
                detail = value
                variantID = value.variants.first?.id ?? ""
                loadingDetail = false
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription; loadingDetail = false }
        }
    }

    func back() {
        detailTask?.cancel()
        loadingDetail = false
        detail = nil
        error = nil
    }

    func download(studio: StudioModel) {
        guard downloading == nil, let detail, let variant = selectedVariant else { return }
        downloading = detail.id
        completed = 0
        total = Double(variant.sizeBytes)
        downloadMessage = nil
        downloadTask = Task {
            do {
                let data = try await CLICommand().output([
                    "hub", "download", detail.id, "--revision", detail.commit, "--folder", variant.id
                ]) { [weak self] line in
                    guard let self, let event = try? JSONDecoder().decode(HubDownloadEvent.self, from: line), event.type == "progress" else { return }
                    self.completed = event.completed ?? self.completed
                    self.total = event.total ?? self.total
                }
                let event = try JSONDecoder().decode(HubDownloadEvent.self, from: data)
                guard event.type == "installed", let model = event.model else { throw MLXL3BridgeError.invalidResponse }
                completed = total
                downloadMessage = L("\(model.name) ajouté à la bibliothèque.", "\(model.name) added to your library.")
                studio.refreshModels(autoLoad: false)
            } catch is CancellationError {
                downloadMessage = L("Téléchargement suspendu. Relance la même variante pour reprendre.", "Download paused. Download the same variant again to resume.")
            } catch { downloadMessage = error.localizedDescription }
            downloading = nil
            downloadTask = nil
        }
    }

    func cancelDownload() { downloadTask?.cancel() }
}
