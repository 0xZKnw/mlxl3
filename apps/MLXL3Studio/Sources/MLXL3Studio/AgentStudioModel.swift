import Foundation
import SwiftUI

enum StudioTab: String, CaseIterable, Identifiable {
  case preview, code, diff, history
  var id: String { rawValue }
  var title: String {
    switch self {
    case .preview: L("Aperçu", "Preview")
    case .code: "Code"
    case .diff: "Diff"
    case .history: L("Historique", "History")
    }
  }
  var icon: String {
    switch self {
    case .preview: "play.rectangle"
    case .code: "chevron.left.forwardslash.chevron.right"
    case .diff: "plus.forwardslash.minus"
    case .history: "clock.arrow.circlepath"
    }
  }
}

@MainActor
final class AgentStudioModel: ObservableObject {
  @Published private(set) var store: AgentProjectStore?
  @Published private(set) var state = AgentProjectState()
  @Published private(set) var versions: [AgentCheckpoint] = []
  @Published private(set) var paths: [String] = []
  @Published private(set) var selected = ""
  @Published private(set) var revision = ""
  @Published private(set) var compare = ""
  @Published private(set) var savedCode = ""
  @Published var code = ""
  @Published private(set) var binary = false
  @Published private(set) var fileExists = false
  @Published private(set) var diff = ""
  @Published var tab: StudioTab = .code
  @Published private(set) var allChanges = false
  @Published private(set) var busy = false
  @Published private(set) var loading = false
  @Published var error: String?
  private var selectionTicket = UUID()

  var isHTML: Bool {
    ["html", "htm"].contains(URL(fileURLWithPath: selected).pathExtension.lowercased())
  }
  var isCurrent: Bool { !revision.isEmpty && revision == versions.first?.id }
  var editable: Bool { isCurrent && fileExists && !binary && !loading }
  var dirty: Bool { editable && code != savedCode }
  var canApply: Bool {
    tab == .diff && allChanges && isCurrent && compare == state.baseline && !diff.isEmpty && !dirty
  }
  var title: String {
    state.source.map { URL(fileURLWithPath: $0).lastPathComponent }
      ?? L("Dossier de la tâche", "Task workspace")
  }

  func open(workspace: URL, file: URL?) async {
    guard store == nil, !busy else { return }
    busy = true
    defer { busy = false }
    do {
      let value = AgentProjectStore(workspace: workspace)
      store = value
      tab = ["html", "htm"].contains(file?.pathExtension.lowercased() ?? "") ? .preview : .code
      if let file {
        selected = String(file.standardizedFileURL.path.dropFirst(value.workspace.path.count + 1))
      }
      try await refreshSnapshot(value)
    } catch { self.error = error.localizedDescription }
  }

  func reload() async {
    guard let store, !busy, !dirty else { return }
    busy = true
    error = nil
    defer { busy = false }
    do { try await refreshSnapshot(store) } catch { self.error = error.localizedDescription }
  }

  private func refreshSnapshot(_ store: AgentProjectStore) async throws {
    let snapshot = try await Task.detached {
      try store.ensureState()
      _ = try store.checkpoint("Workspace snapshot", force: false)
      return (try store.state(), try store.versions())
    }.value
    state = snapshot.0
    versions = snapshot.1
    revision = versions.first?.id ?? ""
    compare = state.baseline ?? versions.dropFirst().first?.id ?? revision
    await loadSelection()
  }

  @discardableResult func select(_ path: String) async -> Bool {
    guard !dirty, !busy, paths.contains(path) else { return false }
    selected = path
    allChanges = false
    await loadSelection()
    return true
  }

  @discardableResult func selectRevision(_ value: String) async -> Bool {
    guard !dirty, !busy, versions.contains(where: { $0.id == value }) else { return false }
    revision = value
    await loadSelection()
    return true
  }

  func compareWith(_ value: String) async {
    guard !dirty, !busy, versions.contains(where: { $0.id == value }) else { return }
    compare = value
    await loadSelection()
  }

  func reviewProject() async {
    guard !dirty, !busy, let baseline = state.baseline, let latest = versions.first?.id else {
      return
    }
    revision = latest
    compare = baseline
    allChanges = true
    tab = .diff
    await loadSelection()
  }

  func showAllChanges(_ value: Bool) async {
    guard !dirty, !busy else { return }
    allChanges = value
    await loadSelection()
  }

  private func loadSelection() async {
    guard let store, !revision.isEmpty else { return }
    let ticket = UUID()
    selectionTicket = ticket
    loading = true
    fileExists = false
    savedCode = ""
    code = ""
    diff = ""
    error = nil
    let revision = revision
    let compare = compare
    let requested = selected
    let all = allChanges
    defer { if selectionTicket == ticket { loading = false } }
    do {
      let result = try await Task.detached {
        let current = try store.paths(at: revision)
        let before = try store.paths(at: compare)
        let paths = Array(Set(current + before)).sorted()
        let path = paths.contains(requested) ? requested : current.first ?? paths.first ?? ""
        let data = current.contains(path) ? try store.contents(path, revision: revision) : Data()
        let diff = try store.diff(
          from: compare, to: revision, path: all || path.isEmpty ? nil : path)
        return (paths, path, data, current.contains(path), diff)
      }.value
      guard selectionTicket == ticket, !Task.isCancelled else { return }
      paths = result.0
      selected = result.1
      fileExists = result.3
      diff = result.4
      binary = result.2.contains(0) || String(data: result.2, encoding: .utf8) == nil
      savedCode = binary ? "" : String(decoding: result.2, as: UTF8.self)
      code = savedCode
      if tab == .preview && !isHTML { tab = .code }
    } catch {
      if selectionTicket == ticket && !Task.isCancelled { self.error = error.localizedDescription }
    }
  }

  func discardEdits() { code = savedCode }

  @discardableResult func save() async -> Bool {
    guard let store, !busy, dirty else { return false }
    let path = selected
    let text = code
    let revision = revision
    return await mutate { try store.saveFile(path, text: text, revision: revision) }
  }

  func checkpoint() async {
    guard let store, !dirty else { return }
    _ = await mutate { _ = try store.checkpoint("Manual checkpoint") }
  }

  func restore() async {
    guard let store, !dirty, !isCurrent else { return }
    let value = revision
    _ = await mutate { try store.restore(value) }
  }

  func apply() async {
    guard let store, canApply else { return }
    let value = revision
    _ = await mutate { try store.apply(reviewedRevision: value) }
  }

  func setPermissions(_ value: AgentPermissions) async {
    guard let store, !busy else { return }
    busy = true
    error = nil
    defer { busy = false }
    do {
      try await Task.detached { try store.setPermissions(value) }.value
      state.permissions = value
    } catch { self.error = error.localizedDescription }
  }

  func exportZIP(to destination: URL) async {
    guard let store, !busy, !dirty, !revision.isEmpty else { return }
    let revision = revision
    busy = true
    error = nil
    defer { busy = false }
    do {
      try await Task.detached { try store.exportZIP(destination, revision: revision) }.value
    } catch { self.error = error.localizedDescription }
  }

  private func mutate(_ action: @escaping @Sendable () throws -> Void) async -> Bool {
    guard let store, !busy else { return false }
    busy = true
    error = nil
    defer { busy = false }
    do {
      try await Task.detached { try action() }.value
      try await refreshSnapshot(store)
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
}
