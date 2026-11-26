import AppKit
import QuickLookUI
import SwiftUI

@MainActor
func selectAgentDeliverableDestination(_ panel: NSSavePanel) -> URL? {
  guard panel.runModal() == .OK else { return nil }
  return panel.url
}

/// The transcript links to the snapshots of this exchange, even after later edits.
struct AgentTurnResultRow: View {
  @EnvironmentObject private var studio: StudioModel
  let workspace: URL
  let result: AgentTurnResult
  var selectDestination: @MainActor (NSSavePanel) -> URL? = selectAgentDeliverableDestination
  @State private var expanded = false
  @State private var pendingPath: String?
  @State private var preview: AgentDeliverablePreview?
  @State private var error: String?

  var body: some View {
    let files = result.deliverableFiles
    VStack(alignment: .leading, spacing: 6) {
      if !files.isEmpty {
        LazyVStack(alignment: .leading, spacing: 1) {
          ForEach(files.prefix(expanded ? files.count : 3), id: \.path) { file in
            deliverable(file.path)
          }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background {
          RoundedRectangle(cornerRadius: 9)
            .fill(StudioTheme.panel)
        }
        .overlay {
          RoundedRectangle(cornerRadius: 9)
            .strokeBorder(StudioTheme.edge, lineWidth: 0.5)
            .allowsHitTesting(false)
        }
      }

      HStack(spacing: 10) {
        if let path = result.newDocument {
          Button {
            studio.studioDocument = StudioDocument(workspace: workspace,
              file: workspace.appendingPathComponent(path), turnResult: result)
          } label: {
            Label(L("Ouvrir dans Agent Studio", "Open in Agent Studio"), systemImage: "doc")
          }
          .buttonStyle(.studio)
          .accessibilityIdentifier("agent.turn.document")
        }
        Button {
          studio.studioDocument = StudioDocument(workspace: workspace, turnResult: result)
        } label: {
          let count = result.files.count
          Label(count == 1 ? L("Examiner 1 fichier", "Review 1 file")
            : L("Examiner \(count) fichiers", "Review \(count) files"), systemImage: "plus.forwardslash.minus")
        }
        .buttonStyle(.studio)
        .accessibilityIdentifier("agent.turn.review")
        .accessibilityHint(L("Afficher les modifications de cette réponse", "Show the changes made during this response"))

        if files.count > 3 {
          Button {
            expanded.toggle()
          } label: {
            Label(expanded ? L("Réduire la liste", "Show fewer")
              : L("Afficher les \(files.count) fichiers", "Show all \(files.count) files"),
              systemImage: expanded ? "chevron.up" : "chevron.down")
          }
          .buttonStyle(.studio)
          .accessibilityIdentifier("agent.deliverable.expand")
        }
      }
    }
    .sheet(item: $preview) { preview in
      AgentDeliverablePreviewSheet(file: preview)
    }
    .alert(L("Fichier indisponible", "File unavailable"), isPresented: Binding(
      get: { error != nil }, set: { if !$0 { error = nil } }
    )) {
      Button(L("Fermer", "Close"), role: .cancel) {}
    } message: {
      if let error { Text(error) }
    }
  }

  private func deliverable(_ path: String) -> some View {
    let file = URL(fileURLWithPath: path)
    return HStack(spacing: 8) {
      Image(systemName: "doc").foregroundStyle(StudioTheme.secondary)
      VStack(alignment: .leading, spacing: 1) {
        Text(file.lastPathComponent)
          .font(.system(size: 12, weight: .medium))
          .truncationMode(.middle)
          .lineLimit(1)
        if let separator = path.lastIndex(of: "/") {
          Text(path[..<separator])
            .font(.system(size: 10.5))
            .foregroundStyle(StudioTheme.secondary)
            .truncationMode(.middle)
            .lineLimit(1)
        }
      }
      .help(path)
      Spacer(minLength: 4)
      if pendingPath == path {
        ProgressView().controlSize(.small)
      }
      HStack(spacing: 4) {
        Button {
          prepare(path)
        } label: {
          Label(L("Aperçu", "Preview"), systemImage: "eye")
        }
        .buttonStyle(.studio)
        .help(L("Aperçu de \(path)", "Preview \(path)"))
        .accessibilityLabel(L("Aperçu de \(path)", "Preview \(path)"))
        .accessibilityIdentifier("agent.deliverable.preview.\(path)")

        Button {
          let panel = NSSavePanel()
          panel.nameFieldStringValue = file.lastPathComponent
          panel.canCreateDirectories = true
          guard let destination = selectDestination(panel) else { return }
          prepare(path, destination: destination)
        } label: {
          Label(L("Enregistrer…", "Save…"), systemImage: "square.and.arrow.down")
        }
        .buttonStyle(.studio)
        .help(L("Enregistrer \(path)", "Save \(path)"))
        .accessibilityLabel(L("Enregistrer \(path)", "Save \(path)"))
        .accessibilityIdentifier("agent.deliverable.save.\(path)")
      }
      .disabled(pendingPath != nil)
    }
    .padding(.vertical, 3)
  }

  private func prepare(_ path: String, destination: URL? = nil) {
    guard pendingPath == nil else { return }
    pendingPath = path
    let store = studio.agentStore(for: workspace)
    let result = result
    Task {
      defer { pendingPath = nil }
      do {
        preview = try await Task.detached(priority: .userInitiated) { () -> AgentDeliverablePreview? in
          let bytes = try store.deliverableContents(path, in: result)
          if let destination {
            try bytes.write(to: destination, options: .atomic)
            return nil
          }
          return try AgentDeliverablePreview(name: URL(fileURLWithPath: path).lastPathComponent, bytes: bytes)
        }.value
      } catch {
        self.error = error.localizedDescription
      }
    }
  }
}

private final class AgentDeliverablePreview: Identifiable, Sendable {
  private let directory: URL
  let url: URL

  init(name: String, bytes: Data) throws {
    let manager = FileManager.default
    let directory = manager.temporaryDirectory.appendingPathComponent("mlxl3-deliverable-\(UUID())", isDirectory: true)
    try manager.createDirectory(at: directory, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    let url = directory.appendingPathComponent(name, isDirectory: false)
    do {
      try bytes.write(to: url, options: .withoutOverwriting)
    } catch {
      try? manager.removeItem(at: directory)
      throw error
    }
    self.directory = directory
    self.url = url
  }

  deinit {
    try? FileManager.default.removeItem(at: directory)
  }
}

private struct AgentDeliverablePreviewSheet: View {
  let file: AgentDeliverablePreview
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 10) {
        Text(file.url.lastPathComponent)
          .font(.system(size: 13, weight: .semibold))
          .lineLimit(1)
          .truncationMode(.middle)
        Spacer()
        Button(L("Fermer", "Close")) { dismiss() }
          .buttonStyle(.studio)
          .keyboardShortcut(.cancelAction)
          .accessibilityIdentifier("agent.deliverable.dismiss")
      }
      .padding(.vertical, 10)
      .padding(.horizontal, 14)
      Divider()
      AgentDeliverableQuickLook(file: file)
    }
    .frame(minWidth: 530, idealWidth: 720, minHeight: 400, idealHeight: 560)
  }
}

private struct AgentDeliverableQuickLook: NSViewRepresentable {
  let file: AgentDeliverablePreview

  func makeNSView(context: Context) -> QLPreviewView {
    let view = QLPreviewView(frame: .zero, style: .normal)!
    view.setAccessibilityIdentifier("agent.deliverable.quicklook")
    view.autostarts = true
    return view
  }

  func updateNSView(_ view: QLPreviewView, context: Context) {
    if view.previewItem?.previewItemURL != file.url {
      view.previewItem = file.url as NSURL
    }
  }

  static func dismantleNSView(_ view: QLPreviewView, coordinator: ()) {
    view.close()
  }
}
