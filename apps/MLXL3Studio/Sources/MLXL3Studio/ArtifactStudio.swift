import AppKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

struct ArtifactRevisionRequest: Sendable {
  let workspace: URL
  let file: String
  let source: String
  let instruction: String
}

private struct ArtifactRevisionKey: EnvironmentKey {
  static let defaultValue: (@MainActor @Sendable (ArtifactRevisionRequest) -> Void)? = nil
}

extension EnvironmentValues {
  var artifactRevision: (@MainActor @Sendable (ArtifactRevisionRequest) -> Void)? {
    get { self[ArtifactRevisionKey.self] }
    set { self[ArtifactRevisionKey.self] = newValue }
  }
}

actor ArtifactLibrary {
  static let shared = ArtifactLibrary()

  func open(
    _ source: String, base: URL = ConversationStore.defaultFileURL().deletingLastPathComponent()
  ) throws -> AgentProjectStore {
    guard source.utf8.count <= HTMLPreview.maximumBytes else { throw CocoaError(.fileReadTooLarge) }
    let artifacts = base.appendingPathComponent("artifacts")
    let digest = AgentProjectStore.digest(Data(source.utf8))
    var identifier = digest
    if let data = try? Data(contentsOf: artifacts.appendingPathComponent(digest + ".json")),
      let saved = try? JSONDecoder().decode(String.self, from: data),
      saved.count == 64,
      saved.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
    {
      identifier = saved
    }
    let store = AgentProjectStore(
      workspace: artifacts.appendingPathComponent(identifier + "/files"))
    try FileManager.default.createDirectory(
      at: store.workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700]
    )
    if !FileManager.default.fileExists(
      atPath: store.workspace.appendingPathComponent("index.html").path)
    {
      try AgentProjectStore.write(Data(source.utf8), path: "index.html", root: store.workspace)
      _ = try store.checkpoint("Generated HTML")
    }
    return store
  }

  func append(_ source: String, to workspace: URL, file: String, base: URL) throws {
    let root = base.appendingPathComponent("artifacts").standardizedFileURL.path + "/"
    guard workspace.standardizedFileURL.path.hasPrefix(root), AgentProjectStore.validPath(file)
    else { throw CocoaError(.fileWriteNoPermission) }
    try AgentProjectStore.write(Data(source.utf8), path: file, root: workspace)
    _ = try AgentProjectStore(workspace: workspace).checkpoint("AI revision")
    let digest = AgentProjectStore.digest(Data(source.utf8))
    let identifier = workspace.deletingLastPathComponent().lastPathComponent
    try JSONEncoder().encode(identifier).write(
      to: base.appendingPathComponent("artifacts/\(digest).json"), options: .atomic)
  }
}

/// The project and artifact inspectors share the same files, Git history and preview boundary.
struct ArtifactStudioView: View {
  var source: String? = nil
  var workspace: URL? = nil
  var file: URL? = nil
  var projectMode = false
  @Environment(\.dismiss) private var dismiss
  @Environment(\.artifactRevision) private var requestRevision
  @EnvironmentObject private var studio: StudioModel
  @State private var store: AgentProjectStore?
  @State private var state = AgentProjectState()
  @State private var versions: [AgentCheckpoint] = []
  @State private var revision = ""
  @State private var compare = ""
  @State private var paths: [String] = []
  @State private var selected = "index.html"
  @State private var code = ""
  @State private var original = ""
  @State private var diff = ""
  @State private var tab = "preview"
  @State private var instruction = ""
  @State private var webView: WKWebView?
  @State private var busy = false
  @State private var error: String?

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 12) {
        MonogramMark(size: 24)
        Text(projectMode ? "Agent Studio" : "Artifacts Studio").font(
          .system(size: 23, design: .serif))
        if let source = state.source {
          Text(URL(fileURLWithPath: source).lastPathComponent).foregroundStyle(StudioTheme.quiet)
        }
        Spacer()
        if busy { ProgressView().controlSize(.small) }
        Button(L("Fermer", "Close")) { dismiss() }.keyboardShortcut(.cancelAction)
      }.padding(18)
      Divider()
      if let store {
        HStack(spacing: 12) {
          if projectMode {
            Button(L("Choisir un projet…", "Choose project…"), systemImage: "folder.badge.plus") {
              chooseProject(store)
            }
            .disabled(state.source != nil || !paths.isEmpty)
            .help(
              L(
                "Importer un projet dans une tâche vide. L’agent travaille sur une copie isolée.",
                "Import a project into an empty task. The Agent works on an isolated copy."))
          }
          Button(L("Checkpoint", "Checkpoint"), systemImage: "flag") {
            perform { _ = try store.checkpoint("Manual checkpoint") }
          }
          Button("ZIP", systemImage: "arrow.down.doc") { exportZIP(store) }.disabled(
            revision.isEmpty)
          if projectMode, state.source != nil {
            Button(
              tab == "diff"
                ? L("Appliquer le diff", "Apply changes") : L("Examiner le diff", "Review changes"),
              systemImage: "checkmark.seal"
            ) {
              if tab != "diff" {
                tab = "diff"
                revision = versions.first?.id ?? ""
                compare = state.baseline ?? compare
              } else {
                perform { try store.apply() }
              }
            }
            .disabled(
              diff.isEmpty || revision != versions.first?.id
                || tab == "diff" && compare != state.baseline
            )
            .help(
              L(
                "Appliquer les modifications et suppressions affichées aux originaux. Refus en cas de conflit. Les anciennes versions restent dans les checkpoints.",
                "Apply the displayed edits and deletions to the originals. Conflicts are refused. Previous versions remain in checkpoints."
              ))
          }
          Spacer()
          Button {
            perform {}
          } label: {
            Image(systemName: "arrow.clockwise")
          }.help(L("Actualiser", "Refresh"))
        }.padding(12)
        if projectMode {
          HStack(spacing: 16) {
            Toggle(L("Lecture", "Read files"), isOn: $state.permissions.read)
            Toggle(L("Écriture", "Write files"), isOn: $state.permissions.write)
              .help(
                L(
                  "Autorise l’outil d’écriture et les écritures des commandes dans la copie.",
                  "Allows the write tool and command writes inside the copy."))
            Toggle(L("Commandes", "Commands"), isOn: $state.permissions.run)
              .help(
                L(
                  "Les commandes peuvent lire les fichiers de la copie. Aucun réseau ni accès aux originaux.",
                  "Commands can read the copy's files. No network or original-file access."))
            Stepper(
              "\(state.permissions.commandSeconds) s", value: $state.permissions.commandSeconds,
              in: 1...30
            )
            .help(
              L(
                "Durée maximale de chaque commande, entre 1 et 30 secondes.",
                "Maximum time for each command, from 1 to 30 seconds."))
          }.font(.caption).toggleStyle(.checkbox).fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 14).padding(.bottom, 10)
            .onChange(of: state.permissions) { _, permissions in
              perform { try store.setPermissions(permissions) }
            }
        }
        HSplitView {
          VStack(alignment: .leading) {
            Text(L("FICHIERS", "FILES")).font(.caption2).foregroundStyle(StudioTheme.quiet).padding(
              .horizontal, 12)
            List(paths, id: \.self, selection: $selected) {
              Text($0).font(.system(size: 11, design: .monospaced))
            }
          }.frame(minWidth: 170, idealWidth: 210, maxWidth: 290)
          VStack(spacing: 10) {
            HStack {
              Picker(L("Vue", "View"), selection: $tab) {
                Text(L("Aperçu", "Preview")).tag("preview")
                Text("Code").tag("code")
                Text("Diff").tag("diff")
                Text(L("Historique", "History")).tag("history")
              }.pickerStyle(.segmented)
              if tab == "code" || tab == "diff" {
                Button(L("Enregistrer", "Save")) {
                  let text = code
                  let path = selected
                  perform {
                    try AgentProjectStore.write(Data(text.utf8), path: path, root: store.workspace)
                    _ = try store.checkpoint("Edited \(path)")
                  }
                }.disabled(revision != versions.first?.id)
              }
            }
            HStack {
              Picker(L("Version", "Version"), selection: $revision) {
                ForEach(versions) { Text("\($0.title) · \($0.id.prefix(7))").tag($0.id) }
              }
              if tab == "diff" {
                Picker(L("Comparer avec", "Compare with"), selection: $compare) {
                  ForEach(versions) { Text("\($0.title) · \($0.id.prefix(7))").tag($0.id) }
                }
              }
              if revision != versions.first?.id, !revision.isEmpty {
                Button(L("Restaurer", "Restore")) {
                  let value = revision
                  perform { try store.restore(value) }
                }
              }
            }.font(.caption)
            if tab == "preview" {
              if let webView {
                HTMLWebView(view: webView)
              } else {
                ContentUnavailableView(
                  L("Sélectionnez un fichier HTML", "Select an HTML file"), systemImage: "safari")
              }
            } else if tab == "code" {
              TextEditor(text: $code).font(.system(size: 12, design: .monospaced)).disabled(
                revision != versions.first?.id)
            } else if tab == "diff" {
              VSplitView {
                ScrollView {
                  Text(diff.isEmpty ? L("Aucune modification", "No changes") : diff).font(
                    .system(size: 11, design: .monospaced)
                  ).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
                HSplitView {
                  TextEditor(text: .constant(original)).font(.system(size: 11, design: .monospaced))
                  TextEditor(text: $code).font(.system(size: 11, design: .monospaced)).disabled(
                    revision != versions.first?.id)
                }
              }
            } else {
              List {
                ForEach(versions) { version in
                  Button {
                    revision = version.id
                  } label: {
                    HStack {
                      Text(version.title)
                      Spacer()
                      Text(version.date, style: .time).foregroundStyle(StudioTheme.quiet)
                    }
                  }.buttonStyle(.plain)
                }
                if projectMode {
                  ForEach(studio.currentConversation?.messages.flatMap(\.toolActivities) ?? []) {
                    action in
                    VStack(alignment: .leading) {
                      Text(action.toolName).font(.headline)
                      Text(action.result ?? L("En cours", "Running")).font(
                        .system(size: 10, design: .monospaced)
                      ).lineLimit(8)
                    }
                  }
                }
              }
            }
            if let requestRevision,
              selected.lowercased().hasSuffix(".html") || selected.lowercased().hasSuffix(".htm")
            {
              HStack {
                TextField(
                  L("Demander une modification à l’IA…", "Ask AI for a change…"), text: $instruction
                )
                Button(L("Modifier", "Revise"), systemImage: "arrow.up") {
                  let request = ArtifactRevisionRequest(
                    workspace: store.workspace, file: selected, source: code,
                    instruction: instruction)
                  dismiss()
                  requestRevision(request)
                }.disabled(
                  instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || studio.isGenerating)
              }
            }
          }.padding(12).frame(minWidth: 450)
        }
        if state.skippedFiles > 0 {
          Text(
            L(
              "\(state.skippedFiles) éléments ignorés : dépendances, métadonnées, fichiers privés, liens ou fichiers trop grands.",
              "\(state.skippedFiles) items skipped: dependencies, metadata, private files, links or oversized files."
            )
          ).font(.caption).foregroundStyle(StudioTheme.quiet).padding(8)
        }
      } else {
        ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
      }
      if let error {
        Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled).padding(10)
      }
    }
    .frame(minWidth: 880, idealWidth: 1100, minHeight: 600, idealHeight: 740)
    .background(StudioTheme.canvas).foregroundStyle(StudioTheme.ink)
    .disabled(busy || studio.isGenerating || studio.agentStudioBusy)
    .task {
      do {
        if let workspace {
          store = AgentProjectStore(workspace: workspace)
        } else if let source {
          store = try await ArtifactLibrary.shared.open(source, base: studio.artifactBaseURL)
        }
        if let file, let workspace {
          selected = String(file.path.dropFirst(workspace.path.count + 1))
        }
        if let store { try await refresh(store) }
      } catch { self.error = error.localizedDescription }
    }
    .task(id: revision + "/" + selected + "/" + compare) { await loadSelection() }
  }

  private func refresh(_ store: AgentProjectStore) async throws {
    let result = try await Task.detached { () -> (AgentProjectState, [AgentCheckpoint], [String]) in
      try FileManager.default.createDirectory(
        at: store.workspace, withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700])
      _ = try store.checkpoint("Workspace review", force: false)
      return (try store.state(), try store.versions(), try store.files().paths)
    }.value
    state = result.0
    versions = result.1
    paths = result.2
    revision = versions.first?.id ?? ""
    compare = state.baseline ?? versions.dropFirst().first?.id ?? revision
    if !paths.contains(selected) {
      selected = paths.first(where: { $0.hasSuffix(".html") }) ?? paths.first ?? ""
    }
    await loadSelection()
  }

  private func loadSelection() async {
    guard let store, !revision.isEmpty, !selected.isEmpty else { return }
    let revision = revision
    let selected = selected
    let compare = compare
    do {
      let result = try await Task.detached { () -> (String, String, String, URL) in
        let data = try store.contents(selected, revision: revision)
        let before = (try? store.contents(selected, revision: compare)) ?? Data()
        return (
          String(data: data, encoding: .utf8) ?? "[Binary file]",
          String(data: before, encoding: .utf8) ?? "[Binary file]",
          try store.diff(from: compare, to: revision), try store.materialize(revision)
        )
      }.value
      guard !Task.isCancelled else { return }
      code = result.0
      original = result.1
      diff = result.2
      webView = nil
      if ["html", "htm"].contains(URL(fileURLWithPath: selected).pathExtension.lowercased()) {
        webView = try await HTMLPreview.makeWebView(
          source: code, workspace: result.3, file: result.3.appendingPathComponent(selected))
      }
    } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
  }

  private func perform(_ action: @escaping @Sendable () throws -> Void) {
    guard let store, !busy else { return }
    busy = true
    error = nil
    Task {
      defer { busy = false }
      do {
        try await Task.detached { try action() }.value
        try await refresh(store)
        if projectMode { studio.refreshAgentPermissions() }
      } catch { self.error = error.localizedDescription }
    }
  }

  private func chooseProject(_ store: AgentProjectStore) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    perform { try store.importProject(url) }
  }

  private func exportZIP(_ store: AgentProjectStore) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.zip]
    panel.nameFieldStringValue = "MLXL3-project.zip"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    let revision = revision
    perform { try store.exportZIP(url, revision: revision) }
  }
}
