import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Full-window workspace for the isolated Agent project.
struct AgentStudioView: View {
  let workspace: URL
  var file: URL? = nil
  var onClose: (() -> Void)? = nil
  @Environment(\.dismiss) private var dismiss
  @EnvironmentObject private var studio: StudioModel
  @StateObject private var model: AgentStudioModel
  @State private var permissions = false
  @State private var pendingNavigation: (() -> Void)?
  @State private var confirmEdits = false
  @State private var confirmRestore = false
  @State private var confirmApply = false
  @State private var previewRoot: URL?

  init(
    workspace: URL, file: URL? = nil, onClose: (() -> Void)? = nil,
    model: AgentStudioModel? = nil
  ) {
    self.workspace = workspace
    self.file = file
    self.onClose = onClose
    _model = StateObject(wrappedValue: model ?? AgentStudioModel())
  }

  private var mutating: Bool { model.busy || studio.isGenerating || studio.agentStudioBusy }

  var body: some View {
    VStack(spacing: 0) {
      header
      Rectangle().fill(StudioTheme.edge).frame(height: 1)
      if let error = model.error {
        HStack(spacing: 12) {
          Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
          Text(error).font(.system(size: 11)).textSelection(.enabled).frame(
            maxWidth: .infinity, alignment: .leading)
          Button {
            model.error = nil
          } label: {
            Image(systemName: "xmark").padding(8)
          }
          .buttonStyle(StudioControlStyle()).accessibilityLabel(
            L("Masquer l’erreur", "Dismiss error"))
        }.padding(.horizontal, 20).padding(.vertical, 10).background(Color.orange.opacity(0.05))
          .fixedSize(horizontal: false, vertical: true)
      }
      HSplitView {
        sidebar.frame(minWidth: 180, idealWidth: 224, maxWidth: 280)
        document.frame(minWidth: 380, maxWidth: .infinity, maxHeight: .infinity)
      }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    .foregroundStyle(StudioTheme.ink).background(StudioTheme.canvas)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    .tint(StudioTheme.accent)
    .task {
      await model.open(workspace: workspace, file: file)
    }
    .onExitCommand { navigate { close() } }
    .confirmationDialog(
      L("Enregistrer les modifications ?", "Save your changes?"), isPresented: $confirmEdits,
      titleVisibility: .visible
    ) {
      Button(L("Enregistrer et continuer", "Save and continue")) {
        Task {
          if await model.save() {
            let action = pendingNavigation
            pendingNavigation = nil
            action?()
          }
        }
      }
      Button(L("Ignorer les modifications", "Discard changes"), role: .destructive) {
        model.discardEdits()
        let action = pendingNavigation
        pendingNavigation = nil
        action?()
      }
      Button(L("Annuler", "Cancel"), role: .cancel) { pendingNavigation = nil }
    }
    .confirmationDialog(
      L("Restaurer tous les fichiers de cette version ?", "Restore all files from this version?"),
      isPresented: $confirmRestore, titleVisibility: .visible
    ) {
      Button(L("Restaurer la copie de travail", "Restore workspace"), role: .destructive) {
        Task { await model.restore() }
      }
      Button(L("Annuler", "Cancel"), role: .cancel) {}
    } message: {
      Text(
        L(
          "Un checkpoint conserve la version actuelle. Les fichiers originaux du projet restent inchangés.",
          "A checkpoint preserves the current version. The original project files stay unchanged."))
    }
    .confirmationDialog(
      L(
        "Appliquer cette version au projet original ?",
        "Apply this version to the original project?"), isPresented: $confirmApply,
      titleVisibility: .visible
    ) {
      Button(L("Appliquer les changements revus", "Apply reviewed changes")) {
        Task { await model.apply() }
      }
      Button(L("Annuler", "Cancel"), role: .cancel) {}
    } message: {
      Text(
        L(
          "Les ajouts, modifications et suppressions du diff complet seront appliqués. Tout conflit ou changement depuis la revue sera refusé.",
          "The complete diff's additions, edits and deletions will be applied. Conflicts or changes since review are refused."
        ))
    }
  }

  private var header: some View {
    HStack(spacing: 16) {
      Button {
        navigate { close() }
      } label: {
        Image(systemName: "arrow.left").frame(width: 32, height: 32)
      }
      .buttonStyle(StudioControlStyle()).help(L("Retour au chat", "Back to chat"))
      .accessibilityLabel(L("Retour au chat", "Back to chat"))
      MonogramMark(size: 25)
      VStack(alignment: .leading, spacing: 3) {
        Text("Agent Studio").font(
          .system(size: 21, design: .serif))
        Text(model.title)
          .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet).lineLimit(1)
      }
      Spacer(minLength: 12)
      if model.busy || studio.agentStudioBusy { ProgressView().controlSize(.small) }
      Button {
        permissions.toggle()
      } label: {
        Label("Sandbox", systemImage: "lock.shield").padding(.horizontal, 10).frame(height: 32)
      }
      .buttonStyle(StudioControlStyle()).popover(isPresented: $permissions) { permissionControls }
      if model.state.source != nil {
        Button {
          navigate {
            if model.canApply {
              confirmApply = true
            } else {
              Task { await model.reviewProject() }
            }
          }
        } label: {
          Label(
            model.canApply
              ? L("Appliquer au projet", "Apply to project")
              : L("Examiner les changements", "Review changes"),
            systemImage: model.canApply ? "checkmark" : "plus.forwardslash.minus"
          )
          .padding(.horizontal, 12).frame(height: 32)
        }.buttonStyle(StudioControlStyle(emphasized: true)).disabled(
          mutating || model.versions.isEmpty)
      }
      Menu {
        Button(
          L("Ouvrir un projet dans une nouvelle tâche…", "Open project in a new task…"),
          systemImage: "folder.badge.plus"
        ) { navigate { chooseProject() } }
        Divider()
        Button(L("Créer un checkpoint", "Create checkpoint"), systemImage: "flag") {
          Task { await model.checkpoint() }
        }.disabled(mutating || model.dirty)
        Button(
          L("Exporter cette version en ZIP…", "Export this version as ZIP…"),
          systemImage: "arrow.down.doc"
        ) { exportZIP() }.disabled(mutating || model.dirty || model.revision.isEmpty)
        Button(L("Afficher le dossier de travail", "Reveal workspace"), systemImage: "folder") {
          if let store = model.store { NSWorkspace.shared.open(store.workspace) }
        }.disabled(model.store == nil)
        Divider()
        Button(L("Actualiser", "Refresh"), systemImage: "arrow.clockwise") {
          navigate { Task { await model.reload() } }
        }.disabled(mutating)
      } label: {
        Image(systemName: "ellipsis").frame(width: 32, height: 32)
      }
      .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().help(
        L("Actions du projet", "Project actions"))
    }
    .padding(.horizontal, 20).frame(height: 74).fixedSize(horizontal: false, vertical: true)
    .background(StudioTheme.canvas)
  }

  private var sidebar: some View {
    VStack(alignment: .leading, spacing: 0) {
      HStack {
        Text(L("EXPLORATEUR", "EXPLORER")).font(.system(size: 9, weight: .medium)).tracking(1.4)
        Spacer()
        Text("\(model.paths.count)").font(.system(size: 10, design: .monospaced))
      }.foregroundStyle(StudioTheme.quiet).padding(.horizontal, 18).frame(height: 47).fixedSize(
        horizontal: false, vertical: true)
      ScrollView {
        VStack(alignment: .leading, spacing: 3) {
          OutlineGroup(StudioFileNode.tree(model.paths), children: \.children) { node in
            if node.children != nil {
              Label(node.name, systemImage: "folder").font(.system(size: 11)).foregroundStyle(
                StudioTheme.quiet
              ).padding(.vertical, 7)
            } else {
              Button {
                navigate { Task { await model.select(node.id) } }
              } label: {
                HStack(spacing: 8) {
                  Image(systemName: node.icon).font(.system(size: 10)).foregroundStyle(
                    StudioTheme.quiet)
                  Text(node.name).font(.system(size: 11, design: .monospaced)).lineLimit(1)
                    .truncationMode(.middle)
                  Spacer(minLength: 2)
                  if node.id == model.selected && model.dirty {
                    Circle().fill(StudioTheme.ink).frame(width: 5, height: 5)
                  }
                }.padding(.horizontal, 10).frame(height: 31)
                  .background(
                    node.id == model.selected ? Color.white.opacity(0.075) : .clear,
                    in: RoundedRectangle(cornerRadius: 6))
              }.buttonStyle(StudioControlStyle()).accessibilityIdentifier("studio.file.\(node.id)")
            }
          }
          if model.paths.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
              Image(systemName: "folder").font(.system(size: 24)).foregroundStyle(StudioTheme.quiet)
              Text(L("Aucun fichier pour le moment", "No files yet")).font(.system(size: 12))
              Text(
                L(
                  "Les fichiers créés par l’agent apparaîtront ici.",
                  "Files created by the Agent will appear here.")
              ).font(.system(size: 11)).foregroundStyle(StudioTheme.quiet)
              Button(L("Ouvrir un projet…", "Open project…")) { chooseProject() }.buttonStyle(
                GlassPillButtonStyle()
              ).disabled(mutating)
            }.padding(12)
          }
        }.padding(.horizontal, 10).padding(.bottom, 20)
      }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
      VStack(alignment: .leading, spacing: 6) {
        Label(
          L("Copie isolée · hors ligne", "Isolated copy · offline"), systemImage: "lock.shield")
        if model.state.skippedFiles > 0 {
          Text(
            L(
              "\(model.state.skippedFiles) éléments exclus à l’import",
              "\(model.state.skippedFiles) items excluded on import"))
        }
      }.font(.system(size: 9)).foregroundStyle(StudioTheme.quiet).padding(16).fixedSize(
        horizontal: false, vertical: true)
    }.frame(maxHeight: .infinity, alignment: .top).background(StudioTheme.sidebar)
  }

  private var document: some View {
    VStack(spacing: 0) {
      HStack(spacing: 22) {
        ForEach(
          StudioTab.allCases.filter {
            $0 != .preview || model.isHTML && model.fileExists && !model.binary
          }
        ) { tab in
          Button {
            model.tab = tab
          } label: {
            VStack(spacing: 0) {
              Label(tab.title, systemImage: tab.icon).font(
                .system(size: 11, weight: model.tab == tab ? .medium : .regular)
              ).frame(height: 44)
              Rectangle().fill(model.tab == tab ? StudioTheme.ink : .clear).frame(height: 2)
            }.foregroundStyle(model.tab == tab ? StudioTheme.ink : StudioTheme.quiet)
              .fixedSize(horizontal: true, vertical: false)
          }.buttonStyle(.plain).accessibilityIdentifier("studio.tab.\(tab.rawValue)")
        }
        Spacer(minLength: 8)
        if model.tab == .code {
          Button {
            Task { _ = await model.save() }
          } label: {
            Label(L("Enregistrer", "Save"), systemImage: "checkmark").font(.system(size: 11))
              .padding(.horizontal, 11).frame(height: 28)
          }.buttonStyle(StudioControlStyle(emphasized: model.dirty)).disabled(
            mutating || !model.dirty
          )
          .keyboardShortcut("s", modifiers: .command)
        }
      }.padding(.horizontal, 22).fixedSize(horizontal: false, vertical: true)
      Rectangle().fill(StudioTheme.edge).frame(height: 1)
      if model.tab != .history { documentBar }
      ZStack {
        StudioTheme.canvas
        if model.tab == .history {
          history
        } else if model.selected.isEmpty {
          emptyState(
            "doc.text", L("Votre espace de travail", "Your workspace"),
            L(
              "Sélectionnez un fichier ou ouvrez un projet pour commencer.",
              "Select a file or open a project to get started."))
        } else if model.tab == .preview {
          preview
        } else if model.tab == .diff {
          if model.diff.isEmpty {
            emptyState(
              "checkmark.circle", L("Aucune différence", "No differences"),
              L("Les versions comparées sont identiques.", "The compared versions are identical."))
          } else {
            StudioCodeEditor(
              text: .constant(model.diff), language: "diff", editable: false,
              documentID:
                "diff/\(model.selected)/\(model.revision)/\(model.compare)/\(model.allChanges)",
              diff: true)
          }
        } else if model.binary {
          emptyState(
            "doc", L("Fichier binaire", "Binary file"),
            L(
              "Ce fichier est conservé dans le projet, mais ne peut pas être modifié comme du texte.",
              "This file stays in the project but cannot be edited as text."))
        } else if !model.fileExists && !model.loading {
          emptyState(
            "doc.badge.minus",
            L("Fichier absent de cette version", "File absent from this version"),
            L(
              "Choisissez une version précédente pour le lire.",
              "Choose an earlier version to read it."))
        } else {
          StudioCodeEditor(
            text: $model.code, language: URL(fileURLWithPath: model.selected).pathExtension,
            editable: model.editable && !mutating, documentID: "\(model.selected)/\(model.revision)"
          )
        }
        if model.loading {
          ProgressView().padding(14).background(
            StudioTheme.sidebar, in: RoundedRectangle(cornerRadius: 10))
        }
      }.frame(maxWidth: .infinity, maxHeight: .infinity)
      HStack {
        Text(model.binary ? L("Binaire", "Binary") : "UTF-8")
        if !model.binary && model.fileExists {
          Text("·")
          Text(
            L(
              "\(model.code.components(separatedBy: "\n").count) lignes",
              "\(model.code.components(separatedBy: "\n").count) lines"))
        }
        Spacer()
        Text(
          model.dirty
            ? L("Modifications non enregistrées", "Unsaved changes")
            : model.isCurrent
              ? L("Copie de travail", "Working copy")
              : L("Version historique · lecture seule", "Historical version · read only"))
      }.font(.system(size: 9)).foregroundStyle(StudioTheme.quiet).padding(.horizontal, 20).frame(
        height: 28
      ).fixedSize(horizontal: false, vertical: true)
        .background(StudioTheme.sidebar)
    }
  }

  private var documentBar: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 10) {
        Text(
          model.tab == .diff && model.allChanges
            ? L("Tous les changements du projet", "All project changes") : model.selected
        )
        .font(.system(size: 11, design: .monospaced)).lineLimit(1).truncationMode(.middle)
        Spacer()
        Menu {
          ForEach(model.versions) { value in
            Button("\(value.title) · \(value.id.prefix(7))") {
              navigate { Task { await model.selectRevision(value.id) } }
            }
          }
        } label: {
          Label(
            model.isCurrent ? L("Actuelle", "Current") : String(model.revision.prefix(7)),
            systemImage: "clock.arrow.circlepath"
          )
          .font(.system(size: 10)).foregroundStyle(StudioTheme.secondary)
        }.menuStyle(.borderlessButton).fixedSize().help(L("Choisir une version", "Choose version"))
        if !model.isCurrent, !model.revision.isEmpty {
          Button(L("Restaurer…", "Restore…")) { navigate { confirmRestore = true } }.font(
            .system(size: 10)
          ).buttonStyle(StudioControlStyle()).disabled(mutating)
        }
      }
      if model.tab == .diff {
        HStack(spacing: 12) {
          Text(L("Comparer avec", "Compare with")).foregroundStyle(StudioTheme.quiet)
          Menu {
            ForEach(model.versions) { value in
              Button("\(value.title) · \(value.id.prefix(7))") {
                navigate { Task { await model.compareWith(value.id) } }
              }
            }
          } label: {
            Text(model.versions.first(where: { $0.id == model.compare })?.title ?? "—")
          }
          .menuStyle(.borderlessButton).fixedSize()
          Spacer()
          Button(
            model.allChanges
              ? L("Fichier sélectionné", "Selected file") : L("Tous les fichiers", "All files")
          ) { navigate { Task { await model.showAllChanges(!model.allChanges) } } }.buttonStyle(
            StudioControlStyle())
        }.font(.system(size: 10))
      }
    }.padding(.horizontal, 22).padding(.vertical, 13).fixedSize(horizontal: false, vertical: true)
      .background(StudioTheme.panel)
  }

  private var preview: some View {
    Group {
      if let root = previewRoot {
        InlineHTMLPreview(
          source: model.savedCode, workspace: root,
          file: root.appendingPathComponent(model.selected))
      } else {
        ProgressView().controlSize(.small)
      }
    }.task(id: "\(model.selected)/\(model.revision)/\(model.loading)") {
      previewRoot = nil
      guard model.isHTML, model.fileExists, !model.binary, !model.loading, let store = model.store
      else { return }
      let revision = model.revision
      do {
        let root = try await Task.detached { try store.materialize(revision) }.value
        guard !Task.isCancelled else { return }
        previewRoot = root
      } catch { if !Task.isCancelled { model.error = error.localizedDescription } }
    }
  }

  private var history: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 24) {
        Text(L("Versions du projet", "Project versions")).font(.system(size: 25, design: .serif))
        ForEach(model.versions) { value in
          Button {
            navigate {
              Task {
                if await model.selectRevision(value.id) {
                  model.tab = .code
                }
              }
            }
          } label: {
            HStack(alignment: .top, spacing: 16) {
              Image(
                systemName: value.id == model.versions.first?.id ? "circle.inset.filled" : "circle"
              ).foregroundStyle(StudioTheme.quiet).padding(.top, 3)
              VStack(alignment: .leading, spacing: 6) {
                Text(value.title).font(.system(size: 12, weight: .medium))
                HStack {
                  Text(value.date, style: .date)
                  Text(value.date, style: .time)
                  Text("· \(value.id.prefix(7))").fontDesign(.monospaced)
                }
                .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
              }
              Spacer()
              Text(
                value.id == model.versions.first?.id
                  ? L("Actuelle", "Current") : L("Ouvrir", "Open")
              ).font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
            }.padding(16).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
          }.buttonStyle(StudioControlStyle()).disabled(model.busy)
        }
        if let actions = studio.currentConversation?.messages.flatMap(\.toolActivities),
          !actions.isEmpty
        {
          Text(L("Actions de l’agent", "Agent actions")).font(.system(size: 22, design: .serif))
            .padding(.top, 8)
          ForEach(actions) { action in
            VStack(alignment: .leading, spacing: 7) {
              Label(
                action.toolName,
                systemImage: action.state == .failed ? "exclamationmark.circle" : "terminal"
              ).font(.system(size: 11, weight: .medium))
              Text(action.result ?? L("En cours", "Running")).font(
                .system(size: 10, design: .monospaced)
              ).foregroundStyle(StudioTheme.quiet).lineLimit(8).textSelection(.enabled)
            }.padding(16).frame(maxWidth: .infinity, alignment: .leading).background(
              StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
          }
        }
      }.padding(30).frame(maxWidth: 920, alignment: .leading).frame(
        maxWidth: .infinity, alignment: .topLeading)
    }
  }

  private var permissionControls: some View {
    VStack(alignment: .leading, spacing: 14) {
      Label(L("Sandbox locale", "Local sandbox"), systemImage: "lock.shield").font(
        .system(size: 14, weight: .medium))
      Text(L("Hors ligne, dans la copie isolée.", "Offline, inside the isolated copy.")).font(
        .system(size: 11)
      ).foregroundStyle(StudioTheme.quiet)
      permission(L("Outil de lecture", "Read tool"), key: \.read)
      permission(L("Écriture des fichiers", "File writes"), key: \.write)
      permission(L("Commandes", "Commands"), key: \.run)
      Stepper(
        L(
          "Durée : \(model.state.permissions.commandSeconds) s",
          "Timeout: \(model.state.permissions.commandSeconds) s"),
        value: Binding(
          get: { model.state.permissions.commandSeconds },
          set: { seconds in
            var value = model.state.permissions
            value.commandSeconds = seconds
            Task {
              await model.setPermissions(value)
              studio.refreshAgentPermissions()
            }
          }), in: 1...30
      ).font(.system(size: 11))
      Text(
        L(
          "Les commandes peuvent lire la copie. Désactiver l’écriture bloque aussi les écritures des commandes.",
          "Commands can read the copy. Disabling writes also blocks command writes.")
      ).font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
    }.padding(20).frame(width: 280).tint(StudioTheme.accent).disabled(mutating)
  }

  private func permission(_ title: String, key: WritableKeyPath<AgentPermissions, Bool>)
    -> some View
  {
    Toggle(
      title,
      isOn: Binding(
        get: { model.state.permissions[keyPath: key] },
        set: { enabled in
          var value = model.state.permissions
          value[keyPath: key] = enabled
          Task {
            await model.setPermissions(value)
            studio.refreshAgentPermissions()
          }
        })
    ).toggleStyle(.switch).font(.system(size: 11))
  }

  private func emptyState(_ icon: String, _ title: String, _ description: String) -> some View {
    VStack(spacing: 13) {
      Image(systemName: icon).font(.system(size: 25, weight: .light)).foregroundStyle(
        StudioTheme.quiet)
      Text(title).font(.system(size: 22, design: .serif))
      Text(description).font(.system(size: 11)).foregroundStyle(StudioTheme.quiet)
        .multilineTextAlignment(.center).frame(maxWidth: 340)
    }.padding(30).frame(maxWidth: .infinity, maxHeight: .infinity)
  }

  private func navigate(_ action: @escaping () -> Void) {
    if model.dirty {
      pendingNavigation = action
      confirmEdits = true
    } else {
      action()
    }
  }
  private func close() { if let onClose { onClose() } else { dismiss() } }
  private func chooseProject() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.message = L(
      "Le projet s’ouvre dans une nouvelle tâche isolée. La tâche actuelle est conservée.",
      "The project opens in a new isolated task. Your current task is kept.")
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await studio.openAgentProject(url) }
  }
  private func exportZIP() {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.zip]
    panel.nameFieldStringValue = "MLXL3-project.zip"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task { await model.exportZIP(to: url) }
  }

}

private struct StudioFileNode: Identifiable {
  let id: String
  let name: String
  let children: [StudioFileNode]?
  var icon: String {
    switch URL(fileURLWithPath: id).pathExtension.lowercased() {
    case "html", "htm": "globe"
    case "js", "ts", "py", "swift", "rs", "css", "json": "chevron.left.forwardslash.chevron.right"
    default: "doc.text"
    }
  }
  static func tree(_ paths: [String], prefix: String = "") -> [StudioFileNode] {
    let groups = Dictionary(grouping: paths) {
      String($0.dropFirst(prefix.count).split(separator: "/").first ?? "")
    }
    return groups.keys.sorted().map { name in
      let path = prefix + name
      return StudioFileNode(
        id: path, name: name,
        children: groups[name]!.contains(path) ? nil : tree(groups[name]!, prefix: path + "/"))
    }
  }
}
