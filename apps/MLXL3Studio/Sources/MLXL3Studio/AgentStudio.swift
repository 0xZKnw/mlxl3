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
  @State private var fullscreen = false
  /// Host list being edited; committed on Return or when the field loses focus.
  @State private var hostsDraft = ""
  /// Folders the user closed; every folder starts open so the selected file is visible.
  @State private var collapsed = Set<String>()

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
    HSplitView {
      // The split view keeps its panes below the title bar; the page sidebars reach the top.
      sidebar.frame(minWidth: 180, idealWidth: studio.sidebarWidth.points, maxWidth: studio.sidebarWidth.points)
        .ignoresSafeArea(edges: .top)
      VStack(spacing: 0) {
        header
        if let error = model.error {
          HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 12, weight: .medium))
              .foregroundStyle(StudioTheme.warning).accessibilityHidden(true)
            Text(error).font(.system(size: 12.5)).textSelection(.enabled).frame(
              maxWidth: .infinity, alignment: .leading)
            Button {
              model.error = nil
            } label: {
              Image(systemName: "xmark").font(.system(size: 10, weight: .semibold)).frame(width: 22, height: 22)
            }
            .buttonStyle(StudioRowStyle()).accessibilityLabel(
              L("Masquer l’erreur", "Dismiss error"))
          }.padding(.leading, 12).padding(.trailing, 6).padding(.vertical, 7)
            .background(StudioTheme.warning.opacity(0.1), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .padding(.horizontal, 24).padding(.bottom, 12)
            .fixedSize(horizontal: false, vertical: true)
        }

        document
          .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
          .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false) }
          .padding(.horizontal, 24).padding(.bottom, 20)
      }.frame(minWidth: 380, maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea(edges: .top)
    }
    .foregroundStyle(StudioTheme.ink)
    .background { StudioAurora(animated: true).background(StudioTheme.canvas) }
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    .tint(studio.accentColor)
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

  /// One compact bar: the task, where its files stand, and the project actions. The editor
  /// keeps the height a display title used to take.
  private var header: some View {
    HStack(alignment: .center, spacing: 10) {
      VStack(alignment: .leading, spacing: 3) {
        Text(model.title).font(.system(size: 16, weight: .semibold)).lineLimit(1).truncationMode(.middle)
          .accessibilityAddTraits(.isHeader)
        Text(subtitle).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary).lineLimit(1)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
      if model.busy || studio.agentStudioBusy { ProgressView().controlSize(.small) }
      Button {
        permissions.toggle()
      } label: {
        Label("Sandbox", systemImage: "lock.shield")
      }
      .buttonStyle(.studio).popover(isPresented: $permissions, arrowEdge: .bottom) { permissionControls }
      .help(L("Ce que l’agent peut lire, écrire et lancer", "What the Agent may read, write and run"))
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
        }.buttonStyle(.studio(model.canApply ? .primary : .secondary)).disabled(
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
        Image(systemName: "ellipsis").font(.system(size: 13, weight: .medium))
          .frame(width: 30, height: 30).contentShape(Rectangle())
      }
      .menuStyle(.button).buttonStyle(StudioRowStyle()).menuIndicator(.hidden).fixedSize().help(
        L("Actions du projet", "Project actions"))
      .accessibilityLabel(L("Actions du projet", "Project actions"))
    }
    .padding(.horizontal, 24).padding(.top, 22).padding(.bottom, 16)
    .fixedSize(horizontal: false, vertical: true)
  }

  private var subtitle: String {
    let files = model.paths.count == 1 ? L("1 fichier", "1 file")
      : L("\(model.paths.count) fichiers", "\(model.paths.count) files")
    let state = model.dirty
      ? L("Modifications non enregistrées", "Unsaved changes")
      : model.isCurrent
        ? L("Copie de travail", "Working copy")
        : L("Version historique · lecture seule", "Historical version · read only")
    return "Agent Studio · " + files + " · " + state
  }

  private var sidebar: some View {
    StudioPageSidebar(title: "Agent Studio", fixedWidth: false, onBack: { navigate { close() } }) {
      HStack {
        Text(L("Fichiers", "Files")).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
        Spacer()
        Text("\(model.paths.count)").font(.system(size: 11)).monospacedDigit().foregroundStyle(StudioTheme.quiet)
      }.padding(.horizontal, 18).frame(height: 26).fixedSize(
        horizontal: false, vertical: true)
      ScrollView {
        VStack(alignment: .leading, spacing: 1) {
          ForEach(StudioFileNode.rows(StudioFileNode.tree(model.paths), collapsed: collapsed), id: \.node.id) { row in
            let node = row.node
            let selected = node.children == nil && node.id == model.selected
            Button {
              if node.children != nil {
                if collapsed.contains(node.id) { collapsed.remove(node.id) } else { collapsed.insert(node.id) }
              } else {
                navigate { Task { await model.select(node.id) } }
              }
            } label: {
              HStack(spacing: 6) {
                Image(systemName: "chevron.right").font(.system(size: 8, weight: .bold))
                  .foregroundStyle(StudioTheme.quiet)
                  .rotationEffect(.degrees(collapsed.contains(node.id) ? 0 : 90))
                  .frame(width: 10)
                  .opacity(node.children == nil ? 0 : 1)
                  .accessibilityHidden(true)
                Image(systemName: node.children == nil ? node.icon : "folder").font(.system(size: 11.5)).foregroundStyle(
                  selected ? StudioTheme.ink : StudioTheme.quiet).frame(width: 16)
                Text(node.name).font(.system(size: 12.5))
                  .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.secondary)
                  .lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 2)
                if selected && model.dirty {
                  Circle().fill(StudioTheme.ink).frame(width: 6, height: 6)
                    .help(L("Modifications non enregistrées", "Unsaved changes"))
                    .accessibilityLabel(L("Modifications non enregistrées", "Unsaved changes"))
                }
              }.padding(.leading, 6 + CGFloat(row.depth) * 14).padding(.trailing, 8).frame(height: 26)
                .contentShape(Rectangle())
            }.buttonStyle(StudioRowStyle(selected: selected, radius: 6))
              .accessibilityIdentifier(node.children == nil ? "studio.file.\(node.id)" : "studio.folder.\(node.id)")
              .accessibilityLabel(node.name)
              .accessibilityValue(node.children == nil ? "" : collapsed.contains(node.id) ? L("Fermé", "Collapsed") : L("Ouvert", "Expanded"))
              .accessibilityAddTraits(selected ? .isSelected : [])
          }
          if model.paths.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
              Text(L("Aucun fichier pour le moment", "No files yet")).font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(StudioTheme.secondary)
              Text(
                L(
                  "Les fichiers créés par l’agent apparaîtront ici.",
                  "Files created by the Agent will appear here.")
              ).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                .fixedSize(horizontal: false, vertical: true)
              Button(L("Ouvrir un projet…", "Open project…")) { chooseProject() }.buttonStyle(
                .studio
              ).disabled(mutating).padding(.top, 4)
            }.padding(.horizontal, 8).padding(.top, 6)
          }
        }.padding(.horizontal, 10).padding(.bottom, 16)
      }.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .scrollIndicators(.hidden)
        .mask(StudioEdgeFade(top: 4, bottom: 16))
    } footer: {
      if model.state.skippedFiles > 0 {
        Text(
          L(
            "\(model.state.skippedFiles) éléments exclus à l’import",
            "\(model.state.skippedFiles) items excluded on import")
        ).font(.system(size: 11)).foregroundStyle(StudioTheme.quiet).padding(.horizontal, 18).padding(.bottom, 6)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
  }

  private var document: some View {
    VStack(spacing: 0) {
      HStack(spacing: 4) {
        ForEach(
          StudioTab.allCases.filter {
            $0 != .preview || model.isHTML && model.fileExists && !model.binary
          }
        ) { tab in
          Button {
            model.tab = tab
          } label: {
            Label(tab.title, systemImage: tab.icon)
              .font(.system(size: 12.5, weight: model.tab == tab ? .medium : .regular))
              .foregroundStyle(model.tab == tab ? StudioTheme.ink : StudioTheme.secondary)
              .padding(.horizontal, 10).frame(height: 28)
              .fixedSize(horizontal: true, vertical: false)
              .contentShape(Rectangle())
          }.buttonStyle(StudioRowStyle(selected: model.tab == tab))
            .accessibilityIdentifier("studio.tab.\(tab.rawValue)")
            .accessibilityAddTraits(model.tab == tab ? .isSelected : [])
        }
        Spacer(minLength: 8)
        if model.tab == .preview, let root = previewRoot {
          Button {
            let state = $fullscreen
            studio.presentCodeFullscreen(
              language: URL(fileURLWithPath: model.selected).pathExtension, source: model.savedCode,
              canPreview: true, showingPreview: true, workspace: root,
              file: root.appendingPathComponent(model.selected),
              onClose: { state.wrappedValue = false })
            state.wrappedValue = true
          } label: {
            Label(L("Plein écran", "Full screen"), systemImage: "arrow.up.left.and.arrow.down.right")
          }
          .buttonStyle(.studio(.secondary, height: 26))
          .accessibilityIdentifier("studio.preview.fullscreen")
          .help(L("Afficher l’aperçu en plein écran", "Show the preview full screen"))
        }
        if model.tab == .code {
          Button {
            Task { _ = await model.save() }
          } label: {
            Label(L("Enregistrer", "Save"), systemImage: "checkmark")
          }.buttonStyle(.studio(model.dirty ? .primary : .secondary, height: 26)).disabled(
            mutating || !model.dirty
          )
          .keyboardShortcut("s", modifiers: .command)
          .help(L("Enregistrer · ⌘S", "Save · ⌘S"))
        }
      }.padding(.horizontal, 10).frame(height: 44).fixedSize(horizontal: false, vertical: true)
      Rectangle().fill(StudioTheme.edge).frame(height: 0.5)
      if model.tab != .history { documentBar }
      ZStack {
        if model.tab == .history {
          history
        } else if model.selected.isEmpty {
          emptyState(
            "doc.text", L("Fichiers du projet", "Project files"),
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
      HStack(spacing: 6) {
        Text(model.binary ? L("Binaire", "Binary") : "UTF-8")
        if !model.binary && model.fileExists {
          Text("·")
          Text(
            L(
              "\(model.code.components(separatedBy: "\n").count) lignes",
              "\(model.code.components(separatedBy: "\n").count) lines"))
        }
        Spacer()
        if model.dirty {
          Circle().fill(StudioTheme.ink).frame(width: 5, height: 5).accessibilityHidden(true)
        }
        Text(
          model.dirty
            ? L("Modifications non enregistrées", "Unsaved changes")
            : model.isCurrent
              ? L("Copie de travail", "Working copy")
              : L("Version historique · lecture seule", "Historical version · read only"))
          .foregroundStyle(model.dirty ? StudioTheme.ink : StudioTheme.quiet)
      }.font(.system(size: 11)).monospacedDigit().foregroundStyle(StudioTheme.quiet).padding(.horizontal, 16).frame(
        height: 26
      ).fixedSize(horizontal: false, vertical: true)
        .background(StudioTheme.panel)
        .overlay(alignment: .top) { Rectangle().fill(StudioTheme.edge).frame(height: 0.5) }
    }
  }

  private var documentBar: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 10) {
        Image(systemName: model.tab == .diff && model.allChanges ? "square.stack.3d.up" : "doc.text")
          .font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet).accessibilityHidden(true)
        Text(
          model.tab == .diff && model.allChanges
            ? L("Tous les changements du projet", "All project changes") : model.selected
        )
        .font(.system(size: 12, design: .monospaced)).lineLimit(1).truncationMode(.middle)
        .textSelection(.enabled)
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
          .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
        }.menuStyle(.borderlessButton).fixedSize().help(L("Choisir une version", "Choose version"))
        if !model.isCurrent, !model.revision.isEmpty {
          Button(L("Restaurer…", "Restore…")) { navigate { confirmRestore = true } }
            .buttonStyle(.studio(.secondary, height: 24)).disabled(mutating)
        }
      }
      if model.tab == .diff {
        HStack(spacing: 10) {
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
          ) { navigate { Task { await model.showAllChanges(!model.allChanges) } } }
            .buttonStyle(.studio(.secondary, height: 24))
        }.font(.system(size: 12))
      }
    }.padding(.horizontal, 16).padding(.vertical, 9).fixedSize(horizontal: false, vertical: true)
      .background(StudioTheme.panel)
      .overlay(alignment: .bottom) { Rectangle().fill(StudioTheme.edge).frame(height: 0.5) }
  }

  private var preview: some View {
    Group {
      if fullscreen {
        // One running page at a time: the full-screen window owns the preview.
        Label(L("Aperçu affiché en plein écran", "Preview shown full screen"),
              systemImage: "arrow.up.left.and.arrow.down.right")
          .font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
      } else if let root = previewRoot {
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
      VStack(alignment: .leading, spacing: 22) {
        VStack(alignment: .leading, spacing: 9) {
          StudioSectionTitle(title: L("Versions du projet", "Project versions"), icon: "clock.arrow.circlepath")
          VStack(spacing: 0) {
            ForEach(Array(model.versions.enumerated()), id: \.element.id) { index, value in
              let current = value.id == model.versions.first?.id
              if index > 0 {
                Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5).accessibilityHidden(true)
              }
              Button {
                navigate {
                  Task {
                    if await model.selectRevision(value.id) {
                      model.tab = .code
                    }
                  }
                }
              } label: {
                HStack(alignment: .center, spacing: 12) {
                  Image(systemName: current ? "circle.inset.filled" : "circle")
                    .font(.system(size: 11)).foregroundStyle(current ? StudioTheme.ink : StudioTheme.quiet)
                    .accessibilityHidden(true)
                  VStack(alignment: .leading, spacing: 3) {
                    Text(value.title).font(.system(size: 13, weight: .medium)).lineLimit(2)
                      .multilineTextAlignment(.leading)
                    HStack(spacing: 5) {
                      Text(value.date, style: .date)
                      Text(value.date, style: .time)
                      Text("· \(value.id.prefix(7))").fontDesign(.monospaced)
                    }
                    .font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet)
                  }
                  Spacer()
                  if current {
                    StudioStatusPill(text: L("Actuelle", "Current"), color: StudioTheme.success)
                  } else {
                    Text(L("Ouvrir", "Open")).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                    Image(systemName: "chevron.right").font(.system(size: 10, weight: .semibold))
                      .foregroundStyle(StudioTheme.quiet)
                  }
                }.padding(.horizontal, 14).padding(.vertical, 11).contentShape(Rectangle())
              }.buttonStyle(StudioRowStyle(radius: 0)).disabled(model.busy)
            }
            if model.versions.isEmpty {
              Text(L("Aucune version enregistrée.", "No saved versions yet."))
                .font(.system(size: 12.5)).foregroundStyle(StudioTheme.quiet)
                .frame(maxWidth: .infinity, alignment: .leading).padding(14)
            }
          }
          .clipShape(RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
          .studioCard(padding: 0)
        }
        if let actions = studio.currentConversation?.messages.flatMap(\.toolActivities),
          !actions.isEmpty
        {
          VStack(alignment: .leading, spacing: 9) {
            StudioSectionTitle(title: L("Actions de l’agent", "Agent actions"), icon: "terminal")
            ForEach(actions) { action in
              let shown = ToolCallPresentation(action)
              VStack(alignment: .leading, spacing: 7) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                  Image(systemName: shown.symbol)
                    .foregroundStyle(action.state == .failed ? StudioTheme.failure : StudioTheme.quiet)
                  Text(shown.title).fontWeight(.medium)
                  if let detail = shown.detail {
                    Text(detail).font(.system(size: 12, design: shown.monospaced ? .monospaced : .default))
                      .foregroundStyle(StudioTheme.quiet).lineLimit(1)
                      .truncationMode(shown.isPath ? .middle : .tail)
                  }
                  Spacer(minLength: 6)
                  if let outcome = shown.outcome {
                    Text(outcome).foregroundStyle(action.state == .failed ? StudioTheme.failure : StudioTheme.quiet)
                  }
                }.font(.system(size: 12.5))
                Text(action.result ?? L("En cours", "Running")).font(
                  .system(size: 11.5, design: .monospaced)
                ).foregroundStyle(StudioTheme.secondary).lineLimit(8).textSelection(.enabled)
              }.studioCard(padding: 14, radius: 10)
            }
          }
        }
      }.padding(24).frame(maxWidth: 920, alignment: .leading).frame(
        maxWidth: .infinity, alignment: .topLeading)
    }
  }

  private var permissionControls: some View {
    VStack(alignment: .leading, spacing: 14) {
      Label(L("Sandbox locale", "Local sandbox"), systemImage: "lock.shield").font(
        .system(size: 14, weight: .semibold))
      VStack(alignment: .leading, spacing: 7) {
        Text(L("Outils du modèle", "Model tools")).font(.system(size: 12, weight: .medium))
          .foregroundStyle(StudioTheme.secondary)
        Picker(
          L("Outils du modèle", "Model tools"),
          selection: Binding(
            get: { model.state.permissions.harness },
            set: { harness in
              var value = model.state.permissions
              value.harness = harness
              Task {
                await model.setPermissions(value)
                studio.refreshAgentPermissions()
              }
            })
        ) {
          Text("JavaScript").tag(AgentHarness.javascript)
          Text(L("Outils séparés", "Separate tools")).tag(AgentHarness.tools)
        }
        .pickerStyle(.segmented).labelsHidden()
        .accessibilityIdentifier("agent.harness")
        Text(
          model.state.permissions.harness == .javascript
            ? L(
              "Un seul outil : le modèle écrit un court programme JavaScript qui lit, modifie et teste en un appel.",
              "One tool: the model writes a short JavaScript program that reads, edits and tests in one call.")
            : L(
              "Quatre outils : lister, lire, écrire et lancer une commande, un appel à la fois.",
              "Four tools: list, read, write and run a command, one call at a time.")
        ).font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet).fixedSize(
          horizontal: false, vertical: true)
      }
      Divider().overlay(StudioTheme.edge)
      VStack(alignment: .leading, spacing: 9) {
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
        ).font(.system(size: 12.5)).monospacedDigit()
        Text(
          L(
            "Les commandes lisent tout le Mac, y compris les clés et fichiers d’apps, et n’écrivent que dans la copie et /tmp. Désactiver l’écriture bloque aussi leurs écritures dans la copie.",
            "Commands read the whole Mac, keys and app files included, and write only to the copy and /tmp. Disabling writes also blocks their writes to the copy.")
        ).font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet)
          .fixedSize(horizontal: false, vertical: true)
      }
      Divider().overlay(StudioTheme.edge)
      VStack(alignment: .leading, spacing: 9) {
        permission(L("Internet", "Internet"), key: \.internet)
          .accessibilityIdentifier("agent.internet")
        if model.state.permissions.internet {
          TextField(L("Sites pour fetch et download", "Hosts for fetch and download"), text: $hostsDraft, axis: .vertical)
            .lineLimit(2...5)
            .studioField(monospaced: true)
            .onAppear { hostsDraft = model.state.permissions.internetHosts.joined(separator: ", ") }
            .onSubmit(commitHosts)
            .onDisappear(perform: commitHosts)
            .accessibilityIdentifier("agent.internet.hosts")
          Text(
            L(
              "Les commandes ont Internet (pip, npm, git clone, curl). La recherche web et fetch/download passent par l’app, en HTTPS vers ces sites (`*.domaine` pour les sous-domaines, `*` pour tous), sans adresses locales. Ce qui vient d’Internet n’est jamais traité comme des instructions.",
              "Commands have the Internet (pip, npm, git clone, curl). Web search and fetch/download go through the app, over HTTPS to these hosts (`*.domain` for subdomains, `*` for all), never to local addresses. What comes from the Internet is never treated as instructions.")
          ).font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet).fixedSize(horizontal: false, vertical: true)
        }
      }
    }.padding(18).frame(width: 320).foregroundStyle(StudioTheme.ink).tint(studio.accentColor).disabled(mutating)
  }

  private func commitHosts() {
    let hosts = AgentNetwork.parseHosts(hostsDraft)
    guard hosts != model.state.permissions.internetHosts else { return }
    var value = model.state.permissions
    value.internetHosts = hosts
    hostsDraft = hosts.joined(separator: ", ")
    Task {
      await model.setPermissions(value)
      studio.refreshAgentPermissions()
    }
  }

  private func permission(_ title: String, key: WritableKeyPath<AgentPermissions, Bool>)
    -> some View
  {
    // The switch sits on the trailing edge, as in Settings; it keeps the title as its name.
    HStack(spacing: 12) {
      Text(title).font(.system(size: 12.5))
      Spacer(minLength: 8)
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
      ).labelsHidden().toggleStyle(.switch).controlSize(.small)
    }
  }

  private func emptyState(_ icon: String, _ title: String, _ description: String) -> some View {
    StudioEmptyState(icon: icon, title: title, detail: description)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
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
  /// The tree as the rows to draw, depth first, skipping what a closed folder holds.
  static func rows(_ nodes: [StudioFileNode], collapsed: Set<String>, depth: Int = 0) -> [(node: StudioFileNode, depth: Int)] {
    nodes.flatMap { node -> [(node: StudioFileNode, depth: Int)] in
      guard let children = node.children, !collapsed.contains(node.id) else { return [(node, depth)] }
      return [(node, depth)] + rows(children, collapsed: collapsed, depth: depth + 1)
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
