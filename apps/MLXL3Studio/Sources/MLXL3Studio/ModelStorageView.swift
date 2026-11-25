import AppKit
import SwiftUI

enum ModelStorageFocus {
    case model(String), download(String)

    func matches(_ item: ModelStorageItem) -> Bool {
        switch self {
        case .model(let name): item.registryName == name
        case .download(let id): item.downloadID == id
        }
    }
}

private struct StorageRemoval {
    let item: ModelStorageItem
    let files: Bool

    var title: String {
        if !files { return L("Retirer de la bibliothèque", "Remove from library") }
        return item.kind == .partialDownload
            ? L("Supprimer les fichiers partiels", "Delete partial files")
            : L("Placer dans la Corbeille", "Move to Trash")
    }

    var message: String {
        if !files {
            return L("Le dossier de \(item.name) restera sur le disque.", "The folder for \(item.name) will stay on disk.")
        }
        let space = ByteCountFormatter.string(fromByteCount: item.reclaimableBytes, countStyle: .file)
        return item.kind == .partialDownload
            ? L("Supprimer les fichiers partiels de \(item.name) ? Environ \(space) pourront être récupérés. Le téléchargement devra recommencer.",
                "Delete the partial files for \(item.name)? About \(space) can be recovered. The download will need to restart.")
            : L("Le dossier de \(item.name) sera placé dans la Corbeille et le modèle retiré de la bibliothèque. Environ \(space) pourront être récupérés une fois la Corbeille vidée.",
                "The folder for \(item.name) will move to Trash and the model will be removed from the library. About \(space) can be recovered after emptying Trash.")
    }
}

struct ModelStorageView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var library: ModelLibrary
    @ObservedObject var comparison: LabComparison
    let initialFocus: ModelStorageFocus?
    @AppStorage("image21ModelDirectory") private var imageModelPath = ""
    @State private var inventory: ModelStorageInventory?
    @State private var selectedID: String?
    @State private var requested: StorageRemoval?
    @State private var refreshID = 0
    @State private var loading = true
    @State private var error: String?
    @State private var appliedInitialFocus = false

    var body: some View {
        GeometryReader { geometry in
            let narrow = geometry.size.width < 610
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        StudioPageHeader(
                            title: L("Stockage des modèles", "Model storage"),
                            detail: L("Ce que les modèles occupent sur le disque, et ce qu’un retrait libérerait.",
                                      "What models take on disk, and what a removal would free.")
                        ) {
                            StudioIconButton(icon: "arrow.clockwise", label: L("Actualiser le stockage", "Refresh storage")) {
                                error = nil
                                refreshID += 1
                            }
                            .disabled(loading)
                        }
                        if let error {
                            StudioNotice(kind: .error, text: error)
                        }
                        if let inventory {
                            summary(inventory)
                            if inventory.items.isEmpty {
                                StudioEmptyState(icon: "internaldrive",
                                                 title: L("Aucun fichier de modèle recensé sur ce Mac.", "No model files found on this Mac."))
                                    .studioCard(padding: 0)
                            } else {
                                ForEach([ModelStorageKind.managedModel, .externalModel, .partialDownload, .sharedCache], id: \.self) { kind in
                                    let items = inventory.items.filter { $0.kind == kind }
                                    if !items.isEmpty { section(kind, items: items, narrow: narrow) }
                                }
                            }
                        } else if loading {
                            HStack(spacing: 10) {
                                ProgressView().controlSize(.small)
                                Text(L("Analyse des dossiers…", "Scanning folders…")).font(.system(size: 12.5))
                                    .foregroundStyle(StudioTheme.secondary)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, narrow ? 24 : 32).padding(.top, 40).padding(.bottom, 36)
                }
                .task(id: refreshID) { await refresh(proxy) }
            }
        }
        .font(.system(size: 13))
        .onChange(of: studio.models) { refreshID += 1 }
        .onChange(of: library.pending) { refreshID += 1 }
        .onChange(of: imageModelPath) { refreshID += 1 }
        .onChange(of: library.downloading) { refreshID += 1 }
        .onChange(of: library.discardingStorageID) { old, new in
            if old != nil, new == nil {
                error = library.error
                refreshID += 1
            }
        }
        .onChange(of: studio.modelInstallState) { _, state in
            if case .failed(let message) = state { error = message }
            if case .succeeded = state { refreshID += 1 }
        }
        .alert(requested?.title ?? "", isPresented: Binding(get: { requested != nil }, set: { if !$0 { requested = nil } }),
               presenting: requested) { removal in
            Button(L("Annuler", "Cancel"), role: .cancel) {}
            Button(removal.title, role: .destructive) { remove(removal) }
        } message: { removal in Text(removal.message) }
    }

    private func summary(_ inventory: ModelStorageInventory) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(L("Espace utilisé", "Space used")).font(.system(size: 12, weight: .medium))
                .foregroundStyle(StudioTheme.secondary)
            Text(bytes(inventory.totalBytes))
                .font(.system(size: 36, weight: .semibold)).tracking(-0.6).monospacedDigit()
            Text(inventory.sharedBytes > 0
                 ? L("sur le disque, dont \(bytes(inventory.sharedBytes)) partagés", "on disk, including \(bytes(inventory.sharedBytes)) shared")
                 : L("sur le disque", "on disk"))
                .foregroundStyle(StudioTheme.secondary)
            if inventory.totalBytes > 0, inventory.sharedBytes > 0 {
                GeometryReader { bounds in
                    let proportion = min(1, Double(inventory.sharedBytes) / Double(inventory.totalBytes))
                    Capsule().fill(StudioTheme.edge)
                        .overlay(alignment: .leading) {
                            Capsule().fill(studio.accentColor.opacity(0.8)).frame(width: proportion * bounds.size.width)
                        }
                }
                .frame(maxWidth: 225).frame(height: 4).padding(.top, 2)
                .accessibilityHidden(true)
            }
        }
        .studioCard(padding: 18, radius: 13)
        .accessibilityElement(children: .combine)
    }

    private func section(_ kind: ModelStorageKind, items: [ModelStorageItem], narrow: Bool) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            StudioSectionTitle(title: sectionTitle(kind), icon: symbol(kind))
            VStack(alignment: .leading, spacing: 0) {
              ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                let selected = selectedID == item.id
                if index > 0 {
                    Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5).accessibilityHidden(true)
                }
                VStack(alignment: .leading, spacing: 0) {
                    Button {
                        selectedID = selected ? nil : item.id
                        error = nil
                    } label: {
                        HStack(spacing: 10) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.name).fontWeight(.medium)
                                    .lineLimit(2).multilineTextAlignment(.leading)
                                if narrow || !item.canRemoveFiles {
                                    Text(item.complete ? bytes(item.allocatedBytes) : L("Taille incomplète", "Size incomplete"))
                                        .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                            }
                            Spacer(minLength: 5)
                            if item.canRemoveFiles {
                                VStack(alignment: .trailing, spacing: 2) {
                                    Text(bytes(item.reclaimableBytes)).fontWeight(.medium).monospacedDigit()
                                    Text(L("récupérables (est.)", "recoverable (est.)")).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                            }
                            Image(systemName: "chevron.right")
                                .font(.system(size: 9.5, weight: .semibold)).foregroundStyle(StudioTheme.quiet)
                                .rotationEffect(.degrees(selected ? 90 : 0))
                        }
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 11).padding(.horizontal, 14)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(StudioRowStyle(selected: selected, radius: 0))
                    .accessibilityIdentifier("storage.row.\(item.id)")
                    .accessibilityValue(selected ? L("Détails affichés", "Details shown") : "")
                    if selected { details(item) }
                }
                .id(item.id)
              }
            }
            .clipShape(RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
            .studioCard(padding: 0)
        }
    }

    private func details(_ item: ModelStorageItem) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.path)])
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "folder")
                    Text((item.path as NSString).abbreviatingWithTildeInPath)
                        .lineLimit(1).truncationMode(.middle)
                    Image(systemName: "arrow.up.right").font(.system(size: 9))
                }
                .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            }
            .buttonStyle(.plain).help(item.path)
            .accessibilityLabel(L("Afficher dans le Finder", "Show in Finder") + ", " + item.path)
            if item.canRemoveFiles {
                Text(L("\(bytes(item.allocatedBytes)) sur le disque", "\(bytes(item.allocatedBytes)) on disk"))
            }
            if item.sharedBytes > 0 {
                let owners = item.otherOwners.joined(separator: ", ")
                Text(owners.isEmpty
                     ? L("\(bytes(item.sharedBytes)) partagés avec d’autres fichiers.", "\(bytes(item.sharedBytes)) shared with other files.")
                     : L("\(bytes(item.sharedBytes)) partagés avec \(owners).", "\(bytes(item.sharedBytes)) shared with \(owners)."))
            }
            if item.kind == .externalModel {
                Text(L("Ce dossier est géré en dehors de Studio.", "This folder is managed outside Studio."))
            } else if item.kind == .sharedCache {
                Text(L("Ce cache peut être utilisé par d’autres applications.", "Other apps may use this cache."))
            } else if !item.canRemoveFiles {
                Text(L("Studio ne peut pas retirer ces fichiers en toute sécurité.", "Studio cannot safely remove these files."))
            } else if item.kind == .managedModel {
                Text(L("L’espace estimé sera libéré après vidage de la Corbeille.", "The estimated space becomes available after emptying Trash."))
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { itemActions(item) }
                VStack(alignment: .leading, spacing: 6) { itemActions(item) }
            }
            .padding(.top, 3)
        }
        .font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14).padding(.top, 4).padding(.bottom, 14)
    }

    @ViewBuilder private func itemActions(_ item: ModelStorageItem) -> some View {
        if item.canRemoveFiles {
            let removal = StorageRemoval(item: item, files: true)
            Button(removal.title + "…") { requested = removal }
                .buttonStyle(.studio(.destructive)).disabled(busy).accessibilityIdentifier("storage.removeFiles")
        }
        if item.registryName != nil {
            let removal = StorageRemoval(item: item, files: false)
            Button(removal.title + "…") { requested = removal }
                .buttonStyle(.studio).disabled(busy).accessibilityIdentifier("storage.unregister")
        }
    }

    private var busy: Bool {
        loading || studio.storageHasActiveWork || studio.modelInstallState.isWorking || library.discardingStorageID != nil
    }

    private func remove(_ removal: StorageRemoval) {
        guard !busy, inventory?.items.contains(removal.item) == true else { return }
        let item = removal.item
        if removal.files, !item.canRemoveFiles { return }
        error = nil
        if let name = item.registryName, let model = studio.models.first(where: {
            $0.name == name && URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().standardizedFileURL.path == item.path
        }) {
            studio.removeLocalModel(model, trashFiles: removal.files, storagePreview: item)
        } else if removal.files, let id = item.downloadID, let job = library.pending.first(where: { $0.id == id }) {
            library.discard(job, storagePreview: item, studio: studio)
        }
    }

    private func refresh(_ proxy: ScrollViewProxy) async {
        loading = true
        let models = studio.models, pending = library.pending, imagePath = imageModelPath
        let scan = Task.detached(priority: .utility) {
            try ModelStorageInventory.scan(models: models, pending: pending, imageModelPath: imagePath)
        }
        do {
            let result = try await withTaskCancellationHandler {
                try await scan.value
            } onCancel: { scan.cancel() }
            try Task.checkCancellation()
            inventory = result
            if !appliedInitialFocus, let initialFocus, let item = result.items.first(where: initialFocus.matches) {
                selectedID = item.id
                appliedInitialFocus = true
                // Let the newly expanded row participate in layout before scrolling.
                await Task.yield()
                if !Task.isCancelled { proxy.scrollTo(item.id, anchor: .center) }
            } else if !result.items.contains(where: { $0.id == selectedID }) {
                selectedID = nil
            }
        } catch {
            if Task.isCancelled { return }
            self.error = error.localizedDescription
            inventory = nil
        }
        if !Task.isCancelled { loading = false }
    }

    private func bytes(_ value: Int64) -> String {
        value == 0 ? L("0 octet", "0 bytes") : ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    private func sectionTitle(_ kind: ModelStorageKind) -> String {
        switch kind {
        case .managedModel: L("Modèles installés", "Installed models")
        case .externalModel: L("Dossiers externes", "External folders")
        case .partialDownload: L("Téléchargements interrompus", "Interrupted downloads")
        case .sharedCache: L("Cache partagé", "Shared cache")
        }
    }

    private func symbol(_ kind: ModelStorageKind) -> String {
        switch kind {
        case .managedModel: "shippingbox"
        case .externalModel: "folder"
        case .partialDownload: "arrow.down.circle"
        case .sharedCache: "square.stack.3d.up"
        }
    }
}
