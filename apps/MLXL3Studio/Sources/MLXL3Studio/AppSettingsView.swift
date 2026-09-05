import SwiftUI

struct AppSettingsView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 13) {
                MonogramMark(size: 26)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Réglages")
                        .font(.system(size: 24, weight: .regular, design: .serif))
                    Text("MLXL3 Desktop")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(RoundGlassButtonStyle())
            }
            .padding(.horizontal, 26)
            .padding(.top, 25)
            .padding(.bottom, 20)

            Rectangle()
                .fill(Color.white.opacity(0.075))
                .frame(height: 1)

            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    UpdateSettingsCard(
                        updater: studio.updateManager,
                        canInstall: !studio.isGenerating,
                        install: studio.installUpdateAndRestart
                    )

                    VStack(alignment: .leading, spacing: 11) {
                        Label("Comment ça marche", systemImage: "shippingbox")
                            .font(.system(size: 13, weight: .semibold))
                        Text("Le moteur MLXL3 et l’interface sont livrés dans le même DMG. Une nouvelle release GitHub met donc les deux à jour ensemble, sans désynchroniser leurs versions.")
                            .font(.system(size: 12.5, weight: .regular))
                            .foregroundStyle(StudioTheme.secondary)
                            .lineSpacing(4)
                        HStack(spacing: 8) {
                            Label("Vérification au démarrage", systemImage: "checkmark.circle.fill")
                            Label("DMG vérifié en SHA-256", systemImage: "lock.shield.fill")
                        }
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(StudioTheme.quiet)
                    }
                    .padding(.vertical, 17)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(26)
            }
        }
        .frame(width: 650, height: 560)
        .background(StudioTheme.canvas)
        .preferredColorScheme(.dark)
    }
}

private struct UpdateSettingsCard: View {
    @ObservedObject var updater: UpdateManager
    let canInstall: Bool
    let install: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                        .fill(Color.white.opacity(0.07))
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(StudioTheme.accent)
                }
                .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Mises à jour")
                        .font(.system(size: 15, weight: .bold))
                    Text("Version installée · \(updater.currentVersion)")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                updateAction
            }

            HStack(spacing: 10) {
                stateIcon
                VStack(alignment: .leading, spacing: 3) {
                    Text(statusTitle)
                        .font(.system(size: 12.5, weight: .semibold))
                    Text(statusDetail)
                        .font(.system(size: 11, weight: .regular))
                        .foregroundStyle(StudioTheme.quiet)
                        .lineLimit(3)
                }
                Spacer(minLength: 0)
            }
            .padding(13)
            .background(Color.black.opacity(0.20), in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.07), lineWidth: 0.7)
            }

            if let release = updater.latestRelease,
               updater.hasAvailableUpdate,
               !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text("NOUVEAUTÉS · \(release.tag.uppercased())")
                            .font(.system(size: 8.5, weight: .bold))
                            .tracking(1.1)
                            .foregroundStyle(StudioTheme.quiet)
                        Spacer()
                        Button("Voir la release") { updater.openReleasePage() }
                            .buttonStyle(.plain)
                            .font(.system(size: 10.5, weight: .semibold))
                            .foregroundStyle(StudioTheme.accent)
                    }
                    Text(release.notes)
                        .font(.system(size: 11.5, weight: .regular))
                        .foregroundStyle(StudioTheme.secondary)
                        .lineSpacing(3)
                        .lineLimit(7)
                        .textSelection(.enabled)
                }
            }

            if case .ready = updater.state, !canInstall {
                Text("Termine ou arrête la génération avant de redémarrer pour installer.")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(Color.orange.opacity(0.82))
            }
        }
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var updateAction: some View {
        switch updater.state {
        case .ready:
            Button(action: install) {
                Label("Redémarrer et installer", systemImage: "arrow.clockwise")
                    .font(.system(size: 11, weight: .bold))
                    .padding(.horizontal, 13)
                    .frame(height: 34)
            }
            .buttonStyle(PrimaryGlassButtonStyle())
            .disabled(!canInstall)
        case .checking, .downloading, .installing:
            ProgressView()
                .controlSize(.small)
                .frame(width: 34, height: 34)
        default:
            Button(action: updater.checkForUpdates) {
                Text("Rechercher")
                    .font(.system(size: 11, weight: .bold))
                    .padding(.horizontal, 15)
                    .frame(height: 34)
            }
            .buttonStyle(GlassPillButtonStyle())
        }
    }

    @ViewBuilder
    private var stateIcon: some View {
        switch updater.state {
        case .checking, .downloading, .installing:
            ProgressView().controlSize(.small)
        case .ready:
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(StudioTheme.accent)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.orange.opacity(0.9))
        case .upToDate:
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(Color(red: 0.42, green: 1.0, blue: 0.69))
        case .idle:
            Image(systemName: "circle.dotted")
                .foregroundStyle(StudioTheme.quiet)
        }
    }

    private var statusTitle: String {
        switch updater.state {
        case .idle: "Prêt à vérifier"
        case .checking: "Recherche sur GitHub…"
        case let .upToDate(date):
            "MLXL3 est à jour · \(date.formatted(date: .omitted, time: .shortened))"
        case let .downloading(release): "Téléchargement de \(release.tag)…"
        case let .ready(release, _): "\(release.tag) est prête"
        case let .installing(release): "Préparation de \(release.tag)…"
        case .failed: "Mise à jour impossible"
        }
    }

    private var statusDetail: String {
        switch updater.state {
        case .idle:
            "Recherche les nouvelles versions du moteur et de l’interface."
        case .checking:
            "Lecture de la dernière release publique de 0xZKnw/mlxl3."
        case .upToDate:
            "Le moteur et l’interface utilisent la dernière version publiée."
        case let .downloading(release):
            "Le DMG de \(ByteCountFormatter.string(fromByteCount: release.asset.size, countStyle: .file)) est téléchargé en arrière-plan."
        case .ready:
            "Le téléchargement et l’empreinte SHA-256 sont validés."
        case .installing:
            "MLXL3 va se fermer, remplacer l’app puis se relancer automatiquement."
        case let .failed(message):
            message
        }
    }
}

struct UpdateStatusButton: View {
    @ObservedObject var updater: UpdateManager
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 30, height: 30)
                if updater.hasAvailableUpdate {
                    Circle()
                        .fill(StudioTheme.accent)
                        .frame(width: 7, height: 7)
                        .overlay(Circle().stroke(StudioTheme.sidebar, lineWidth: 1.5))
                        .offset(x: 1, y: -1)
                }
            }
        }
        .buttonStyle(RoundGlassButtonStyle())
        .help(updater.hasAvailableUpdate ? "Mise à jour prête" : "Réglages")
    }
}
