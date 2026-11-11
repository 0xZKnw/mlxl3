import SwiftUI

struct DownloadNotificationButton: View {
    @ObservedObject var library: ModelLibrary
    @EnvironmentObject private var studio: StudioModel
    @State private var isPresented = false

    private var statusDescription: String {
        switch library.downloadStatus {
        case .idle: L("Aucun téléchargement", "No downloads")
        case .transferring:
            library.downloadProgress.fraction.map {
                L("En cours, \($0.formatted(.percent.precision(.fractionLength(0))))", "In progress, \($0.formatted(.percent.precision(.fractionLength(0))))")
            } ?? L("En cours", "In progress")
        case .paused: L("En pause", "Paused")
        case .complete: L("Terminé", "Complete")
        case .failed: L("Interrompu", "Interrupted")
        }
    }

    var body: some View {
        Button { isPresented.toggle() } label: {
            Image(systemName: "arrow.down.circle")
                .frame(width: 30, height: 30)
                .overlay(alignment: .topTrailing) {
                    if library.downloadStatus == .transferring || library.downloadStatus == .paused || library.downloadStatus == .failed || library.downloadStatus == .complete {
                        Circle().fill(library.downloadStatus == .failed ? Color.orange : StudioTheme.accent)
                            .frame(width: 6, height: 6).offset(x: -2, y: 2)
                    }
                }
        }
        .buttonStyle(StudioControlStyle(emphasized: isPresented))
        .help(L("Téléchargements", "Downloads"))
        .accessibilityLabel(L("Téléchargements", "Downloads"))
        .accessibilityValue(statusDescription)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 16) {
                Text(L("Téléchargements", "Downloads"))
                    .font(.system(size: 14, weight: .semibold))
                if let repo = library.lastDownloadRepo {
                    let job = library.pending.first { $0.repo == repo }
                    DownloadProgressCard(
                        repository: repo, progress: library.downloadProgress,
                        status: library.downloadStatus, message: library.downloadMessage,
                        pause: library.cancelDownload,
                        resume: (library.downloadStatus == .paused || library.downloadStatus == .failed) && job != nil
                            ? { if let job { library.resume(job, studio: studio) } } : nil)
                } else {
                    Text(L("Aucun téléchargement en cours", "No downloads in progress"))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                }
            }
            .padding(18)
            .frame(width: 330, alignment: .leading)
            .foregroundStyle(StudioTheme.ink)
        }
    }
}

struct DownloadProgressCard: View {
    let repository: String
    let progress: ModelDownloadProgress
    let status: ModelDownloadStatus
    let message: String?
    let pause: () -> Void
    let resume: (() -> Void)?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var active: Bool { status == .transferring }
    private var transferring: Bool { progress.phase == .download }
    private var title: String {
        switch status {
        case .complete: L("Terminé", "Complete")
        case .paused: L("En pause", "Paused")
        case .failed: L("Interrompu", "Interrupted")
        default:
            switch progress.phase {
            case .download: L("Téléchargement", "Downloading")
            case .verify: L("Vérification", "Verifying")
            case .publication: L("Publication", "Publishing")
            case .install: L("Installation", "Installing")
            case .processing: L("Préparation", "Preparing")
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: status == .complete ? "checkmark.circle" : status == .failed ? "exclamationmark.circle" : "shippingbox")
                    .font(.system(size: 16)).foregroundStyle(status == .failed ? .orange : StudioTheme.secondary)
                    .frame(width: 20, height: 22)
                VStack(alignment: .leading, spacing: 3) {
                    Text(repository.split(separator: "/").last.map(String.init) ?? repository)
                        .font(.system(size: 12, weight: .semibold)).lineLimit(2).truncationMode(.middle)
                    Text(title).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                }
                Spacer(minLength: 4)
                if active {
                    Button(action: pause) { Image(systemName: "pause.fill").frame(width: 28, height: 28) }
                        .buttonStyle(StudioControlStyle())
                        .help(L("Suspendre le téléchargement", "Pause download"))
                        .accessibilityLabel(L("Suspendre le téléchargement", "Pause download"))
                } else if let resume {
                    Button(L("Reprendre", "Resume"), action: resume).controlSize(.small)
                }
            }
            if let fraction = progress.fraction {
                ProgressView(value: fraction).tint(StudioTheme.accent)
                    .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: fraction)
                    .accessibilityLabel(L("Progression du téléchargement", "Download progress"))
                    .accessibilityValue(fraction.formatted(.percent.precision(.fractionLength(0))))
                HStack {
                    Text(volume(progress.completed) + " / " + (progress.total > 0 ? volume(progress.total) : "—"))
                    Spacer()
                    Text(fraction.formatted(.percent.precision(.fractionLength(0))))
                }.font(.system(size: 11)).foregroundStyle(StudioTheme.secondary).monospacedDigit()
            } else if active {
                ProgressView().controlSize(.small).accessibilityLabel(title)
            }
            if active && transferring, let rate = progress.bytesPerSecond {
                Text((rate / 1_000_000).formatted(.number.precision(.fractionLength(1))) + " " + L("Mo/s", "MB/s"))
                    .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary).monospacedDigit()
            }
            if let message, status != .complete {
                Text(message).font(.system(size: 11))
                    .foregroundStyle(status == .failed ? .orange : StudioTheme.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func volume(_ bytes: Double) -> String {
        if bytes >= 1_000_000_000 {
            return (bytes / 1_000_000_000).formatted(.number.precision(.fractionLength(2))) + " " + L("Go", "GB")
        }
        return (bytes / 1_000_000).formatted(.number.precision(.fractionLength(1))) + " " + L("Mo", "MB")
    }
}
