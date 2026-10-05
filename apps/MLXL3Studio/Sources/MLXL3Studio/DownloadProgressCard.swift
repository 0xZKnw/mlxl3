import SwiftUI

struct DownloadProgressCard: View {
    let repository: String
    let progress: ModelDownloadProgress
    let status: ModelDownloadStatus
    let message: String?
    let pause: () -> Void
    let resume: (() -> Void)?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var active: Bool { status == .transferring }
    private var icon: String {
        switch status {
        case .complete: "checkmark"
        case .paused: "pause"
        case .failed: "exclamationmark"
        default: "arrow.down"
        }
    }
    private var title: String {
        switch status {
        case .complete: L("AJOUTÉ À VOTRE BIBLIOTHÈQUE", "ADDED TO YOUR LIBRARY")
        case .paused: L("TÉLÉCHARGEMENT EN PAUSE", "DOWNLOAD PAUSED")
        case .failed: L("TÉLÉCHARGEMENT INTERROMPU", "DOWNLOAD INTERRUPTED")
        default: L("TÉLÉCHARGEMENT DU MODÈLE", "MODEL DOWNLOAD")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.system(size: 20, weight: .light))
                    .foregroundStyle(status == .failed ? .orange : StudioTheme.accent)
                    .frame(width: 46, height: 46)
                    .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
                    .overlay { RoundedRectangle(cornerRadius: 12).stroke(StudioTheme.edge, lineWidth: 0.5) }
                VStack(alignment: .leading, spacing: 5) {
                    Text(title).font(.system(size: 8, weight: .medium)).tracking(1.4)
                        .foregroundStyle(StudioTheme.quiet)
                    Text(repository.split(separator: "/").last.map(String.init) ?? repository)
                        .font(.system(size: 14, weight: .medium)).lineLimit(1).truncationMode(.middle)
                    if let owner = repository.split(separator: "/").dropLast().first {
                        Text(owner).font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                    }
                }
                Spacer(minLength: 16)
                if let fraction = progress.fraction {
                    Text(fraction.formatted(.percent.precision(.fractionLength(0))))
                        .font(.system(size: 27, weight: .regular, design: .serif)).monospacedDigit()
                }
                if active {
                    Button(action: pause) { Image(systemName: "pause.fill").frame(width: 32, height: 32) }
                        .buttonStyle(RoundGlassButtonStyle())
                        .help(L("Suspendre le téléchargement", "Pause download"))
                        .accessibilityLabel(L("Suspendre le téléchargement", "Pause download"))
                } else if let resume {
                    Button(action: resume) { Label(L("Reprendre", "Resume"), systemImage: "play.fill") }
                        .buttonStyle(GlassPillButtonStyle())
                }
            }
            if let fraction = progress.fraction {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.07))
                        Capsule().fill(LinearGradient(colors: [StudioTheme.accent.opacity(0.6), StudioTheme.accent],
                                                      startPoint: .leading, endPoint: .trailing))
                            .frame(width: geometry.size.width * fraction)
                    }
                }.frame(height: 6)
                    .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: fraction)
                    .accessibilityLabel(L("Progression du téléchargement", "Download progress"))
                    .accessibilityValue(fraction.formatted(.percent.precision(.fractionLength(0))))
            } else if active { ProgressView().controlSize(.small) }
            HStack(alignment: .firstTextBaseline) {
                Text(volume(progress.completed) + " / " + (progress.total > 0 ? volume(progress.total) : "—"))
                    .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary).monospacedDigit()
                Spacer()
                if active {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.down").font(.system(size: 9))
                        Text(progress.bytesPerSecond.map {
                            ($0 / 1_000_000).formatted(.number.precision(.fractionLength(1))) + " " + L("Mo/s", "MB/s")
                        } ?? L("Calcul du débit…", "Measuring speed…"))
                        .monospacedDigit()
                    }.font(.system(size: 11, weight: .medium)).foregroundStyle(StudioTheme.accent)
                }
            }
            if let message {
                Text(message).font(.system(size: 11))
                    .foregroundStyle(status == .failed ? .orange : StudioTheme.secondary)
                    .textSelection(.enabled).lineLimit(3)
            }
        }
        .padding(18).foregroundStyle(StudioTheme.ink)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(StudioTheme.edge, lineWidth: 0.5) }
    }

    private func volume(_ bytes: Double) -> String {
        if bytes >= 1_000_000_000 {
            return (bytes / 1_000_000_000).formatted(.number.precision(.fractionLength(2))) + " " + L("Go", "GB")
        }
        return (bytes / 1_000_000).formatted(.number.precision(.fractionLength(1))) + " " + L("Mo", "MB")
    }
}
