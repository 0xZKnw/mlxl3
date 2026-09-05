import SwiftUI

struct ModelManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.dismiss) private var dismiss
    @State private var repository = ""
    @State private var revision = ""
    @State private var modelName = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Modèles")
                        .font(.system(size: 24, weight: .semibold, design: .rounded))
                    Text("Importe un dossier local ou télécharge un checkpoint EXL3.")
                        .font(.system(size: 12))
                        .foregroundStyle(StudioTheme.secondary)
                }
                Spacer()
                Button(action: dismiss.callAsFunction) {
                    Image(systemName: "xmark")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(RoundGlassButtonStyle())
            }

            Button(action: studio.importModelFolder) {
                Label("Choisir un dossier EXL3…", systemImage: "folder")
                    .frame(maxWidth: .infinity)
                    .frame(height: 42)
            }
            .buttonStyle(GlassPillButtonStyle())
            .disabled(studio.modelInstallState.isWorking)

            HStack(spacing: 10) {
                Rectangle().fill(StudioTheme.edge).frame(height: 1)
                Text("OU HUGGING FACE")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(1.2)
                    .foregroundStyle(StudioTheme.quiet)
                Rectangle().fill(StudioTheme.edge).frame(height: 1)
            }

            VStack(alignment: .leading, spacing: 12) {
                ModelField(
                    title: "DÉPÔT",
                    placeholder: "UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw",
                    text: $repository
                )
                HStack(spacing: 12) {
                    ModelField(
                        title: "RÉVISION (OPTIONNEL)",
                        placeholder: "2.49bpw",
                        text: $revision
                    )
                    ModelField(
                        title: "NOM LOCAL (OPTIONNEL)",
                        placeholder: "qwen3.6-35b-a3b",
                        text: $modelName
                    )
                }
            }

            Button {
                studio.downloadModel(
                    repo: repository,
                    revision: revision,
                    name: modelName
                )
            } label: {
                HStack(spacing: 9) {
                    if studio.modelInstallState.isWorking {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "arrow.down.circle.fill")
                    }
                    Text(studio.modelInstallState.isWorking ? "Téléchargement…" : "Télécharger")
                }
                .frame(maxWidth: .infinity)
                .frame(height: 42)
            }
            .buttonStyle(PrimaryGlassButtonStyle())
            .disabled(
                repository.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || studio.modelInstallState.isWorking
            )

            installStatus
        }
        .padding(26)
        .frame(width: 620)
        .background(StudioTheme.canvas)
    }

    @ViewBuilder
    private var installStatus: some View {
        switch studio.modelInstallState {
        case .idle:
            Label(
                "Les modèles restent sur ce Mac. Les dépôts privés utilisent le token HF local.",
                systemImage: "lock.fill"
            )
            .foregroundStyle(StudioTheme.quiet)
        case let .working(message):
            Label(message, systemImage: "arrow.down.circle")
                .foregroundStyle(StudioTheme.accent)
        case let .succeeded(message):
            Label(message, systemImage: "checkmark.circle.fill")
                .foregroundStyle(Color(red: 0.42, green: 1.0, blue: 0.69))
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(Color(red: 1, green: 0.56, blue: 0.56))
        }
    }
}

private struct ModelField: View {
    let title: String
    let placeholder: String
    @Binding var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title)
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(1.1)
                .foregroundStyle(StudioTheme.quiet)
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .padding(.horizontal, 12)
                .frame(height: 38)
                .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(StudioTheme.edge, lineWidth: 0.7)
                }
        }
        .frame(maxWidth: .infinity)
    }
}
