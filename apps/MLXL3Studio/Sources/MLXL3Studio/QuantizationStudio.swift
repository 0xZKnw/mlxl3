import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct QuantizationSource: Codable, Sendable {
  let source: String
  let revision: String
  let fingerprint: String
  let modelType: String
  let inferenceSupported: Bool
  let conversionSupported: Bool
  let reason: String
  let sourceBytes: Int64
  let parameters: Int64
  let modules: [String]
  let headerChecked: Bool
  let tiedHead: Bool
}

struct QuantizationSettings: Encodable, Sendable {
  var bits = 4.0
  var headBits = 6
  var rows = 1024
  var sequence = 1024
  var excluded: [String] = []
  var keepHead = false
  var calibration = ""
  var validation = ""
}

@MainActor
final class QuantizationModel: ObservableObject {
  @Published var input = "" {
    didSet {
      if input != oldValue && !converting {
        task?.cancel()
        command?.cancel()
        operation = UUID()
        inspecting = false
        source = nil
        output = nil
        quality = nil
      }
    }
  }
  @Published var settings = QuantizationSettings()
  @Published private(set) var source: QuantizationSource?
  @Published private(set) var inspecting = false
  @Published private(set) var converting = false
  @Published private(set) var progress: Double?
  @Published private(set) var phase = ""
  @Published private(set) var logs = ""
  @Published private(set) var error: String?
  @Published private(set) var output: URL?
  @Published private(set) var quality: String?
  private var task: Task<Void, Never>?
  private var operation = UUID()
  private var command: CLICommand?
  private var job: URL?
  private var stopRequested = false
  private let workerOverride: URL?
  private let jobsDirectory: URL

  init(worker: URL? = nil, jobsDirectory: URL? = nil) {
    workerOverride = worker
    self.jobsDirectory =
      jobsDirectory
      ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("io.mlxl3.desktop/quantization-jobs")
  }

  var worker: URL? {
    if let workerOverride { return workerOverride }
    if let override = ProcessInfo.processInfo.environment["MLXL3_QUANTIZER"], !override.isEmpty {
      return URL(fileURLWithPath: override)
    }
    guard let resources = Bundle.main.resourceURL else { return nil }
    let url = resources.appendingPathComponent("quantizer/mlxl3-quantize")
    return FileManager.default.isExecutableFile(atPath: url.path) ? url : nil
  }

  func inspect() {
    guard !converting else { return }
    task?.cancel()
    command?.cancel()
    operation = UUID()
    let ticket = operation
    let sourceInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
    source = nil
    output = nil
    error = nil
    quality = nil
    inspecting = true
    task = Task {
      defer { if ticket == operation { inspecting = false } }
      do {
        guard let worker else {
          throw MLXL3BridgeError.commandFailed(
            L(
              "Le convertisseur est absent de ce paquet. Utilise le DMG bêta 5 complet.",
              "This package has no converter. Use the full beta 5 DMG."))
        }
        let runner = CLICommand(executable: worker)
        command = runner
        let result = try await runner.output(
          [],
          input: JSONSerialization.data(withJSONObject: [
            "operation": "probe", "source": sourceInput,
          ]))
        guard ticket == operation, !Task.isCancelled else { return }
        let line = result.split(separator: 10).last ?? result[...]
        let event = try JSONSerialization.jsonObject(with: Data(line)) as! [String: Any]
        guard event["type"] as? String == "result", let value = event["result"] else {
          throw MLXL3BridgeError.invalidResponse
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        source = try decoder.decode(
          QuantizationSource.self, from: JSONSerialization.data(withJSONObject: value))
      } catch {
        if ticket == operation && !Task.isCancelled { self.error = error.localizedDescription }
      }
    }
  }

  func convert(to destination: URL, prepare: @escaping @MainActor () async -> Void) {
    guard !inspecting, !converting, let source, source.conversionSupported, let worker else {
      return
    }
    operation = UUID()
    let ticket = operation
    stopRequested = false
    job = nil
    error = nil
    logs = ""
    output = nil
    quality = nil
    progress = nil
    converting = true
    phase = L("Libération du modèle interactif…", "Unloading the interactive model…")
    let settings = settings
    task = Task {
      defer { if ticket == operation { converting = false } }
      await prepare()
      guard !Task.isCancelled, ticket == operation else { return }
      do {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let root = jobsDirectory
        var sourceValue =
          try JSONSerialization.jsonObject(with: encoder.encode(source)) as! [String: Any]
        sourceValue["modules"] = []
        let request: [String: Any] = [
          "operation": "convert", "engine": CLIResolver.executable()?.path ?? "",
          "source": sourceValue,
          "settings": try JSONSerialization.jsonObject(with: encoder.encode(settings)),
          "output": destination.path, "jobs": root.path,
        ]
        let runner = CLICommand(executable: worker)
        command = runner
        let final = try await runner.output(
          [], input: JSONSerialization.data(withJSONObject: request)
        ) { [weak self = self] line in
          guard let self, self.operation == ticket, self.converting, !self.stopRequested else {
            return
          }
          self.receive(line, destination: destination)
        }
        guard !Task.isCancelled, ticket == operation else { return }
        if let last = final.split(separator: 10).last {
          receive(Data(last), destination: destination)
        }
        if output == nil {
          throw MLXL3BridgeError.commandFailed(
            L(
              "Conversion terminée sans validation complète.",
              "Conversion ended without complete validation."))
        }
      } catch {
        if ticket == operation && !Task.isCancelled && self.error == nil {
          self.error = error.localizedDescription
        }
      }
    }
  }

  private func receive(_ line: Data, destination: URL) {
    guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
      let type = event["type"] as? String
    else { return }
    switch type {
    case "progress":
      let text = event["text"] as? String ?? ""
      phase = (event["phase"] as? String ?? "") + " · " + text
      let completed = event["completed"] as? Double ?? 0
      let total = event["total"] as? Double ?? 0
      progress =
        total > 0 && completed.isFinite && total.isFinite ? min(1, max(0, completed / total)) : nil
    case "log":
      logs = String((logs + (event["text"] as? String ?? "") + "\n").suffix(24_000))
    case "job": job = (event["path"] as? String).map { URL(fileURLWithPath: $0) }
    case "error": error = event["text"] as? String
    case "result":
      guard let result = event["result"] as? [String: Any],
        result["status"] as? String == "complete", result["output"] as? String == destination.path
      else { return }
      output = destination
      progress = 1
      phase = L(
        "Conversion EXL3 terminée et fichiers vérifiés",
        "EXL3 conversion complete; checkpoint files verified")
      if let metrics = result["quality"] as? [String: Any], let ppl = metrics["ppl"] as? Double,
        let teacher = metrics["teacher_ppl"] as? Double, let kl = metrics["kl"] as? Double
      {
        quality = String(
          format: "PPL %.4f · Source %.4f · KL %.6f · %@ tokens", ppl, teacher, kl,
          String(describing: metrics["tokens"] ?? 0))
      }
    default: break
    }
  }

  func cancel() {
    stopRequested = true
    task?.cancel()
    command?.cancel()
    progress = nil
    phase = L(
      "Arrêt demandé. Les fichiers partiels restent hors de la bibliothèque.",
      "Stop requested. Partial files remain outside the library.")
    if let job, let data = try? Data(contentsOf: job.appendingPathComponent("job.json")),
      var state = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    {
      state["status"] = "interrupted"
      if let saved = try? JSONSerialization.data(withJSONObject: state, options: .prettyPrinted) {
        try? saved.write(to: job.appendingPathComponent("job.json"), options: .atomic)
      }
    }
    job = nil
  }
}

struct QuantizationStudioView: View {
  @EnvironmentObject private var studio: StudioModel
  @ObservedObject var quantizer: QuantizationModel
  @State private var expert = false
  @State private var preset = 4.0
  @State private var exclusions = ""
  @State private var validate = false

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        Text("Quantization Studio").font(.system(size: 29, design: .serif))
        Text(L("Safetensors → EXL3, sur ce Mac.", "Safetensors → EXL3, on this Mac."))
          .foregroundStyle(StudioTheme.quiet)
        HStack {
          TextField(
            L("Dossier local ou lien Hugging Face", "Local folder or Hugging Face URL"),
            text: $quantizer.input
          ).textFieldStyle(.roundedBorder).onSubmit { quantizer.inspect() }
          Button(L("Dossier…", "Folder…"), systemImage: "folder") { chooseSource() }
          Button(L("Vérifier", "Inspect")) { quantizer.inspect() }.disabled(quantizer.input.isEmpty)
        }.disabled(quantizer.inspecting || quantizer.converting)
        if quantizer.inspecting {
          HStack {
            ProgressView(
              L(
                "Lecture de la configuration et des en-têtes…", "Reading configuration and headers…"
              ))
            Button(L("Annuler", "Cancel")) { quantizer.cancel() }
          }
        }
        if let source = quantizer.source {
          VStack(alignment: .leading, spacing: 9) {
            Text(source.modelType.uppercased()).font(.headline)
            compatibility(
              L("Architecture du moteur", "Engine architecture"),
              available: source.inferenceSupported)
            compatibility(
              L("Conversion EXL3", "EXL3 conversion"), available: source.conversionSupported)
            Text(source.reason).font(.caption).foregroundStyle(
              source.conversionSupported ? StudioTheme.secondary : .orange)
            Text(
              ByteCountFormatter.string(fromByteCount: source.sourceBytes, countStyle: .file)
                + " · "
                + (source.headerChecked
                  ? L("En-têtes vérifiés", "Headers verified")
                  : L(
                    "En-têtes vérifiés après téléchargement, avant conversion",
                    "Headers checked after download, before conversion"))
            ).font(.caption).foregroundStyle(StudioTheme.quiet)
            Text(
              source.revision == "local"
                ? source.source : source.source + " @ " + source.revision.prefix(12)
            ).font(.system(size: 10, design: .monospaced)).textSelection(.enabled)
          }.padding(15).frame(maxWidth: .infinity, alignment: .leading).background(
            StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
          if source.conversionSupported {
            Picker(L("Mode", "Mode"), selection: $expert) {
              Text("Simple").tag(false)
              Text("Expert").tag(true)
            }.pickerStyle(.segmented)
            Group {
              if !expert {
                Picker(L("Objectif", "Target"), selection: $preset) {
                  Text(L("Mémoire réduite · 3 bpw", "Smaller weights · 3 bpw")).tag(3.0)
                  Text(L("Équilibré · 4 bpw", "Balanced · 4 bpw")).tag(4.0)
                  Text(L("Plus fidèle · 6 bpw", "Higher fidelity · 6 bpw")).tag(6.0)
                }.onChange(of: preset) { _, value in quantizer.settings.bits = value }
                Text(
                  L(
                    "Les embeddings et autres tenseurs non quantifiés s’ajoutent aux poids EXL3. La calibration utilise du texte WikiText-2 inclus. Aucun réglage ne garantit à lui seul la qualité ou la RAM nécessaire.",
                    "Embeddings and other plain tensors add to the EXL3 weights. Calibration uses the included WikiText-2 excerpt. A setting alone does not guarantee quality or RAM usage."
                  )
                ).font(.caption).foregroundStyle(StudioTheme.quiet)
              } else {
                expertSettings(source)
              }
            }.disabled(quantizer.converting)
            HStack {
              Button(L("Quantifier…", "Quantize…"), systemImage: "slider.horizontal.3") { start() }
                .buttonStyle(.borderedProminent).disabled(
                  quantizer.converting || studio.isGenerating || studio.localServer.active
                    || studio.agentStudioBusy
                    || expert && validate && quantizer.settings.validation.isEmpty)
              if quantizer.converting {
                Button(L("Arrêter", "Stop"), role: .destructive) { quantizer.cancel() }
              }
              Spacer()
              if let output = quantizer.output {
                Button(L("Ajouter à la bibliothèque", "Add to library")) {
                  studio.addQuantizedModel(output)
                }
                Button {
                  NSWorkspace.shared.activateFileViewerSelecting([output])
                } label: {
                  Image(systemName: "folder")
                }
              }
            }
          }
        }
        if !quantizer.phase.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            if let progress = quantizer.progress {
              ProgressView(value: progress)
            } else if quantizer.converting {
              ProgressView().controlSize(.small)
            }
            Text(quantizer.phase).font(.caption).textSelection(.enabled)
            if let quality = quantizer.quality {
              Text(quality).font(.system(size: 11, design: .monospaced))
              Text(
                L(
                  "Validation limitée à une fenêtre indépendante de 256 tokens maximum.",
                  "Validation covers one independent window of at most 256 tokens.")
              ).font(.caption).foregroundStyle(StudioTheme.quiet)
            }
          }
        }
        if let error = quantizer.error {
          Text(error).foregroundStyle(.orange).font(.caption).textSelection(.enabled)
        }
        if !quantizer.logs.isEmpty {
          DisclosureGroup(L("Journal de conversion", "Conversion log")) {
            ScrollView {
              Text(quantizer.logs).font(.system(size: 10, design: .monospaced)).textSelection(
                .enabled
              ).frame(maxWidth: .infinity, alignment: .leading)
            }.frame(height: 170)
          }
        }
        Text(
          L(
            "Les modèles d’inférence sont éjectés avant conversion. Les poids source restent inchangés et la sortie doit être un nouveau dossier. LFM2, LFM2-MoE et Ling natif disposent d’adaptateurs ; les autres architectures peuvent être lisibles par le moteur sans être convertibles ici.",
            "Interactive models are unloaded before conversion. Source weights remain unchanged and output must be a new folder. LFM2, LFM2-MoE and native Ling have adapters; other architectures may run in the engine without being convertible here."
          )
        ).font(.caption).foregroundStyle(StudioTheme.quiet)
      }.padding(26)
    }
  }

  private func compatibility(_ label: String, available: Bool) -> some View {
    Label(
      label + " · "
        + (available ? L("pris en charge", "supported") : L("non pris en charge", "unsupported")),
      systemImage: available ? "checkmark.circle" : "minus.circle"
    ).font(.caption)
  }

  private func hint(_ title: String, _ explanation: String) -> some View {
    HStack {
      Text(title)
      Image(systemName: "questionmark.circle").foregroundStyle(StudioTheme.quiet).help(explanation)
        .accessibilityLabel(explanation)
    }.font(.caption)
  }

  private func expertSettings(_ source: QuantizationSource) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        hint(
          "BPW",
          L(
            "Bits moyens par poids des projections quantifiées, entre 2 et 8. Moins de bits réduit la taille et peut diminuer la qualité. Les tenseurs FP16 sont en plus.",
            "Average bits per quantized projection weight, from 2 to 8. Fewer bits reduces size and can reduce quality. Plain FP16 tensors are additional."
          ))
        Slider(value: $quantizer.settings.bits, in: 2...8, step: 0.1)
        Text(String(format: "%.1f", quantizer.settings.bits)).monospacedDigit().frame(width: 32)
      }
      HStack {
        hint(
          L("Lignes de calibration", "Calibration rows"),
          L(
            "Nombre d’activations capturées par projection pour guider LDLQ. Plus de lignes utilise davantage de temps, de mémoire et de disque.",
            "Activation rows captured per projection for LDLQ. More rows use more time, memory and disk."
          ))
        Stepper(
          "\(quantizer.settings.rows)", value: $quantizer.settings.rows, in: 32...8192, step: 32)
        hint(
          L("Contexte", "Sequence"),
          L(
            "Longueur des séquences de calibration, indépendante du contexte utilisé en Chat.",
            "Calibration sequence length, independent of Chat context."))
        Stepper(
          "\(quantizer.settings.sequence)", value: $quantizer.settings.sequence, in: 32...4096,
          step: 32)
      }
      HStack {
        hint(
          L("Texte de calibration", "Calibration text"),
          L(
            "Un fichier UTF-8 représentatif. Sans fichier, le convertisseur utilise son extrait WikiText-2 inclus.",
            "Representative UTF-8 text. Without a file, the converter uses its included WikiText-2 excerpt."
          ))
        Text(
          quantizer.settings.calibration.isEmpty
            ? "WikiText-2" : URL(fileURLWithPath: quantizer.settings.calibration).lastPathComponent
        ).font(.caption)
        Button(L("Choisir…", "Choose…")) { chooseText(validation: false) }
        if !quantizer.settings.calibration.isEmpty {
          Button(L("Réinitialiser", "Reset")) { quantizer.settings.calibration = "" }
        }
      }
      hint(
        L("Couches exclues", "Excluded modules"),
        L(
          "Préfixes exacts, un par ligne, par exemple model.layers.0. Ces poids restent en FP16 ; les noms sans correspondance sont refusés.",
          "Exact prefixes, one per line, e.g. model.layers.0. These weights remain FP16; unmatched names are refused."
        ))
      TextField("model.layers.0", text: $exclusions, axis: .vertical).textFieldStyle(.roundedBorder)
      if !source.modules.isEmpty {
        DisclosureGroup(L("Noms des modules", "Module names")) {
          ScrollView {
            Text(source.modules.prefix(1000).joined(separator: "\n")).font(
              .system(size: 9, design: .monospaced)
            ).textSelection(.enabled)
          }.frame(height: 100)
        }
      }
      HStack {
        hint(
          "lm_head",
          L(
            "Projection du vocabulaire. Une précision supérieure peut aider la qualité, au prix de poids plus lourds. Une tête liée suit l’embedding conservé en FP16.",
            "Vocabulary projection. Higher precision can help quality at a size cost. A tied head follows the plain FP16 embedding."
          ))
        if source.tiedHead {
          Text(L("Liée à l’embedding · FP16", "Tied to embedding · FP16")).font(.caption)
        } else {
          Toggle(L("Conserver en FP16", "Keep FP16"), isOn: $quantizer.settings.keepHead)
            .toggleStyle(.checkbox)
          Stepper(
            "\(quantizer.settings.headBits) bits", value: $quantizer.settings.headBits, in: 2...8
          ).disabled(quantizer.settings.keepHead)
        }
      }
      HStack {
        Toggle(L("Validation KL / perplexité", "KL / perplexity validation"), isOn: $validate)
          .toggleStyle(.checkbox).disabled(source.modelType == "bailing_hybrid")
          .onChange(of: validate) { _, enabled in if !enabled { quantizer.settings.validation = "" }
          }
        hint(
          "",
          L(
            "Compare BF16 et EXL3 sur un texte séparé, 256 tokens maximum, un seul modèle GPU à la fois. Exige que le teacher complet tienne en RAM ; le résultat ne certifie pas toute la qualité.",
            "Compare BF16 and EXL3 on separate text, up to 256 tokens, one GPU model at a time. Requires the full teacher to fit in RAM; this is limited quality evidence."
          ))
        if validate {
          Button(
            quantizer.settings.validation.isEmpty
              ? L("Choisir le texte test…", "Choose held-out text…")
              : URL(fileURLWithPath: quantizer.settings.validation).lastPathComponent
          ) { chooseText(validation: true) }
        }
      }
    }
  }

  private func chooseSource() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = true
    panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    quantizer.input = url.path
    quantizer.inspect()
  }

  private func chooseText(validation: Bool) {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.plainText]
    panel.canChooseDirectories = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    if validation {
      quantizer.settings.validation = url.path
    } else {
      quantizer.settings.calibration = url.path
    }
  }

  private func start() {
    if !expert {
      quantizer.settings = QuantizationSettings()
      quantizer.settings.bits = preset
    } else {
      quantizer.settings.excluded = exclusions.split(whereSeparator: { $0.isNewline || $0 == "," })
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
    guard !validate || !expert || !quantizer.settings.validation.isEmpty else { return }
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "Model-EXL3-\(String(format: "%.1f", quantizer.settings.bits))bpw"
    panel.canCreateDirectories = true
    panel.message = L(
      "Nouveau dossier de sortie. Les fichiers existants ne seront pas écrasés.",
      "New output folder. Existing files will not be overwritten.")
    guard panel.runModal() == .OK, let output = panel.url else { return }
    quantizer.convert(to: output) { await studio.prepareForQuantization() }
  }
}
