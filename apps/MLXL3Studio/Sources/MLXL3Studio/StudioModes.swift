import AppKit
import Darwin
import Metal
import SwiftUI

enum StudioMode: String, CaseIterable, Identifiable, Codable {
  case chat, agent, server, lab
  var id: String { rawValue }
  var title: String { rawValue.capitalized }
  var icon: String {
    switch self {
    case .chat: "bubble.left.and.bubble.right"
    case .agent: "terminal"
    case .server: "server.rack"
    case .lab: "chart.xyaxis.line"
    }
  }
}

struct LabSample: Codable, Identifiable, Sendable {
  var id = UUID()
  var date = Date()
  let model: String
  let modelPath: String
  let bits: Double?
  let mode: String
  let runtime: String
  let system: String
  let hardware: String
  let context: Int
  let temperature: Double
  let topK: Int
  let mtp: Bool
  let dflash: Bool
  let stats: GenerationStats

  var valid: Bool {
    !model.isEmpty && stats.promptTokens >= 0 && stats.generatedTokens > 0
      && [stats.ttftSeconds, stats.prefillTps, stats.decodeTps].allSatisfy {
        $0.isFinite && $0 >= 0
      }
  }
}

enum LabHardware {
  static func value(_ key: String) -> String {
    var size = 0
    guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return "—" }
    var bytes = [CChar](repeating: 0, count: size)
    guard sysctlbyname(key, &bytes, &size, nil, 0) == 0 else { return "—" }
    return String(
      decoding: bytes.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
  }
  static let chip = value("machdep.cpu.brand_string")
  static let machine = value("hw.model")
  static let gpu: (name: String, recommendedBytes: UInt64)? = {
    guard let device = MTLCreateSystemDefaultDevice() else { return nil }
    return (device.name, device.recommendedMaxWorkingSetSize)
  }()
  static var identity: String {
    "\(machine) · \(chip) · \(ProcessInfo.processInfo.physicalMemory) B"
  }
}

@MainActor
final class LocalServer: ObservableObject {
  @Published private(set) var running = false
  @Published private(set) var busy = false
  @Published private(set) var port: Int?
  @Published private(set) var apiKey: String?
  @Published private(set) var error: String?
  @Published private(set) var log = ""
  private var process: Process?
  private var ticket = UUID()
  private var pending = Data()
  private var watchdog: Task<Void, Never>?
  var active: Bool { process != nil }

  func start(model: String, port: Int, context: Int, executable: URL? = CLIResolver.executable()) {
    guard !active else { return }
    guard (1024...65535).contains(port), let executable else {
      error = L(
        "Port invalide (1024–65535) ou moteur absent.",
        "Invalid port (1024–65535) or missing engine.")
      return
    }
    let process = Process()
    let output = Pipe()
    let ticket = UUID()
    self.ticket = ticket
    process.executableURL = executable
    let apiKey = UUID().uuidString
    process.arguments = [
      "serve", model, "--host", "127.0.0.1", "--port", String(port), "--api-key", apiKey,
      "--context-length", String(context),
    ]
    process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = output
    process.standardError = output
    self.process = process
    busy = true
    running = false
    self.port = nil
    self.apiKey = apiKey
    error = nil
    log = ""
    pending = Data()
    do { try process.run() } catch {
      self.process = nil
      busy = false
      self.error = error.localizedDescription
      return
    }
    Task.detached { [weak self] in
      while true {
        let data = output.fileHandleForReading.availableData
        if data.isEmpty { break }
        await self?.consume(data, ticket: ticket)
      }
      process.waitUntilExit()
      await self?.exited(ticket: ticket, status: process.terminationStatus)
    }
    watchdog = Task { [weak self] in
      try? await Task.sleep(for: .seconds(180))
      guard !Task.isCancelled, let self, self.ticket == ticket, !self.running else { return }
      self.error = L(
        "Le serveur n’a pas répondu dans les 180 s.", "Server readiness timed out after 180 s.")
      self.stop()
    }
  }

  private func consume(_ data: Data, ticket: UUID) {
    guard self.ticket == ticket else { return }
    pending.append(data)
    if pending.count > 65_536 {
      error = L("Réponse serveur trop longue.", "Server response too long.")
      pending = Data()
      stop()
      return
    }
    while let end = pending.firstIndex(of: 10) {
      let line = Data(pending[..<end])
      pending.removeSubrange(...end)
      log = String((log + String(decoding: line, as: UTF8.self) + "\n").suffix(16_000))
      if let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
        event["type"] as? String == "ready", let port = event["port"] as? Int,
        (1024...65535).contains(port)
      {
        self.port = port
        running = true
        busy = false
        watchdog?.cancel()
      }
    }
  }

  private func exited(ticket: UUID, status: Int32) {
    guard self.ticket == ticket else { return }
    watchdog?.cancel()
    if status != 0 && !busy {
      error = L("Le serveur s’est arrêté (\(status)).", "Server exited (\(status)).")
    }
    if port == nil && error == nil && status != 0 { error = String(log.suffix(2000)) }
    process = nil
    running = false
    busy = false
    port = nil
    apiKey = nil
  }

  func stop() {
    watchdog?.cancel()
    running = false
    guard let process else {
      busy = false
      return
    }
    busy = true
    if process.isRunning { process.terminate() }
    Task {
      try? await Task.sleep(for: .seconds(3))
      if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }
  }
}

struct ServerWorkspaceView: View {
  @EnvironmentObject private var studio: StudioModel
  @ObservedObject var server: LocalServer
  @State private var port = "8000"

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 22) {
        Label("Server", systemImage: "server.rack").font(.system(size: 32, design: .serif))
        Text(
          L(
            "Une API compatible OpenAI pour vos applications locales.",
            "An OpenAI-compatible API for your local applications.")
        ).foregroundStyle(StudioTheme.secondary)
        GroupBox {
          VStack(alignment: .leading, spacing: 16) {
            LabeledContent(L("Modèle", "Model"), value: studio.selectedModelName ?? "—")
            HStack {
              Text(L("Port local", "Local port"))
              TextField("8000", text: $port).frame(width: 90).disabled(server.active)
              Spacer()
              if server.busy { ProgressView().controlSize(.small) }
              Button(server.active ? L("Arrêter", "Stop") : L("Démarrer", "Start")) {
                if server.active { server.stop() } else { studio.startServer(port: Int(port) ?? 0) }
              }.disabled(!server.active && (studio.isGenerating || studio.selectedModel == nil))
            }
            Text(
              server.running
                ? L("En ligne sur ce Mac", "Online on this Mac")
                : server.busy
                  ? L("Chargement / arrêt…", "Loading / stopping…") : L("Arrêté", "Stopped"))
            Text(
              L(
                "Écoute limitée à 127.0.0.1. Le serveur utilise le modèle à la place de Chat et Agent ; les autres applications locales peuvent appeler cette API.",
                "Listens on 127.0.0.1 only. The server uses the model in place of Chat and Agent; other local applications can call this API."
              )
            ).font(.caption).foregroundStyle(StudioTheme.quiet)
          }.padding(12)
        }
        if let port = server.port, server.running {
          VStack(alignment: .leading, spacing: 12) {
            Text("http://127.0.0.1:\(port)/v1").font(.system(.body, design: .monospaced))
              .textSelection(.enabled)
            Button(L("Copier l’URL", "Copy URL")) {
              NSPasteboard.general.clearContents()
              NSPasteboard.general.setString("http://127.0.0.1:\(port)/v1", forType: .string)
            }
            if let key = server.apiKey {
              Button(L("Copier la clé API", "Copy API key")) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(key, forType: .string)
              }
              Text(
                L(
                  "Une nouvelle clé est générée à chaque démarrage. Ajoutez-la comme Bearer token dans votre application.",
                  "A new key is generated at each start. Use it as the Bearer token in your application."
                )
              ).font(.caption).foregroundStyle(StudioTheme.quiet)
            }
            Text("GET /v1/models · POST /v1/chat/completions").font(.caption.monospaced())
          }
        }
        if let error = server.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
        if !server.log.isEmpty {
          DisclosureGroup(L("Journal du serveur", "Server log")) {
            Text(server.log).font(.caption.monospaced()).textSelection(.enabled).frame(
              maxWidth: .infinity, alignment: .leading)
          }
        }
      }.padding(36).frame(maxWidth: 850, alignment: .leading).frame(maxWidth: .infinity)
    }
  }
}

struct LabWorkspaceView: View {
  @EnvironmentObject private var studio: StudioModel
  @State private var model = ""
  private var samples: [LabSample] {
    studio.labSamples.filter { model.isEmpty || $0.model == model }.reversed()
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 22) {
        Label("Lab", systemImage: "chart.xyaxis.line").font(.system(size: 32, design: .serif))
        TimelineView(.periodic(from: .now, by: 2)) { _ in
          GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 28, verticalSpacing: 12) {
              GridRow {
                Text(L("Matériel", "Hardware"))
                Text("\(LabHardware.chip) · \(LabHardware.machine)")
              }
              GridRow {
                Text("CPU / RAM")
                Text(
                  "\(ProcessInfo.processInfo.processorCount) cores · \(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory))"
                )
              }
              GridRow {
                Text("GPU Metal")
                Text(LabHardware.gpu?.name ?? "—")
              }
              GridRow {
                Text(L("Budget GPU conseillé", "Recommended GPU budget"))
                Text(bytes(LabHardware.gpu?.recommendedBytes))
              }
              GridRow {
                Text("macOS")
                Text(ProcessInfo.processInfo.operatingSystemVersionString)
              }
              GridRow {
                Text("MLXL3 / MLX")
                Text(
                  "\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev") · \(studio.runtimeIdentity ?? "—")"
                )
              }
              GridRow {
                Text(L("RAM interface / Chat-Agent", "UI / Chat-Agent RAM"))
                Text(
                  "\(bytes(studio.interfaceMemoryFootprintBytes)) / \(bytes(studio.engineMemoryFootprintBytes))"
                )
              }
              GridRow {
                Text(L("État thermique / économie d’énergie", "Thermal state / low power"))
                Text(
                  "\(ProcessInfo.processInfo.thermalState.rawValue) / \(ProcessInfo.processInfo.isLowPowerModeEnabled ? "ON" : "OFF")"
                )
              }
            }.font(.system(size: 12)).padding(12)
          }
        }
        HStack {
          Text(L("Mesures par génération", "Measurements by generation")).font(.headline)
          Spacer()
          Picker(L("Modèle", "Model"), selection: $model) {
            Text(L("Tous les modèles", "All models")).tag("")
            ForEach(Array(Set(studio.labSamples.map(\.model))).sorted(), id: \.self) {
              Text($0).tag($0)
            }
          }.frame(maxWidth: 260)
          Button(L("Exporter JSON", "Export JSON"), action: studio.exportLab).disabled(
            studio.labSamples.isEmpty)
        }
        Text(
          L(
            "Mesures réelles de Chat et Agent, conservées avec le modèle, la quantification et les réglages. Le cache, le contexte et la température changent les résultats : ce tableau n’est pas un benchmark contrôlé. Les appels Server ne sont pas inclus.",
            "Actual Chat and Agent measurements, saved with model, quantization and settings. Cache, context and temperature affect results: this table is not a controlled benchmark. Server requests are not included."
          )
        ).font(.caption).foregroundStyle(StudioTheme.quiet)
        if samples.isEmpty {
          ContentUnavailableView(
            L("Pas encore de mesure", "No measurements yet"), systemImage: "chart.bar",
            description: Text(
              L(
                "Générez une réponse dans Chat ou Agent pour alimenter le Lab.",
                "Generate a response in Chat or Agent to populate Lab.")))
        }
        ForEach(samples) { sample in
          VStack(alignment: .leading, spacing: 10) {
            HStack {
              Text(sample.model).font(.headline)
              Text(sample.bits.map { String(format: "%.2f bpw", $0) } ?? "—").foregroundStyle(
                StudioTheme.quiet)
              Spacer()
              Text(sample.date, style: .time).font(.caption)
            }
            HStack(spacing: 28) {
              metric("Decode", sample.stats.decodeTps, "tok/s")
              metric("Prefill", sample.stats.prefillTps, "tok/s")
              metric("TTFT", sample.stats.ttftSeconds * 1000, "ms")
              if let ttft = sample.stats.endToEndTTFTSeconds {
                metric("TTFT E2E", ttft * 1000, "ms")
              }
            }
            Text(
              "\(sample.mode) · \(sample.stats.promptTokens) → \(sample.stats.generatedTokens) tokens · cache \(sample.stats.cachedPromptTokens ?? 0) · ctx \(sample.context) · T \(sample.temperature.formatted()) · MTP \(sample.mtp ? "ON" : "OFF") · DFlash \(sample.dflash ? "ON" : "OFF")"
            ).font(.caption).foregroundStyle(StudioTheme.secondary)
            if let memory = sample.stats.memory {
              Text(
                "MLX peak \(bytes(memory.mlxPeakBytes)) · process \(bytes(memory.processFootprintBytes))"
              ).font(.caption).foregroundStyle(StudioTheme.quiet)
            }
          }.padding(18).frame(maxWidth: .infinity, alignment: .leading).background(
            StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
        }
      }.padding(36).frame(maxWidth: 1000, alignment: .leading).frame(maxWidth: .infinity)
    }
  }

  private func bytes(_ value: UInt64?) -> String {
    value.map { ByteCountFormatter.string(fromByteCount: Int64(clamping: $0), countStyle: .memory) }
      ?? "—"
  }
  private func metric(_ title: String, _ value: Double, _ unit: String) -> some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(title).font(.caption).foregroundStyle(StudioTheme.quiet)
      Text(value, format: .number.precision(.fractionLength(1))).font(
        .system(size: 21, weight: .medium, design: .monospaced))
      Text(unit).font(.caption2).foregroundStyle(StudioTheme.quiet)
    }
  }
}
