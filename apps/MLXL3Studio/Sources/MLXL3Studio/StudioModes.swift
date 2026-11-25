import AppKit
import Darwin
import Metal
import SwiftUI

enum StudioMode: String, CaseIterable, Identifiable, Codable {
  case chat, agent, decision, server, lab
  var id: String { rawValue }
  var title: String { rawValue.capitalized }
  var icon: String {
    switch self {
    case .chat: "bubble.left.and.bubble.right"
    case .agent: "terminal"
    case .decision: "checklist.checked"
    case .server: "server.rack"
    case .lab: "chart.xyaxis.line"
    }
  }
  /// Modes with their own workspace instead of a conversation.
  var standalone: Bool { self == .decision || self == .server || self == .lab }
}

struct LabSample: Codable, Identifiable, Sendable {
  var id = UUID()
  var date = Date()
  var localDay: Int? = nil
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
  private var resolution: Task<Void, Never>?
  var active: Bool { process != nil || resolution != nil }

  /// `mlxl3 decide-serve`: a d1 decision model on `POST /v1/decide`, same readiness and key handling.
  func startDecision(modelPath: String, port: Int, host: String = "127.0.0.1", executable: URL? = nil) {
    start(model: modelPath, port: port, context: 0, host: host, executable: executable, decision: true)
  }

  func start(model: String, port: Int, context: Int, host: String = "127.0.0.1", executable: URL? = nil, decision: Bool = false) {
    guard !active else { return }
    let launchTicket = UUID()
    ticket = launchTicket
    busy = true
    error = nil
    resolution = Task { [weak self] in
      let resolved = await Task.detached(priority: .userInitiated) {
        executable ?? CLIResolver.executable()
      }.value
      guard !Task.isCancelled, let self, self.ticket == launchTicket else { return }
      self.resolution = nil
      self.launch(model: model, port: port, context: context, host: host, executable: resolved, decision: decision)
    }
  }

  private func launch(model: String, port: Int, context: Int, host: String, executable: URL?, decision: Bool) {
    guard (1024...65535).contains(port), (0...Int(Int32.max)).contains(context),
      ["127.0.0.1", "0.0.0.0"].contains(host), let executable else {
      busy = false
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
    let secret = Pipe()
    do {
      try secret.fileHandleForWriting.write(contentsOf: Data(apiKey.utf8))
      try secret.fileHandleForWriting.close()
    } catch {
      busy = false
      self.error = error.localizedDescription
      return
    }
    process.arguments = decision
      ? ["decide-serve", model, "--host", host, "--port", String(port), "--api-key-stdin"]
      : [
        "serve", model, "--host", host, "--port", String(port), "--api-key-stdin",
        "--context-length", String(context),
      ]
    process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
    process.standardInput = secret
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
    resolution?.cancel()
    resolution = nil
    if process == nil { ticket = UUID() }
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
  @AppStorage("studio.server.port") private var port = "8000"
  @AppStorage("studio.server.network") private var network = false
  @AppStorage("studio.server.context") private var context = ""
  private var validPort: Int? { Int(port).flatMap { (1024...65535).contains($0) ? $0 : nil } }
  private var validContext: Int? {
    if context.isEmpty { return studio.savedContextLength }
    return Int(context).flatMap { (0...(studio.modelContextLimit ?? Int(Int32.max))).contains($0) ? $0 : nil }
  }

  private var statusText: String {
    server.running
      ? (network ? L("En ligne sur le réseau local", "Online on the local network") : L("En ligne sur ce Mac", "Online on this Mac"))
      : server.busy
        ? L("Chargement / arrêt…", "Loading / stopping…") : L("Arrêté", "Stopped")
  }

  private var canStart: Bool {
    !(studio.isGenerating || studio.selectedModel == nil || validPort == nil || validContext == nil)
  }

  var body: some View {
    StudioPageColumn(spacing: 22, top: 20) {
      StudioPageHeader(
        title: "Server",
        detail: L("API locale compatible OpenAI.", "Local OpenAI-compatible API.")
      ) {
        StudioStatusPill(
          text: statusText, color: server.running ? StudioTheme.success : StudioTheme.quiet,
          busy: server.busy && !server.running)
      }

      SettingSection(
        title: L("Configuration", "Configuration"), icon: "slider.horizontal.3",
        footer: L(
          "Le serveur utilise le modèle à la place de Chat et Agent.",
          "The server uses the model in place of Chat and Agent.")
      ) {
        SettingRow(
          title: L("Modèle", "Model"),
          detail: studio.selectedModel == nil
            ? L("Choisissez un modèle dans la barre du haut.", "Choose a model in the top bar.") : nil
        ) {
          Text(studio.selectedModelName ?? "—").font(.system(size: 13))
            .foregroundStyle(StudioTheme.secondary).lineLimit(1).truncationMode(.middle)
        }
        SettingRow(
          title: L("Contexte", "Context"),
          detail: L("En tokens. Vide : le réglage du modèle.", "In tokens. Empty: the model’s setting.")
        ) {
          TextField(L("Réglage du modèle", "Model setting"), text: $context)
            .studioField(monospaced: true).frame(width: 150).disabled(server.active)
            .accessibilityLabel(L("Contexte", "Context"))
        }
        SettingRow(
          title: L("Accès", "Access"),
          detail: network
            ? L(
              "Accessible sur le réseau avec la clé API. HTTP n’est pas chiffré.",
              "Accessible on your network with the API key. HTTP is not encrypted.")
            : L("Uniquement depuis ce Mac.", "From this Mac only.")
        ) {
          Picker(L("Accès", "Access"), selection: $network) {
            Text(L("Ce Mac", "This Mac")).tag(false)
            Text(L("Réseau local", "Local network")).tag(true)
          }.pickerStyle(.segmented).labelsHidden().fixedSize().disabled(server.active)
        }
        SettingRow(title: L("Port local", "Local port"), detail: "1024–65535") {
          TextField("8000", text: $port).studioField(monospaced: true).frame(width: 90)
            .disabled(server.active)
            .accessibilityLabel(L("Port local", "Local port"))
        }
      }

      if !server.active && (validPort == nil || validContext == nil) {
        StudioNotice(
          kind: .error,
          text: L("Vérifiez le port et le contexte du modèle.", "Check the port and model context."))
      }
      if let error = server.error { StudioNotice(kind: .error, text: error) }

      HStack(spacing: 12) {
        Button {
          if server.active { server.stop() } else {
            studio.startServer(
              port: validPort ?? 0, host: network ? "0.0.0.0" : "127.0.0.1", context: validContext)
          }
        } label: {
          Label(
            server.active ? L("Arrêter", "Stop") : L("Démarrer", "Start"),
            systemImage: server.active ? "stop.fill" : "play.fill")
        }
        .buttonStyle(.studio(server.active ? .secondary : .primary, height: 34))
        .disabled(!server.active && !canStart)
        if server.busy { ProgressView().controlSize(.small) }
        Spacer(minLength: 0)
      }

      if let port = server.port, server.running {
        let url = "http://127.0.0.1:\(port)/v1"
        SettingSection(
          title: L("Connexion", "Connection"), icon: "link",
          footer: network
            ? L(
              "Depuis un autre appareil, remplacez 127.0.0.1 par l’adresse réseau de ce Mac.",
              "From another device, replace 127.0.0.1 with this Mac’s network address.")
            : nil
        ) {
          StudioCopyField(label: L("URL de base", "Base URL"), value: url)
          if let key = server.apiKey {
            StudioCopyField(label: L("Clé API", "API key"), value: key, secret: true)
            StudioCopyField(
              label: L("Commande de test", "Test command"),
              value: "curl -H 'Authorization: Bearer \(key)' \(url)/models",
              shown: "curl -H 'Authorization: Bearer ••••' \(url)/models")
          }
          SettingRow(
            title: L("Points d’accès", "Endpoints"),
            detail: server.apiKey == nil
              ? nil
              : L(
                "Une nouvelle clé est générée à chaque démarrage. Ajoutez-la comme Bearer token dans votre application.",
                "A new key is generated at each start. Use it as the Bearer token in your application.")
          ) {
            Text("GET /v1/models\nPOST /v1/chat/completions")
              .font(.system(size: 11.5, design: .monospaced)).foregroundStyle(StudioTheme.secondary)
              .multilineTextAlignment(.trailing).textSelection(.enabled)
          }
        }
      }

      if !server.log.isEmpty {
        DisclosureGroup(L("Journal du serveur", "Server log")) {
          ScrollView {
            Text(server.log).font(.system(size: 11.5, design: .monospaced)).textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
          .frame(maxHeight: 220)
          .studioCard(padding: 12)
          .padding(.top, 8)
        }
        .font(.system(size: 12.5, weight: .medium)).foregroundStyle(StudioTheme.secondary)
      }
    }
  }
}
