import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

// Liquid d1 decision models: typed answers read off one forward pass, no generated text.
// The Playground drives `mlxl3 decide-bridge`; the Server tab runs `mlxl3 decide-serve`.

enum DecisionQuestionKind: String, CaseIterable, Codable, Identifiable, Sendable {
  case noul, choice, multiple, score, text, number, date, estimate, box, boxes
  var id: String { rawValue }
  var title: String {
    switch self {
    case .noul: L("Oui / non", "Yes / no")
    case .choice: L("Choix", "Choice")
    case .multiple: L("Choix multiple", "Multiple choice")
    case .score: L("Score", "Score")
    case .text: L("Texte lu", "Read text")
    case .number: L("Nombre lu", "Read number")
    case .date: L("Date lue", "Read date")
    case .estimate: L("Estimation", "Estimate")
    case .box: L("Boîte", "Box")
    case .boxes: L("Boîtes", "Boxes")
    }
  }
  var usesOptions: Bool { self == .choice || self == .multiple || self == .score }
  var needsPicture: Bool { self == .box || self == .boxes }
}

struct DecisionOption: Codable, Hashable, Identifiable, Sendable {
  var id = UUID()
  var name: String
  var detail: String = ""
}

struct DecisionQuestion: Codable, Hashable, Identifiable, Sendable {
  var id = UUID()
  var name: String
  var kind: DecisionQuestionKind
  var instructions: String
  /// A spoken question's clip (16-bit PCM WAV); replaces the written instructions.
  var spokenPath: String?
  var options: [DecisionOption] = []
  var yesMeans = ""
  var noMeans = ""
  var low = "0"
  var high = "10"
  var step = "1"
  var unit = ""
  var maxBoxes = ""

  /// The question in the API's JSON. Throws a message naming what to fix.
  func json() throws -> [String: Any] {
    let key = name.trimmingCharacters(in: .whitespaces)
    guard !key.isEmpty else { throw DecisionDraftError(L("Chaque question a un nom.", "Every question needs a name.")) }
    var out: [String: Any] = [:]
    if let spokenPath {
      out["instructions"] = ["type": "audio", "path": spokenPath]
    } else {
      guard !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw DecisionDraftError(L("« \(key) » n’a pas de consigne.", "“\(key)” has no instructions."))
      }
      out["instructions"] = instructions
    }
    let named = options.filter { !$0.name.trimmingCharacters(in: .whitespaces).isEmpty }
    switch kind {
    case .noul:
      out["type"] = "noul"
      if !yesMeans.isEmpty || !noMeans.isEmpty { out["criteria"] = ["true": yesMeans, "false": noMeans] }
    case .choice, .multiple:
      guard !named.isEmpty else { throw DecisionDraftError(L("« \(key) » n’a pas d’option.", "“\(key)” has no options.")) }
      guard Set(named.map(\.name)).count == named.count else {
        throw DecisionDraftError(L("« \(key) » a deux options du même nom.", "“\(key)” has two options with the same name."))
      }
      out["type"] = "choice"
      if kind == .multiple { out["multiple"] = true }
      var criteria = [String: Any]()
      for option in named { criteria[option.name] = option.detail.isEmpty ? NSNull() : option.detail }
      out["criteria"] = OrderedJSONObject(keys: named.map(\.name), values: criteria)
    case .score:
      guard (2...10).contains(named.count) else {
        throw DecisionDraftError(L("Un score a de 2 à 10 niveaux.", "A score has 2 to 10 levels."))
      }
      out["type"] = "score"
      out["criteria"] = named.map(\.name)
    case .text, .date:
      out["type"] = kind.rawValue
    case .number:
      out["type"] = "number"
      if !unit.isEmpty { out["unit"] = unit }
    case .estimate:
      guard let lowValue = Double(low.replacingOccurrences(of: ",", with: ".")),
        let highValue = Double(high.replacingOccurrences(of: ",", with: ".")), lowValue < highValue
      else { throw DecisionDraftError(L("« \(key) » : l’intervalle va du bas au haut.", "“\(key)”: the range runs low to high.")) }
      out["type"] = "number"
      out["range"] = [lowValue, highValue]
      if !step.isEmpty {
        guard let stepValue = Double(step.replacingOccurrences(of: ",", with: ".")), stepValue > 0 else {
          throw DecisionDraftError(L("« \(key) » : le pas est un nombre positif.", "“\(key)”: the step is a positive number."))
        }
        out["step"] = stepValue
      }
      if !unit.isEmpty { out["unit"] = unit }
    case .box:
      out["type"] = "box"
    case .boxes:
      out["type"] = "boxes"
      if !maxBoxes.isEmpty {
        guard let value = Int(maxBoxes), (1...16).contains(value) else {
          throw DecisionDraftError(L("« \(key) » : au plus 1 à 16 boîtes.", "“\(key)”: at most 1 to 16 boxes."))
        }
        out["max"] = value
      }
    }
    return out
  }
}

struct DecisionDraftError: LocalizedError, Sendable {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}

/// A JSON object that keeps its keys in order: d1 reads options in the order they are listed.
struct OrderedJSONObject {
  let keys: [String]
  let values: [String: Any]
}

enum DecisionMediaKind: String, Codable, Sendable { case image, audio }

struct DecisionMediaItem: Codable, Hashable, Identifiable, Sendable {
  var id = UUID()
  var kind: DecisionMediaKind
  var path: String
  var label: String
}

struct DecisionDraft: Codable, Hashable, Sendable {
  var state = ""
  var stateIsJSON = false
  var media: [DecisionMediaItem] = []
  var questions: [DecisionQuestion] = []

  static let example = DecisionDraft(
    state: "I was charged twice and my parcel never arrived. Please refund one charge.",
    questions: [
      DecisionQuestion(name: "refund", kind: .noul, instructions: "Is the customer asking for a refund?"),
      DecisionQuestion(
        name: "team", kind: .choice, instructions: "Which team should handle this?",
        options: [
          DecisionOption(name: "billing", detail: "Charges, refunds, invoices"),
          DecisionOption(name: "shipping", detail: "Delivery and parcels"),
          DecisionOption(name: "technical", detail: "App or site faults"),
        ]),
      DecisionQuestion(
        name: "urgency", kind: .score, instructions: "How urgent is this ticket?",
        options: [DecisionOption(name: "low"), DecisionOption(name: "medium"), DecisionOption(name: "high")]),
    ])

  /// The bridge request `{"state", "questions", "media"}`, with local paths.
  func request() throws -> Data {
    guard !questions.isEmpty else { throw DecisionDraftError(L("Ajoutez au moins une question.", "Add at least one question.")) }
    let names = questions.map { $0.name.trimmingCharacters(in: .whitespaces) }
    guard Set(names).count == names.count else {
      throw DecisionDraftError(L("Deux questions portent le même nom.", "Two questions have the same name."))
    }
    let pictures = media.filter { $0.kind == .image }.count
    if questions.contains(where: { $0.kind.needsPicture }) && pictures != 1 {
      throw DecisionDraftError(L("Une question boîte lit exactement une image.", "A box question reads exactly one picture."))
    }
    let trimmed = state.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty || !media.isEmpty else {
      throw DecisionDraftError(L("Donnez un état (texte) ou un média.", "Give a state (text) or a media item."))
    }
    var stateValue: Any = NSNull()
    if !trimmed.isEmpty {
      if stateIsJSON {
        do {
          stateValue = try JSONSerialization.jsonObject(with: Data(state.utf8), options: [.fragmentsAllowed])
        } catch { throw DecisionDraftError(L("L’état n’est pas du JSON valide.", "The state is not valid JSON.")) }
      } else {
        stateValue = state
      }
    }
    var text = "{\"state\":" + (try DecisionJSON.encode(stateValue))
    text += ",\"questions\":{"
    for (index, question) in questions.enumerated() {
      if index > 0 { text += "," }
      text += try DecisionJSON.encode(names[index]) + ":" + DecisionJSON.encode(try question.json())
    }
    text += "}"
    if !media.isEmpty {
      text += ",\"media\":" + (try DecisionJSON.encode(media.map { ["type": $0.kind.rawValue, "path": $0.path] }))
    }
    return Data((text + "}").utf8)
  }
}

/// JSON text that keeps `OrderedJSONObject` key order (JSONSerialization sorts nothing but loses order).
enum DecisionJSON {
  static func encode(_ value: Any) throws -> String {
    switch value {
    case let object as OrderedJSONObject:
      return "{" + (try object.keys.map { try encode($0) + ":" + encode(object.values[$0] ?? NSNull()) }.joined(separator: ",")) + "}"
    case let dictionary as [String: Any]:
      return "{" + (try dictionary.keys.sorted().map { try encode($0) + ":" + encode(dictionary[$0]!) }.joined(separator: ",")) + "}"
    case let array as [Any]:
      return "[" + (try array.map(encode).joined(separator: ",")) + "]"
    default:
      let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes])
      return String(decoding: data, as: UTF8.self)
    }
  }
}

/// A decision model on disk: a folder whose config records `d1_attention`.
struct DecisionModelEntry: Hashable, Identifiable, Sendable {
  let name: String
  let path: String
  let audio: Bool
  let custom: Bool
  var id: String { path }

  static func inspect(path: String, name: String? = nil, custom: Bool) -> DecisionModelEntry? {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url.appending(path: "config.json")),
      let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      config["d1_attention"] != nil,
      FileManager.default.fileExists(atPath: url.appending(path: "quantization_config.json").path)
    else { return nil }
    return DecisionModelEntry(
      name: name ?? url.lastPathComponent, path: url.path,
      audio: (config["model_type"] as? String) == "lfm3_omni", custom: custom)
  }
}

/// Media files as the engine reads them: pictures as PNG/JPEG/WebP, sound as 16-bit PCM WAV.
enum DecisionMediaImport {
  static var folder: URL {
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    return root.appending(path: "io.mlxl3.desktop/Decision/Media", directoryHint: .isDirectory)
  }

  static func prepare(_ url: URL) throws -> DecisionMediaItem {
    let type = UTType(filenameExtension: url.pathExtension.lowercased())
    if let type, type.conforms(to: .image) {
      if [UTType.png, .jpeg, .webP].contains(where: { type.conforms(to: $0) }) {
        return DecisionMediaItem(kind: .image, path: url.path, label: url.lastPathComponent)
      }
      guard let image = NSImage(contentsOf: url), let tiff = image.tiffRepresentation,
        let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:])
      else { throw DecisionDraftError(L("Image illisible.", "Unreadable picture.")) }
      let out = try destination(extension: "png")
      try png.write(to: out)
      return DecisionMediaItem(kind: .image, path: out.path, label: url.lastPathComponent)
    }
    return DecisionMediaItem(kind: .audio, path: try normalizedAudio(url).path, label: url.lastPathComponent)
  }

  static func destination(extension ext: String) throws -> URL {
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder.appending(path: UUID().uuidString + "." + ext)
  }

  /// Any sound file Core Audio reads, as 16-bit PCM WAV at its own rate and channels: the engine then
  /// mixes and resamples it to 16 kHz exactly as the reference does. A 16-bit WAV passes through.
  static func normalizedAudio(_ url: URL) throws -> URL {
    let input: AVAudioFile
    do { input = try AVAudioFile(forReading: url) } catch {
      throw DecisionDraftError(L("Fichier audio illisible.", "Unreadable audio file."))
    }
    let file = input.fileFormat.settings
    if url.pathExtension.lowercased() == "wav", (file[AVLinearPCMBitDepthKey] as? Int) == 16,
      (file[AVLinearPCMIsFloatKey] as? Bool) != true, (file[AVFormatIDKey] as? UInt32) == kAudioFormatLinearPCM
    {
      return url
    }
    let format = input.processingFormat
    guard format.channelCount >= 1, format.channelCount <= 8 else {
      throw DecisionDraftError(L("Audio : 1 à 8 canaux.", "Audio: 1 to 8 channels."))
    }
    guard Double(input.length) / format.sampleRate <= 300 else {
      throw DecisionDraftError(L("Un clip audio dure au plus 5 minutes.", "An audio clip is at most five minutes long."))
    }
    let out = try destination(extension: "wav")
    let settings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: format.sampleRate,
      AVNumberOfChannelsKey: format.channelCount, AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
    ]
    let output = try AVAudioFile(
      forWriting: out, settings: settings, commonFormat: format.commonFormat, interleaved: format.isInterleaved)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 32_768) else {
      throw DecisionDraftError(L("Conversion audio impossible.", "Audio conversion failed."))
    }
    while input.framePosition < input.length {
      try input.read(into: buffer)
      if buffer.frameLength == 0 { break }
      try output.write(from: buffer)
    }
    return out
  }
}

@MainActor
final class DecisionStudio: ObservableObject {
  enum EngineState: Equatable {
    case idle, loading, ready, failed(String)
  }

  struct Capabilities: Equatable {
    var model: String
    var images: Bool
    var audio: Bool
    var setSize: Int
  }

  struct Outcome: Equatable {
    var answers: [String: NSDictionary]
    var order: [String]
    var inputTokens: Int
    var elapsedMilliseconds: Double
    var raw: String
  }

  @Published var draft: DecisionDraft { didSet { persist() } }
  @Published private(set) var models: [DecisionModelEntry] = []
  @Published var selectedPath: String? {
    didSet {
      preferences.set(selectedPath, forKey: Self.modelKey)
      if selectedPath != loadedPath { unload() }
    }
  }
  @Published private(set) var state: EngineState = .idle
  @Published private(set) var capabilities: Capabilities?
  @Published private(set) var running = false
  @Published private(set) var outcome: Outcome?
  @Published var error: String?
  @Published private(set) var recordingTarget: UUID?
  let server = LocalServer()

  /// Frees the chat model before a decision model loads; set by `StudioModel`.
  var prepareForLoad: (() async -> Void)?
  var executable: () -> URL? = { CLIResolver.executable() }

  private let preferences: UserDefaults
  private var process: Process?
  private var input: FileHandle?
  private var buffer = Data()
  private var stderrTail = ""
  private var loadedPath: String?
  private var ticket = UUID()
  private var nextID = 1
  private var waiting: [Int: CheckedContinuation<Data, Error>] = [:]
  private var readyWaiters: [CheckedContinuation<Void, Error>] = []
  private var recorder: AVAudioRecorder?
  private var recordingURL: URL?
  private static let draftKey = "decision.draft"
  private static let modelKey = "decision.model"
  private static let foldersKey = "decision.folders"
  static let bridgeLineLimit = 96 * 1024 * 1024

  init(preferences: UserDefaults = .standard) {
    self.preferences = preferences
    if let data = preferences.data(forKey: Self.draftKey),
      let saved = try? JSONDecoder().decode(DecisionDraft.self, from: data)
    {
      draft = saved
    } else {
      draft = .example
    }
    selectedPath = preferences.string(forKey: Self.modelKey)
  }

  var active: Bool { process != nil }
  var selectedModel: DecisionModelEntry? { models.first { $0.path == selectedPath } }

  private func persist() {
    if let data = try? JSONEncoder().encode(draft) { preferences.set(data, forKey: Self.draftKey) }
  }

  // MARK: models

  /// Decision models among the registered ones, plus folders the user added.
  func refresh(registered: [LocalModel]) {
    var found = registered.compactMap { DecisionModelEntry.inspect(path: $0.path, name: $0.name, custom: false) }
    for path in preferences.stringArray(forKey: Self.foldersKey) ?? [] where !found.contains(where: { $0.path == path }) {
      if let entry = DecisionModelEntry.inspect(path: path, custom: true) { found.append(entry) }
    }
    models = found
    if selectedPath == nil || !found.contains(where: { $0.path == selectedPath }) { selectedPath = found.first?.path }
  }

  @discardableResult
  func addFolder(_ url: URL, registered: [LocalModel]) -> Bool {
    guard let entry = DecisionModelEntry.inspect(path: url.path, custom: true) else {
      error = L(
        "Ce dossier n’est pas un modèle de décision EXL3 (config sans d1_attention).",
        "This folder is not an EXL3 decision model (no d1_attention in its config).")
      return false
    }
    var folders = preferences.stringArray(forKey: Self.foldersKey) ?? []
    if !folders.contains(entry.path) { folders.append(entry.path) }
    preferences.set(folders, forKey: Self.foldersKey)
    refresh(registered: registered)
    selectedPath = entry.path
    error = nil
    return true
  }

  func removeFolder(_ entry: DecisionModelEntry, registered: [LocalModel]) {
    preferences.set((preferences.stringArray(forKey: Self.foldersKey) ?? []).filter { $0 != entry.path }, forKey: Self.foldersKey)
    if selectedPath == entry.path { selectedPath = nil }
    refresh(registered: registered)
  }

  // MARK: engine

  func load() async throws {
    guard let path = selectedPath else {
      throw DecisionDraftError(L("Choisissez un modèle de décision.", "Choose a decision model."))
    }
    if loadedPath == path, process != nil {
      if state == .ready { return }
      return try await withCheckedThrowingContinuation { readyWaiters.append($0) }
    }
    unload()
    if server.active { server.stop() }
    await prepareForLoad?()
    guard let executable = executable() else {
      state = .failed(L("Moteur MLXL3 introuvable.", "MLXL3 engine not found."))
      throw DecisionDraftError(L("Moteur MLXL3 introuvable.", "MLXL3 engine not found."))
    }
    let process = Process()
    let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
    process.executableURL = executable
    process.arguments = ["decide-bridge", path]
    process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
    process.standardInput = stdin
    process.standardOutput = stdout
    process.standardError = stderr
    let ticket = UUID()
    self.ticket = ticket
    loadedPath = path
    state = .loading
    capabilities = nil
    buffer = Data()
    stderrTail = ""
    do { try process.run() } catch {
      loadedPath = nil
      state = .failed(error.localizedDescription)
      throw error
    }
    self.process = process
    input = stdin.fileHandleForWriting
    Task.detached { [weak self] in
      while true {
        let data = stdout.fileHandleForReading.availableData
        if data.isEmpty { break }
        await self?.consume(data, ticket: ticket)
      }
      process.waitUntilExit()
      await self?.exited(ticket: ticket, status: process.terminationStatus)
    }
    Task.detached { [weak self] in
      while true {
        let data = stderr.fileHandleForReading.availableData
        if data.isEmpty { break }
        await self?.appendStderr(data, ticket: ticket)
      }
    }
    try await withCheckedThrowingContinuation { readyWaiters.append($0) }
  }

  func unload() {
    ticket = UUID()
    let message = L("Moteur de décision arrêté.", "Decision engine stopped.")
    failAll(DecisionDraftError(message))
    if let process, process.isRunning {
      try? input?.close()
      process.terminate()
      Task {
        try? await Task.sleep(for: .seconds(3))
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      }
    }
    process = nil
    input = nil
    loadedPath = nil
    capabilities = nil
    state = .idle
  }

  func shutdown() {
    stopRecording(keep: false)
    unload()
    server.stop()
  }

  private func failAll(_ error: Error) {
    for (_, waiter) in waiting { waiter.resume(throwing: error) }
    waiting.removeAll()
    for waiter in readyWaiters { waiter.resume(throwing: error) }
    readyWaiters.removeAll()
  }

  private func appendStderr(_ data: Data, ticket: UUID) {
    guard self.ticket == ticket else { return }
    stderrTail = String((stderrTail + String(decoding: data, as: UTF8.self)).suffix(4_000))
  }

  private func consume(_ data: Data, ticket: UUID) {
    guard self.ticket == ticket else { return }
    buffer.append(data)
    if buffer.count > Self.bridgeLineLimit {
      fail(L("Réponse du moteur trop longue.", "Engine response too long."))
      return
    }
    while let end = buffer.firstIndex(of: 10) {
      let line = Data(buffer[buffer.startIndex..<end])
      buffer.removeSubrange(buffer.startIndex...end)
      handle(line)
    }
  }

  private func handle(_ line: Data) {
    guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
      let type = event["type"] as? String
    else { return }
    switch type {
    case "ready":
      capabilities = Capabilities(
        model: event["model"] as? String ?? "", images: event["images"] as? Bool ?? false,
        audio: event["audio"] as? Bool ?? false, setSize: event["set_size"] as? Int ?? 16)
      state = .ready
      let waiters = readyWaiters
      readyWaiters.removeAll()
      for waiter in waiters { waiter.resume() }
    case "result", "error":
      guard let id = event["id"] as? Int, let waiter = waiting.removeValue(forKey: id) else {
        if type == "error", state == .loading { fail(event["message"] as? String ?? "load failed") }
        return
      }
      if type == "result" { waiter.resume(returning: line) } else {
        waiter.resume(throwing: DecisionDraftError(event["message"] as? String ?? L("Échec de la décision.", "Decision failed.")))
      }
    default:
      break
    }
  }

  private func fail(_ message: String) {
    let process = self.process
    ticket = UUID()
    failAll(DecisionDraftError(message))
    state = .failed(message)
    self.process = nil
    input = nil
    loadedPath = nil
    if let process, process.isRunning { process.terminate() }
  }

  private func exited(ticket: UUID, status: Int32) {
    guard self.ticket == ticket else { return }
    let detail = stderrTail.trimmingCharacters(in: .whitespacesAndNewlines)
    let message = detail.isEmpty
      ? L("Le moteur de décision s’est arrêté (\(status)).", "The decision engine exited (\(status)).")
      : String(detail.suffix(1_200))
    fail(message)
  }

  /// One request on the loaded bridge; loads the selected model first when needed.
  func run() {
    guard !running else { return }
    let request: Data
    do { request = try draft.request() } catch {
      self.error = error.localizedDescription
      return
    }
    error = nil
    running = true
    Task {
      defer { running = false }
      do {
        try await load()
        if let capabilities {
          let needsAudio = draft.media.contains { $0.kind == .audio } || draft.questions.contains { $0.spokenPath != nil }
          if needsAudio && !capabilities.audio {
            throw DecisionDraftError(L("Ce modèle ne prend pas l’audio : choisissez d1-Omni.", "This model takes no audio: choose d1-Omni."))
          }
        }
        let reply = try await send(request)
        outcome = try Self.parse(reply)
      } catch {
        self.error = error.localizedDescription
      }
    }
  }

  private func send(_ request: Data) async throws -> Data {
    guard let input else { throw DecisionDraftError(L("Moteur non chargé.", "Engine not loaded.")) }
    let id = nextID
    nextID += 1
    var line = Data("{\"id\":\(id),\"request\":".utf8)
    line.append(request)
    line.append(Data("}\n".utf8))
    guard line.count <= Self.bridgeLineLimit else {
      throw DecisionDraftError(L("Requête trop grande (96 Mo).", "Request too large (96 MB)."))
    }
    return try await withCheckedThrowingContinuation { continuation in
      waiting[id] = continuation
      do { try input.write(contentsOf: line) } catch {
        waiting.removeValue(forKey: id)
        continuation.resume(throwing: error)
        return
      }
      Task { [weak self] in
        try? await Task.sleep(for: .seconds(600))
        guard let self, let waiter = self.waiting.removeValue(forKey: id) else { return }
        waiter.resume(throwing: DecisionDraftError(L("Pas de réponse après 10 min.", "No answer after 10 min.")))
      }
    }
  }

  /// The keys of the JSON object at `path`, in the order the text writes them (JSONSerialization loses it).
  static func objectKeys(in bytes: [UInt8], path: [String]) -> [String] {
    var index = 0
    func skipSpace() { while index < bytes.count, [9, 10, 13, 32].contains(bytes[index]) { index += 1 } }
    func string() -> String? {
      guard index < bytes.count, bytes[index] == 34 else { return nil }
      let start = index
      index += 1
      while index < bytes.count, bytes[index] != 34 { index += bytes[index] == 92 ? 2 : 1 }
      index += 1
      guard index <= bytes.count else { return nil }
      return (try? JSONSerialization.jsonObject(with: Data(bytes[start..<index]), options: .fragmentsAllowed)) as? String
    }
    func skipValue() {
      skipSpace()
      guard index < bytes.count else { return }
      if bytes[index] == 34 { _ = string(); return }
      if bytes[index] == 123 || bytes[index] == 91 {
        var depth = 0
        while index < bytes.count {
          switch bytes[index] {
          case 34: _ = string(); continue
          case 123, 91: depth += 1
          case 125, 93: depth -= 1
          default: break
          }
          index += 1
          if depth == 0 { return }
        }
        return
      }
      while index < bytes.count, ![44, 125, 93].contains(bytes[index]) { index += 1 }
    }
    /// Walks one object's members; `body` returns true to descend into the member's value.
    func members(_ body: (String) -> Bool) -> Bool {
      skipSpace()
      guard index < bytes.count, bytes[index] == 123 else { return false }
      index += 1
      while true {
        skipSpace()
        if index < bytes.count, bytes[index] == 125 { index += 1; return true }
        guard let key = string() else { return false }
        skipSpace()
        guard index < bytes.count, bytes[index] == 58 else { return false }
        index += 1
        if body(key) { return true }
        skipValue()
        skipSpace()
        if index < bytes.count, bytes[index] == 44 { index += 1 }
      }
    }
    var keys: [String] = []
    func descend(_ remaining: ArraySlice<String>) {
      if remaining.isEmpty {
        _ = members { keys.append($0); return false }
        return
      }
      _ = members { key in
        guard key == remaining.first else { return false }
        descend(remaining.dropFirst())
        return true
      }
    }
    descend(path[...])
    return keys
  }

  static func parse(_ line: Data) throws -> Outcome {
    guard let event = try JSONSerialization.jsonObject(with: line) as? [String: Any],
      let result = event["result"] as? [String: Any], let answers = result["answers"] as? [String: Any]
    else { throw DecisionDraftError(L("Réponse du moteur invalide.", "Invalid engine answer.")) }
    // Keep the request's question order: the bridge writes answers in that order.
    let text = String(decoding: line, as: UTF8.self)
    let listed = objectKeys(in: Array(line), path: ["result", "answers"])
    let order = listed.filter { answers[$0] != nil } + answers.keys.filter { !listed.contains($0) }.sorted()
    let usage = result["usage"] as? [String: Any]
    let pretty = (try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]))
      .map { String(decoding: $0, as: UTF8.self) } ?? text
    return Outcome(
      answers: answers.compactMapValues { $0 as? NSDictionary },
      order: order, inputTokens: usage?["input_tokens"] as? Int ?? 0,
      elapsedMilliseconds: event["elapsed_ms"] as? Double ?? 0, raw: pretty)
  }

  // MARK: media

  func importMedia(_ urls: [URL], spokenFor question: UUID? = nil) {
    for url in urls {
      let scoped = url.startAccessingSecurityScopedResource()
      defer { if scoped { url.stopAccessingSecurityScopedResource() } }
      do {
        let item = try DecisionMediaImport.prepare(url)
        if let question, let index = draft.questions.firstIndex(where: { $0.id == question }) {
          guard item.kind == .audio else { throw DecisionDraftError(L("Une question parlée est un clip audio.", "A spoken question is an audio clip.")) }
          draft.questions[index].spokenPath = item.path
        } else {
          guard draft.media.count < 8 else { throw DecisionDraftError(L("8 médias au plus.", "At most 8 media items.")) }
          draft.media.append(item)
        }
        error = nil
      } catch { self.error = error.localizedDescription }
    }
  }

  /// Records a 16 kHz mono 16-bit WAV into the state's media, or as a question's spoken instructions.
  func toggleRecording(for question: UUID? = nil) {
    let target = question ?? DecisionStudio.stateTarget
    if recordingTarget != nil {
      let same = recordingTarget == target
      stopRecording(keep: true)
      if same { return }
    }
    Task {
      guard await AVCaptureDevice.requestAccess(for: .audio) else {
        error = L("Accès au micro refusé (Réglages Système › Confidentialité).", "Microphone access denied (System Settings › Privacy).")
        return
      }
      do {
        let url = try DecisionMediaImport.destination(extension: "wav")
        let recorder = try AVAudioRecorder(url: url, settings: [
          AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16_000, AVNumberOfChannelsKey: 1,
          AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
        ])
        guard recorder.record(forDuration: 300) else { throw DecisionDraftError(L("Enregistrement impossible.", "Recording failed.")) }
        self.recorder = recorder
        recordingURL = url
        recordingTarget = target
      } catch { self.error = error.localizedDescription }
    }
  }

  static let stateTarget = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

  func stopRecording(keep: Bool) {
    guard let recorder, let url = recordingURL else { return }
    recorder.stop()
    let target = recordingTarget
    self.recorder = nil
    recordingURL = nil
    recordingTarget = nil
    guard keep else { return }
    let label = L("Enregistrement", "Recording") + " " + Date().formatted(date: .omitted, time: .shortened)
    if target == Self.stateTarget {
      draft.media.append(DecisionMediaItem(kind: .audio, path: url.path, label: label))
    } else if let index = draft.questions.firstIndex(where: { $0.id == target }) {
      draft.questions[index].spokenPath = url.path
    }
  }

  // MARK: server

  func startServer(port: Int, network: Bool) {
    guard let path = selectedPath else {
      error = L("Choisissez un modèle de décision.", "Choose a decision model.")
      return
    }
    unload()
    Task {
      await prepareForLoad?()
      server.startDecision(modelPath: path, port: port, host: network ? "0.0.0.0" : "127.0.0.1", executable: executable())
    }
  }

  /// The current draft as an HTTP request body: media inlined as data: URLs.
  func serverBody() throws -> String {
    let data = try draft.request()
    guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return "" }
    func inline(_ item: [String: Any]) throws -> [String: Any] {
      guard let type = item["type"] as? String, let path = item["path"] as? String else { return item }
      let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
      let mime = type == "audio" ? "audio/wav" : (UTType(filenameExtension: URL(fileURLWithPath: path).pathExtension)?.preferredMIMEType ?? "image/png")
      return ["type": type, "url": "data:\(mime);base64,\(bytes.base64EncodedString())"]
    }
    if let media = object["media"] as? [[String: Any]] { object["media"] = try media.map(inline) }
    if var questions = object["questions"] as? [String: Any] {
      for (key, value) in questions {
        if var question = value as? [String: Any], let spoken = question["instructions"] as? [String: Any] {
          question["instructions"] = try inline(spoken)
          questions[key] = question
        }
      }
      object["questions"] = questions
    }
    // Option order matters: rebuild through the ordered encoder from the draft's own order.
    let names = draft.questions.map { $0.name.trimmingCharacters(in: .whitespaces) }
    var text = "{\"state\":" + (try DecisionJSON.encode(object["state"] ?? NSNull())) + ",\"questions\":{"
    for (index, question) in draft.questions.enumerated() {
      if index > 0 { text += "," }
      var json = try question.json()
      if let spoken = (object["questions"] as? [String: Any])?[names[index]] as? [String: Any],
        let instructions = spoken["instructions"] as? [String: Any]
      {
        json["instructions"] = instructions
      }
      text += try DecisionJSON.encode(names[index]) + ":" + DecisionJSON.encode(json)
    }
    text += "}"
    if let media = object["media"] { text += ",\"media\":" + (try DecisionJSON.encode(media)) }
    return text + "}"
  }
}

// MARK: - Views

struct DecisionWorkspaceView: View {
  @EnvironmentObject private var studio: StudioModel
  @ObservedObject var decision: DecisionStudio
  @AppStorage("decision.tab") private var tab = "playground"

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        HStack(alignment: .firstTextBaseline) {
          Label(L("Décision", "Decision"), systemImage: "checklist.checked").font(.system(size: 32, design: .serif))
          Spacer()
          Picker("", selection: $tab) {
            Text("Playground").tag("playground")
            Text("Server").tag("server")
          }
          .pickerStyle(.segmented).frame(width: 220).labelsHidden()
          .accessibilityIdentifier("decision.tab")
        }
        Text(
          L(
            "Modèles de décision Liquid d1 : réponses typées et calibrées en une seule passe, sans génération de texte.",
            "Liquid d1 decision models: typed, calibrated answers in one forward pass, with no generated text.")
        ).foregroundStyle(StudioTheme.secondary)
        DecisionModelBar(decision: decision)
        if tab == "server" {
          DecisionServerView(decision: decision, server: decision.server)
        } else {
          DecisionPlaygroundView(decision: decision)
        }
        if let error = decision.error {
          Text(error).foregroundStyle(.red).textSelection(.enabled).accessibilityIdentifier("decision.error")
        }
      }
      .padding(36).frame(maxWidth: 980, alignment: .leading).frame(maxWidth: .infinity)
    }
    .onAppear { decision.refresh(registered: studio.models) }
  }
}

private struct DecisionModelBar: View {
  @EnvironmentObject private var studio: StudioModel
  @ObservedObject var decision: DecisionStudio

  var body: some View {
    GroupBox {
      HStack(spacing: 12) {
        if decision.models.isEmpty {
          Text(
            L(
              "Aucun modèle de décision. Ajoutez le dossier d’un export EXL3 de d1-2B ou d1-Omni-2B.",
              "No decision model. Add the folder of a d1-2B or d1-Omni-2B EXL3 export.")
          ).foregroundStyle(StudioTheme.secondary)
        } else {
          Picker(L("Modèle", "Model"), selection: $decision.selectedPath) {
            ForEach(decision.models) { model in
              Text(model.name + (model.audio ? " · " + L("texte, images, audio", "text, pictures, audio") : " · " + L("texte, images", "text, pictures")))
                .tag(Optional(model.path))
            }
          }
          .frame(maxWidth: 460).accessibilityIdentifier("decision.model")
        }
        Spacer()
        switch decision.state {
        case .loading: ProgressView().controlSize(.small)
        case .ready: Label(L("Chargé", "Loaded"), systemImage: "checkmark.circle").foregroundStyle(StudioTheme.secondary)
        default: EmptyView()
        }
        if decision.active {
          Button(L("Décharger", "Unload")) { decision.unload() }
        }
        Button(L("Ajouter un dossier…", "Add folder…")) {
          let panel = NSOpenPanel()
          panel.canChooseDirectories = true
          panel.canChooseFiles = false
          panel.prompt = L("Ajouter", "Add")
          if panel.runModal() == .OK, let url = panel.url { decision.addFolder(url, registered: studio.models) }
        }
        if let model = decision.selectedModel, model.custom {
          Button { decision.removeFolder(model, registered: studio.models) } label: { Image(systemName: "minus.circle") }
            .buttonStyle(.borderless).help(L("Retirer ce dossier de la liste", "Remove this folder from the list"))
        }
      }.padding(8)
    }
    if case let .failed(message) = decision.state {
      Text(message).font(.caption).foregroundStyle(.red).textSelection(.enabled)
    }
  }
}

private struct DecisionPlaygroundView: View {
  @ObservedObject var decision: DecisionStudio
  @State private var importingFor: UUID?
  @State private var importing = false

  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      GroupBox(L("État", "State")) {
        VStack(alignment: .leading, spacing: 10) {
          HStack {
            Picker("", selection: $decision.draft.stateIsJSON) {
              Text(L("Texte", "Text")).tag(false)
              Text("JSON").tag(true)
            }.pickerStyle(.segmented).frame(width: 160).labelsHidden()
            Spacer()
            Button(L("Exemple", "Example")) { decision.draft = .example }
          }
          TextEditor(text: $decision.draft.state)
            .font(.system(size: 13, design: decision.draft.stateIsJSON ? .monospaced : .default))
            .frame(minHeight: 110)
            .scrollContentBackground(.hidden)
            .padding(6)
            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
            .accessibilityIdentifier("decision.state")
          mediaRow
        }.padding(6)
      }
      GroupBox(L("Questions", "Questions")) {
        VStack(alignment: .leading, spacing: 12) {
          ForEach($decision.draft.questions) { $question in
            DecisionQuestionEditor(
              question: $question, decision: decision,
              importSpoken: { importingFor = question.id; importing = true },
              remove: { decision.draft.questions.removeAll { $0.id == question.id } })
            Divider()
          }
          Menu(L("Ajouter une question", "Add a question")) {
            ForEach(DecisionQuestionKind.allCases) { kind in
              Button(kind.title) {
                let index = decision.draft.questions.count + 1
                var question = DecisionQuestion(name: "q\(index)", kind: kind, instructions: "")
                if kind.usesOptions { question.options = [DecisionOption(name: "a"), DecisionOption(name: "b")] }
                decision.draft.questions.append(question)
              }
            }
          }.fixedSize()
        }.padding(6)
      }
      HStack {
        Button {
          decision.run()
        } label: {
          Label(L("Décider", "Decide"), systemImage: "play.fill").frame(minWidth: 120)
        }
        .buttonStyle(.borderedProminent).controlSize(.large)
        .disabled(decision.running || decision.selectedPath == nil || decision.server.active)
        .keyboardShortcut(.return, modifiers: .command)
        .accessibilityIdentifier("decision.run")
        if decision.running { ProgressView().controlSize(.small) }
        if decision.server.active {
          Text(L("Le serveur de décision utilise le modèle.", "The decision server is using the model."))
            .font(.caption).foregroundStyle(StudioTheme.quiet)
        }
      }
      if let outcome = decision.outcome {
        DecisionResultsView(outcome: outcome, draft: decision.draft)
      }
    }
    .fileImporter(
      isPresented: $importing, allowedContentTypes: importingFor == nil ? [.image, .audio] : [.audio],
      allowsMultipleSelection: importingFor == nil
    ) { result in
      if case let .success(urls) = result { decision.importMedia(urls, spokenFor: importingFor) }
      importingFor = nil
    }
  }

  private var mediaRow: some View {
    VStack(alignment: .leading, spacing: 8) {
      if !decision.draft.media.isEmpty {
        ScrollView(.horizontal, showsIndicators: false) {
          HStack(spacing: 10) {
            ForEach(decision.draft.media) { item in
              HStack(spacing: 6) {
                if item.kind == .image, let image = NSImage(contentsOfFile: item.path) {
                  Image(nsImage: image).resizable().scaledToFill().frame(width: 34, height: 34)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                } else {
                  Image(systemName: "waveform").frame(width: 34, height: 34)
                }
                Text(item.label).font(.caption).lineLimit(1).frame(maxWidth: 140)
                Button { decision.draft.media.removeAll { $0.id == item.id } } label: { Image(systemName: "xmark") }
                  .buttonStyle(.borderless)
              }
              .padding(6).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
            }
          }
        }
      }
      HStack(spacing: 10) {
        Button { importingFor = nil; importing = true } label: {
          Label(L("Image ou audio…", "Picture or audio…"), systemImage: "paperclip")
        }
        Button { decision.toggleRecording() } label: {
          Label(
            decision.recordingTarget == DecisionStudio.stateTarget ? L("Arrêter", "Stop") : L("Enregistrer", "Record"),
            systemImage: decision.recordingTarget == DecisionStudio.stateTarget ? "stop.circle.fill" : "mic")
        }
        Text(L("Images PNG, JPEG, WebP… · audio jusqu’à 5 min (pas de vidéo)", "PNG, JPEG, WebP… pictures · audio up to 5 min (no video)"))
          .font(.caption).foregroundStyle(StudioTheme.quiet)
      }
    }
    .onDrop(of: [.fileURL], isTargeted: nil) { providers in
      for provider in providers {
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
          guard let url else { return }
          Task { @MainActor in decision.importMedia([url]) }
        }
      }
      return true
    }
  }
}

private struct DecisionQuestionEditor: View {
  @Binding var question: DecisionQuestion
  @ObservedObject var decision: DecisionStudio
  let importSpoken: () -> Void
  let remove: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 10) {
        TextField(L("nom", "name"), text: $question.name).frame(width: 130).font(.system(.body, design: .monospaced))
        Picker("", selection: $question.kind) {
          ForEach(DecisionQuestionKind.allCases) { Text($0.title).tag($0) }
        }.labelsHidden().frame(width: 170)
        Spacer()
        Button(action: remove) { Image(systemName: "trash") }.buttonStyle(.borderless)
          .help(L("Supprimer la question", "Delete the question"))
      }
      HStack(spacing: 8) {
        if let spoken = question.spokenPath {
          Label(L("Question parlée", "Spoken question") + " · " + URL(fileURLWithPath: spoken).lastPathComponent, systemImage: "waveform")
            .font(.callout)
          Button(L("Écrire à la place", "Type instead")) { question.spokenPath = nil }
          Spacer()
        } else {
          TextField(L("Consigne, ex. « Le client demande-t-il un remboursement ? »", "Instructions, e.g. “Is the customer asking for a refund?”"), text: $question.instructions)
        }
        Menu {
          Button(L("Fichier audio…", "Audio file…"), action: importSpoken)
          Button(decision.recordingTarget == question.id ? L("Arrêter l’enregistrement", "Stop recording") : L("Enregistrer", "Record")) {
            decision.toggleRecording(for: question.id)
          }
        } label: {
          Image(systemName: decision.recordingTarget == question.id ? "stop.circle.fill" : "mic")
        }
        .menuStyle(.borderlessButton).fixedSize()
        .help(L("Poser la question à voix haute (d1-Omni)", "Ask the question aloud (d1-Omni)"))
      }
      switch question.kind {
      case .noul:
        HStack {
          TextField(L("Oui signifie… (facultatif)", "Yes means… (optional)"), text: $question.yesMeans)
          TextField(L("Non signifie… (facultatif)", "No means… (optional)"), text: $question.noMeans)
        }
      case .choice, .multiple, .score:
        ForEach($question.options) { $option in
          HStack {
            TextField(question.kind == .score ? L("niveau", "level") : L("option", "option"), text: $option.name).frame(width: 160)
            if question.kind != .score {
              TextField(L("description (facultatif)", "description (optional)"), text: $option.detail)
            }
            Button { question.options.removeAll { $0.id == option.id } } label: { Image(systemName: "minus.circle") }
              .buttonStyle(.borderless)
          }
        }
        Button(question.kind == .score ? L("Ajouter un niveau (du plus bas au plus haut)", "Add a level (lowest first)") : L("Ajouter une option", "Add an option")) {
          question.options.append(DecisionOption(name: ""))
        }.buttonStyle(.borderless)
      case .number:
        TextField(L("Unité (facultatif)", "Unit (optional)"), text: $question.unit).frame(width: 200)
      case .estimate:
        HStack {
          TextField(L("min", "low"), text: $question.low).frame(width: 80)
          TextField(L("max", "high"), text: $question.high).frame(width: 80)
          TextField(L("pas (vide : 128 valeurs)", "step (empty: 128 values)"), text: $question.step).frame(width: 190)
          TextField(L("unité", "unit"), text: $question.unit).frame(width: 90)
        }
      case .boxes:
        TextField(L("Au plus (1–16, facultatif)", "At most (1–16, optional)"), text: $question.maxBoxes).frame(width: 200)
      case .text, .date, .box:
        EmptyView()
      }
    }
    .textFieldStyle(.roundedBorder)
  }
}

private struct DecisionResultsView: View {
  let outcome: DecisionStudio.Outcome
  let draft: DecisionDraft
  @State private var showRaw = false

  private var picture: DecisionMediaItem? {
    let pictures = draft.media.filter { $0.kind == .image }
    return pictures.count == 1 ? pictures[0] : nil
  }

  var body: some View {
    GroupBox(L("Réponses", "Answers")) {
      VStack(alignment: .leading, spacing: 14) {
        Text(
          "\(outcome.inputTokens) " + L("tokens lus · 0 généré · ", "tokens read · 0 generated · ")
            + String(format: "%.0f ms", outcome.elapsedMilliseconds)
        ).font(.caption).foregroundStyle(StudioTheme.quiet).accessibilityIdentifier("decision.usage")
        ForEach(outcome.order, id: \.self) { name in
          if let answer = outcome.answers[name] as? [String: Any] {
            DecisionAnswerRow(name: name, answer: answer, state: draft.stateIsJSON ? nil : draft.state)
          }
        }
        if let picture, let image = NSImage(contentsOfFile: picture.path) {
          let boxes = outcome.order.flatMap { name -> [[String: Double]] in
            guard let answer = outcome.answers[name] as? [String: Any] else { return [] }
            if let box = answer["box"] as? [String: Double] { return [box] }
            return (answer["boxes"] as? [[String: Any]] ?? []).compactMap { $0["box"] as? [String: Double] }
          }
          if !boxes.isEmpty {
            Image(nsImage: image).resizable().scaledToFit().frame(maxWidth: 420, maxHeight: 320)
              .overlay {
                GeometryReader { geometry in
                  ForEach(Array(boxes.enumerated()), id: \.offset) { _, box in
                    let rect = CGRect(
                      x: (box["left"] ?? 0) * geometry.size.width, y: (box["top"] ?? 0) * geometry.size.height,
                      width: ((box["right"] ?? 0) - (box["left"] ?? 0)) * geometry.size.width,
                      height: ((box["bottom"] ?? 0) - (box["top"] ?? 0)) * geometry.size.height)
                    Rectangle().stroke(Color.yellow, lineWidth: 2).frame(width: rect.width, height: rect.height)
                      .position(x: rect.midX, y: rect.midY)
                  }
                }
              }
          }
        }
        DisclosureGroup(L("JSON brut", "Raw JSON"), isExpanded: $showRaw) {
          Text(outcome.raw).font(.caption.monospaced()).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }.padding(6)
    }
  }
}

private struct DecisionAnswerRow: View {
  let name: String
  let answer: [String: Any]
  let state: String?

  private func bar(_ label: String, _ value: Double, strong: Bool) -> some View {
    HStack(spacing: 8) {
      Text(label).font(.callout).frame(width: 150, alignment: .leading).lineLimit(1)
      GeometryReader { geometry in
        ZStack(alignment: .leading) {
          RoundedRectangle(cornerRadius: 3).fill(StudioTheme.panel)
          RoundedRectangle(cornerRadius: 3).fill(StudioTheme.accent.opacity(strong ? 0.85 : 0.35))
            .frame(width: max(2, geometry.size.width * min(max(value, 0), 1)))
        }
      }.frame(height: 10)
      Text(String(format: "%.1f %%", value * 100)).font(.caption.monospacedDigit()).frame(width: 64, alignment: .trailing)
    }
  }

  private func probabilities() -> [(String, Double)] {
    guard let raw = answer["probabilities"] as? [String: Any] else { return [] }
    let legend = answer["legend"] as? [String: String]
    let rows = raw.compactMap { key, value in (value as? Double).map { (key: key, value: $0) } }
    // Levels keep their order; options are listed most likely first.
    let ordered = rows.allSatisfy({ Int($0.key) != nil })
      ? rows.sorted { Int($0.key)! < Int($1.key)! } : rows.sorted { ($0.value, $1.key) > ($1.value, $0.key) }
    return ordered.map { (legend?[$0.key] ?? $0.key, $0.value) }
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 8) {
        Text(name).font(.system(.body, design: .monospaced).weight(.semibold))
        Text(headline).foregroundStyle(StudioTheme.ink)
      }
      switch answer["type"] as? String {
      case "noul":
        bar(L("oui", "yes"), answer["noul"] as? Double ?? 0, strong: true)
      case "choice", "score":
        let chosen = Set((answer["choices"] as? [String] ?? []) + [answer["choice"] as? String].compactMap { $0 })
        ForEach(probabilities(), id: \.0) { label, value in
          bar(label, value, strong: chosen.contains(label) || label == headline)
        }
      case "text", "number", "date":
        if let found = answer["found"] as? Double {
          bar(L("trouvé", "found"), found, strong: found >= 0.5)
        }
        if let state, let span = answer["span"] as? [Int], span.count == 2 {
          highlighted(state, span: span)
        }
      case "box", "boxes":
        if let found = answer["found"] as? Double { bar(L("présent", "present"), found, strong: found >= 0.5) }
        if let presence = answer["presence"] as? [Double] {
          ForEach(Array(presence.prefix(6).enumerated()), id: \.offset) { index, value in
            bar("≥ \(index + 1)", value, strong: value >= 0.5)
          }
        }
      default:
        EmptyView()
      }
    }
  }

  private var headline: String {
    switch answer["type"] as? String {
    case "noul":
      let p = answer["noul"] as? Double ?? 0
      return p >= 0.5 ? L("oui", "yes") : L("non", "no")
    case "choice":
      if let choices = answer["choices"] as? [String] { return choices.isEmpty ? "—" : choices.joined(separator: ", ") }
      return answer["choice"] as? String ?? "—"
    case "score":
      let level = Int(answer["score"] as? Double ?? 0)
      return (answer["legend"] as? [String: String])?[String(level)] ?? "\(level)"
    case "number":
      if let interval = answer["interval"] as? [Double], let value = answer["number"] as? Double {
        return String(format: "%g  [%g – %g]", value, interval[0], interval[1]) + ((answer["unit"] as? String).map { " " + $0 } ?? "")
      }
      fallthrough
    case "text", "date":
      if let text = answer["text"] as? String { return "« \(text) »" }
      return L("absent", "absent")
    case "box":
      return (answer["box"] as? [String: Double]).map { _ in L("trouvée", "found") } ?? L("absente", "absent")
    case "boxes":
      let count = answer["count"] as? Int ?? 0
      return "\(count)" + ((answer["at_least"] as? Bool) == true ? "+" : "")
    default:
      return ""
    }
  }

  private func highlighted(_ text: String, span: [Int]) -> some View {
    let scalars = Array(text.unicodeScalars)
    let start = min(max(span[0], 0), scalars.count), end = min(max(span[1], start), scalars.count)
    func piece(_ range: Range<Int>) -> String {
      var out = String.UnicodeScalarView()
      out.append(contentsOf: scalars[range])
      return String(out)
    }
    var attributed = AttributedString(piece(max(0, start - 60)..<start))
    var hit = AttributedString(piece(start..<end))
    hit.backgroundColor = .yellow.opacity(0.35)
    hit.font = .body.bold()
    attributed.append(hit)
    attributed.append(AttributedString(piece(end..<min(scalars.count, end + 60))))
    return Text(attributed).font(.callout).foregroundStyle(StudioTheme.secondary).lineLimit(3)
  }
}

private struct DecisionServerView: View {
  @ObservedObject var decision: DecisionStudio
  @ObservedObject var server: LocalServer
  @AppStorage("decision.server.port") private var port = "8100"
  @AppStorage("decision.server.network") private var network = false
  private var validPort: Int? { Int(port).flatMap { (1024...65535).contains($0) ? $0 : nil } }

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      GroupBox {
        VStack(alignment: .leading, spacing: 14) {
          Text(L("API HTTP locale : POST /v1/decide", "Local HTTP API: POST /v1/decide")).font(.headline)
          Picker(L("Accès", "Access"), selection: $network) {
            Text(L("Ce Mac", "This Mac")).tag(false)
            Text(L("Réseau local", "Local network")).tag(true)
          }.pickerStyle(.segmented).disabled(server.active).frame(maxWidth: 360)
          HStack {
            Text(L("Port local", "Local port"))
            TextField("8100", text: $port).frame(width: 90).disabled(server.active).textFieldStyle(.roundedBorder)
            Spacer()
            if server.busy { ProgressView().controlSize(.small) }
            Button(server.active ? L("Arrêter", "Stop") : L("Démarrer", "Start")) {
              if server.active { server.stop() } else { decision.startServer(port: validPort ?? 0, network: network) }
            }
            .disabled(!server.active && (decision.selectedPath == nil || validPort == nil))
            .accessibilityIdentifier("decision.server.toggle")
          }
          Text(
            server.running
              ? L("En ligne", "Online") : server.busy ? L("Chargement / arrêt…", "Loading / stopping…") : L("Arrêté", "Stopped")
          ).foregroundStyle(StudioTheme.secondary)
          Text(
            L(
              "Le serveur charge le modèle de décision à la place du Playground et du modèle de chat. Les médias y passent en data: URL (pas de chemins locaux).",
              "The server loads the decision model in place of the Playground and the chat model. Media travel as data: URLs (no local paths).")
          ).font(.caption).foregroundStyle(StudioTheme.quiet)
        }.padding(10)
      }
      if let port = server.port, server.running {
        let base = "http://127.0.0.1:\(port)"
        VStack(alignment: .leading, spacing: 10) {
          Text(base + "/v1/decide").font(.system(.body, design: .monospaced)).textSelection(.enabled)
          HStack {
            Button(L("Copier l’URL", "Copy URL")) { copy(base + "/v1/decide") }
            if let key = server.apiKey {
              Button(L("Copier la clé API", "Copy API key")) { copy(key) }
              Button(L("Copier une requête curl", "Copy a curl request")) {
                do {
                  let body = try decision.serverBody()
                  let file = try DecisionMediaImport.destination(extension: "json")
                  try Data(body.utf8).write(to: file)
                  copy("curl -s -H 'Authorization: Bearer \(key)' -H 'Content-Type: application/json' --data-binary @'\(file.path)' \(base)/v1/decide")
                } catch { decision.error = error.localizedDescription }
              }
            }
          }
          Text("GET /health · GET /v1/models · GET /v1/capabilities · POST /v1/decide").font(.caption.monospaced())
          Text(
            L(
              "Corps : {\"state\": …, \"questions\": {nom: question}, \"media\": [{\"type\": \"image\"|\"audio\", \"url\": \"data:…\"}]}. Une nouvelle clé est générée à chaque démarrage.",
              "Body: {\"state\": …, \"questions\": {name: question}, \"media\": [{\"type\": \"image\"|\"audio\", \"url\": \"data:…\"}]}. A new key is generated at each start.")
          ).font(.caption).foregroundStyle(StudioTheme.quiet).textSelection(.enabled)
          if network {
            Text(L("Depuis un autre appareil, remplacez 127.0.0.1 par l’adresse réseau de ce Mac.", "From another device, replace 127.0.0.1 with this Mac’s network address."))
              .font(.caption).foregroundStyle(StudioTheme.quiet)
          }
        }
      }
      if let error = server.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
      if !server.log.isEmpty {
        DisclosureGroup(L("Journal du serveur", "Server log")) {
          Text(server.log).font(.caption.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
      }
    }
  }

  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
}
