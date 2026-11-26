import AppKit
import Foundation
import QuickLookUI
import SwiftUI

@main struct StudioWorkspaceCheck {
  private struct DeliverableCheckError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
  }

  @MainActor static func main() {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.regular)
    Task { @MainActor in
      do {
        try await runChecks()
        NSApp.terminate(nil)
      } catch {
        FileHandle.standardError.write(Data((String(describing: error) + "\n").utf8))
        exit(1)
      }
    }
    NSApp.run()
  }

  @MainActor private static func runChecks() async throws {
    let files = FileManager.default
    let root = files.temporaryDirectory.appendingPathComponent("mlxl3-workspace-check-\(UUID())")
    let suite = "io.mlxl3.workspace-check.\(UUID())"
    let prefs = UserDefaults(suiteName: suite)!
    defer {
      try? files.removeItem(at: root)
      prefs.removePersistentDomain(forName: suite)
    }
    var count = 0
    func check(_ valid: Bool, _ message: String) {
      precondition(valid, message)
      count += 1
    }
    // The assistant's result stays pinned to its own request and response, across later edits.
    let task = AgentProjectStore(workspace: root.appendingPathComponent("turn-results/files"))
    func deliverableDenied(_ path: String, _ result: AgentTurnResult) throws -> Bool {
      do {
        _ = try task.deliverableContents(path, in: result)
        return false
      } catch let error as CocoaError {
        return error.code == .fileReadNoPermission
      }
    }
    let user = UUID(), assistant = UUID()
    try task.checkpoint(messageID: user, startingTurn: true)
    let receipts = AgentCommandReceipt.folder(metadata: task.metadata, turn: user)
    let incomplete = AgentCommandReceipt.start(command: "npm test", in: receipts)
    let shell = AgentCommandReceipt.start(command: "npm test || true", in: receipts)
    shell?.finish(.init(output: "all tests passed", status: 7, interrupted: false, raw: nil, omitted: nil))
    let path = "notes avec espaces.md"
    try AgentProjectStore.write(Data("first version".utf8), path: path, root: task.workspace)
    try task.checkpoint(messageID: assistant)
    guard let result = try task.turnResult(user: user, assistant: assistant) else {
      preconditionFailure("Turn with a new document did not produce a result")
    }
    check(result.files == [.init(path: path, added: true)] && result.newDocument == path,
      "A new text document did not expose its review and document actions")
    check(result.commands.commands == ["failure": 1] && result.commands.tests == ["unconfirmed": 1],
      "Unfinished jobs and compound commands were described as passing tests")
    let nextUser = UUID(), nextAssistant = UUID()
    try task.checkpoint(messageID: nextUser, startingTurn: true)
    incomplete?.finish(.init(output: "", status: 0, interrupted: false, raw: nil, omitted: nil))
    try files.removeItem(at: task.workspace.appendingPathComponent(path))
    try AgentProjectStore.write(Data("unrelated".utf8), path: "next.txt", root: task.workspace)
    try task.checkpoint(messageID: nextAssistant)
    guard let secondTurn = try task.turnResult(user: nextUser, assistant: nextAssistant) else {
      preconditionFailure("Turn with an added and a deleted file did not produce a result")
    }
    check(secondTurn.files.count == 2 && secondTurn.newDocument == nil && secondTurn.commands.isEmpty,
      "Added/deleted files or command results crossed into another turn")
    check(secondTurn.deliverableFiles.map(\.path) == ["next.txt"], "A deleted file was offered for export")
    check(try deliverableDenied(path, secondTurn), "A deleted deliverable was readable")
    check(try task.deliverableContents(path, in: result) == Data("first version".utf8),
      "A later deletion prevented exporting the historical bytes")
    check(try task.turnResult(user: user, assistant: assistant) == result,
      "Later edits or a late background job changed a historical result")
    check(try task.turnResult(user: UUID(), assistant: assistant) == nil,
      "Missing legacy checkpoint silently substituted the current workspace")
    let reviewedTurn = AgentStudioModel()
    await reviewedTurn.open(workspace: task.workspace, file: nil, turnResult: result)
    check(reviewedTurn.tab == .diff && reviewedTurn.diff.contains("+first version")
      && !reviewedTurn.diff.contains("unrelated") && !reviewedTurn.canApply,
      "Review shortcut opened the current diff or allowed applying a historic turn")
    let openedDocument = AgentStudioModel()
    await openedDocument.open(workspace: task.workspace, file: task.workspace.appendingPathComponent(path),
      turnResult: result)
    check(openedDocument.tab == .code && openedDocument.code == "first version" && !openedDocument.editable,
      "Historical document shortcut showed today's content or made past content editable")
    let silent = UUID(), reply = UUID()
    try task.checkpoint(messageID: silent, startingTurn: true)
    try task.checkpoint(messageID: reply)
    let unchanged = try task.turnResult(user: silent, assistant: reply)
    check(unchanged?.files.isEmpty == true && unchanged?.commands.isEmpty == true,
      "An unchanged turn displayed a result")
    let binaryRequest = UUID(), binaryResponse = UUID()
    try task.checkpoint(messageID: binaryRequest, startingTurn: true)
    try AgentProjectStore.write(Data([0, 255]), path: "actually-binary.md", root: task.workspace)
    try task.checkpoint(messageID: binaryResponse)
    guard let binaryResult = try task.turnResult(user: binaryRequest, assistant: binaryResponse) else {
      preconditionFailure("Turn with binary data did not produce a result")
    }
    check(binaryResult.files.count == 1 && binaryResult.newDocument == nil,
      "A misleading extension offered to open binary bytes as a document")
    check(try task.deliverableContents("actually-binary.md", in: binaryResult) == Data([0, 255]),
      "Exporting a binary deliverable changed its bytes")
    check(try deliverableDenied("next.txt", binaryResult), "An unchanged file was readable as a deliverable")
    check(try deliverableDenied("../next.txt", binaryResult), "An invalid relative path was readable")
    let reportRequest = UUID(), reportResponse = UUID()
    try task.checkpoint(messageID: reportRequest, startingTurn: true)
    let firstReport = "reports [A]/summary.csv", secondReport = "reports B/summary.csv"
    let originalReport = Data("Score,Note\n5,\"été\"\n".utf8)
    try AgentProjectStore.write(originalReport, path: firstReport, root: task.workspace)
    try AgentProjectStore.write(Data(), path: secondReport, root: task.workspace)
    try AgentProjectStore.write(Data("updated".utf8), path: "next.txt", root: task.workspace)
    try task.checkpoint(messageID: reportResponse)
    guard let reports = try task.turnResult(user: reportRequest, assistant: reportResponse) else {
      preconditionFailure("Turn with several deliverables did not produce a result")
    }
    check(Set(reports.deliverableFiles.map(\.path)) == [firstReport, secondReport, "next.txt"],
      "A turn lost a nested or modified deliverable")
    let firstReportURL = task.workspace.appendingPathComponent(firstReport)
    let outsideURL = root.appendingPathComponent("outside-workspace.csv")
    let outsideContents = Data("outside".utf8)
    try outsideContents.write(to: outsideURL)
    try files.removeItem(at: firstReportURL)
    try files.createSymbolicLink(at: firstReportURL, withDestinationURL: outsideURL)
    check(try task.deliverableContents(firstReport, in: reports) == originalReport,
      "Export used a live symlink instead of the archived literal path")
    check(try task.deliverableContents(secondReport, in: reports).isEmpty,
      "A same-name file with zero bytes could not be exported separately")
    check(try task.deliverableContents("next.txt", in: reports) == Data("updated".utf8)
      && task.deliverableContents("next.txt", in: secondTurn) == Data("unrelated".utf8),
      "Exporting a modified file did not respect both historical versions")
    check(try Data(contentsOf: outsideURL) == outsideContents
      && files.destinationOfSymbolicLink(atPath: firstReportURL.path) == outsideURL.path,
      "Reading a deliverable modified the live workspace or its link target")
    try files.removeItem(at: firstReportURL)
    try originalReport.write(to: firstReportURL)
    let userWithoutReceipts = UUID(), responseWithoutReceipts = UUID()
    try task.checkpoint(messageID: userWithoutReceipts, startingTurn: true)
    let damagedReceipts = AgentCommandReceipt.folder(metadata: task.metadata, turn: userWithoutReceipts)
    try Data("truncated".utf8).write(to: damagedReceipts.appendingPathComponent(UUID().uuidString))
    try AgentProjectStore.write(Data("kept".utf8), path: "recovered.txt", root: task.workspace)
    try task.checkpoint(messageID: responseWithoutReceipts)
    let recovered = try task.turnResult(user: userWithoutReceipts, assistant: responseWithoutReceipts)
    check(recovered?.newDocument == "recovered.txt" && recovered?.commands.isEmpty == true,
      "A malformed command receipt lost the file result or reported success")
    let source = root.appendingPathComponent("original")
    let workspace = root.appendingPathComponent("task/files")
    try files.createDirectory(at: source, withIntermediateDirectories: true)
    try Data("Bonjour, voici une blague.\nDeuxième ligne.\n".utf8).write(
      to: source.appendingPathComponent("blague.txt"))
    try Data("<!doctype html><html><body><h1>Workspace</h1></body></html>".utf8).write(
      to: source.appendingPathComponent("index.html"))
    try Data("const value = 1;\n// a comment\nconsole.log('hello');\n".utf8).write(
      to: source.appendingPathComponent("main.js"))
    try Data("old file".utf8).write(to: source.appendingPathComponent("obsolete.txt"))
    try Data("Moved document".utf8).write(to: source.appendingPathComponent("draft.txt"))
    try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: source.appendingPathComponent("blague.txt").path)
    try Data([0, 255, 14]).write(to: source.appendingPathComponent("image.bin"))
    let store = AgentProjectStore(workspace: workspace)
    try store.importProject(source)
    let baseline = try store.versions().first!.id
    let model = AgentStudioModel()
    await model.open(workspace: workspace, file: workspace.appendingPathComponent("blague.txt"))
    check(
      model.error == nil && model.tab == .code && model.code.contains("Bonjour"),
      "TXT did not open directly as code")
    model.code = "Edited text\n"
    check(model.dirty, "Dirty editor not detected")
    check(
      !(await model.select("main.js")) && model.code == "Edited text\n",
      "File navigation silently discarded editor")
    check(
      !(await model.selectRevision(baseline)) && model.dirty,
      "Version navigation silently discarded editor")
    let versionsBefore = try store.versions().count
    var permissions = model.state.permissions
    permissions.run = false
    await model.setPermissions(permissions)
    check(model.code == "Edited text\n" && model.dirty, "Changing permissions erased edits")
    check(
      try store.versions().count == versionsBefore && store.state().permissions.run == false,
      "Permission update rewrote history")
    check(await model.save(), "Text save failed")
    check(
      try String(contentsOf: workspace.appendingPathComponent("blague.txt"), encoding: .utf8)
        == "Edited text\n", "Saved wrong file")
    check(
      try String(contentsOf: source.appendingPathComponent("blague.txt"), encoding: .utf8).contains(
        "Bonjour"), "Save changed original before review")
    model.code = "KEEP MY DRAFT"
    try AgentProjectStore.write(Data("External write".utf8), path: "blague.txt", root: workspace)
    check(
      !(await model.save()) && model.code == "KEEP MY DRAFT" && model.error != nil,
      "Concurrent writer overwrote editor or was silently lost")
    model.discardEdits()
    await model.reload()
    check(
      model.code == "External write" && model.error == nil, "Refresh lost actual workspace state")
    try files.removeItem(at: workspace.appendingPathComponent("obsolete.txt"))
    try files.moveItem(at: workspace.appendingPathComponent("draft.txt"), to: workspace.appendingPathComponent("final.txt"))
    try AgentProjectStore.write(Data("New file".utf8), path: "new.txt", root: workspace)
    _ = try store.checkpoint("Added and deleted files")
    await model.reload()
    await model.select("obsolete.txt")
    check(!model.fileExists && model.code.isEmpty, "Deleted file displayed stale content")
    await model.selectRevision(baseline)
    check(
      model.fileExists && model.code == "old file" && !model.editable,
      "History failed to expose deleted file read-only")
    check(!model.paths.contains("new.txt"), "Old history lists files that did not exist yet")
    await model.selectRevision(model.versions.first!.id)
    await model.select("image.bin")
    check(model.binary && !model.editable && !model.dirty, "Binary file became text-editable")
    await model.select("main.js")
    model.code = "const value = 2;\n// updated\nconsole.log('hello');\n"
    check(await model.save(), "JS save failed")
    check(
      model.diff.contains("+const value = 2") && !model.diff.contains("External write"),
      "File diff contains unrelated file changes")
    await model.reviewProject()
    check(
      model.canApply && model.allChanges && model.diff.contains("External write")
        && model.diff.contains("old file"), "Full review missed changes/deletions")
    let reviewed = model.revision
    try AgentProjectStore.write(Data("UNREVIEWED".utf8), path: "main.js", root: workspace)
    do {
      try store.apply(reviewedRevision: reviewed)
      preconditionFailure("Unreviewed file was applied")
    } catch { count += 1 }
    check(
      try String(contentsOf: source.appendingPathComponent("main.js"), encoding: .utf8).contains(
        "value = 1"), "Rejected apply changed originals")
    _ = try store.checkpoint("Changed since review")
    do {
      try store.apply(reviewedRevision: reviewed)
      preconditionFailure("Stale revision was applied")
    } catch { count += 1 }
    await model.reload()
    await model.reviewProject()
    await model.apply()
    check(model.error == nil && !model.canApply, "Apply did not advance baseline")
    check(
      try !files.fileExists(atPath: source.appendingPathComponent("obsolete.txt").path)
        && String(contentsOf: source.appendingPathComponent("new.txt"), encoding: .utf8)
          == "New file",
      "Apply lost additions/deletions")
    check(
      try !files.fileExists(atPath: source.appendingPathComponent("draft.txt").path)
        && String(contentsOf: source.appendingPathComponent("final.txt"), encoding: .utf8) == "Moved document",
      "Applying a rename must move the document in the original project")
    check(
      (try files.attributesOfItem(atPath: source.appendingPathComponent("blague.txt").path)[.posixPermissions] as? NSNumber)?.intValue == 0o600,
      "Editing a private document must preserve its access permissions")
    for _ in 0..<5 {
      let first = Task { await model.select("blague.txt") }
      await Task.yield()
      let second = Task { await model.select("main.js") }
      _ = await first.value
      _ = await second.value
      check(
        model.selected == "main.js" && model.code == "UNREVIEWED",
        "Stale file load replaced newer selection")
    }
    let literalSource = root.appendingPathComponent("literal-source")
    try files.createDirectory(at: literalSource, withIntermediateDirectories: true)
    for name in ["a.txt", "[a].txt", ":notes.txt"] {
      try Data("Initial text\n".utf8).write(to: literalSource.appendingPathComponent(name))
    }
    let literalStore = AgentProjectStore(
      workspace: root.appendingPathComponent("literal-task/files"))
    try literalStore.importProject(literalSource)
    let literalBefore = try literalStore.versions().first!.id
    check(try literalStore.contents(":notes.txt", revision: literalBefore) == Data("Initial text\n".utf8),
      "A leading colon in a filename must survive the initial checkpoint")
    try AgentProjectStore.write(
      Data("UNRELATED_GLOB_MATCH\n".utf8), path: "a.txt", root: literalStore.workspace)
    try AgentProjectStore.write(
      Data("INTENDED_LITERAL_FILE\n".utf8), path: "[a].txt", root: literalStore.workspace)
    let literalAfter = try literalStore.checkpoint("Literal filename changes")
    let literalDiff = try literalStore.diff(from: literalBefore, to: literalAfter, path: "[a].txt")
    check(
      literalDiff.contains("+INTENDED_LITERAL_FILE")
        && !literalDiff.contains("UNRELATED_GLOB_MATCH"),
      "Selected-file diff interpreted a filename as a Git pattern")
    let renameSource = root.appendingPathComponent("rename-source")
    try files.createDirectory(at: renameSource, withIntermediateDirectories: true)
    let renameStore = AgentProjectStore(workspace: root.appendingPathComponent("rename-task/files"))
    let modes: [(name: String, imported: Int, original: Int, reviewed: Int, expected: Int)] = [
      ("shared", 0o660, 0o660, 0o660, 0o660),
      ("restricted", 0o644, 0o600, 0o644, 0o600),
      ("tool", 0o644, 0o600, 0o755, 0o700),
      ("was-executable", 0o754, 0o740, 0o644, 0o640),
      ("group-execute", 0o650, 0o650, 0o650, 0o650),
    ]
    func fileMode(_ url: URL) throws -> Int {
      (try files.attributesOfItem(atPath: url.path)[.posixPermissions] as! NSNumber).intValue
    }
    for fixture in modes {
      let url = renameSource.appendingPathComponent(fixture.name)
      try Data(fixture.name.utf8).write(to: url)
      try files.setAttributes([.posixPermissions: fixture.imported], ofItemAtPath: url.path)
    }
    try renameStore.importProject(renameSource)
    for fixture in modes {
      let sourceFile = renameSource.appendingPathComponent(fixture.name)
      try files.setAttributes([.posixPermissions: fixture.original], ofItemAtPath: sourceFile.path)
      let workspaceFile = renameStore.workspace.appendingPathComponent(fixture.name)
      let renamed = renameStore.workspace.appendingPathComponent("new-" + fixture.name)
      try files.moveItem(at: workspaceFile, to: renamed)
      try files.setAttributes([.posixPermissions: fixture.reviewed], ofItemAtPath: renamed.path)
    }
    try AgentProjectStore.write(Data("child".utf8), path: "zz-blocked/child", root: renameStore.workspace)
    let renameRevision = try renameStore.checkpoint("Renamed documents")
    let collision = renameSource.appendingPathComponent("new-tool")
    let blocker = renameSource.appendingPathComponent("zz-blocked")
    // A destination collision is caught in preflight. A blocked directory exercises rollback.
    for url in [collision, blocker] {
      try Data("keep".utf8).write(to: url)
      do {
        try renameStore.apply(reviewedRevision: renameRevision)
        preconditionFailure("A blocked rename must fail")
      } catch { count += 1 }
      check(try String(contentsOf: url, encoding: .utf8) == "keep", "A collision must preserve the existing file")
      for fixture in modes {
        let original = renameSource.appendingPathComponent(fixture.name)
        let renamed = renameSource.appendingPathComponent("new-" + fixture.name)
        check(try fileMode(original) == fixture.original && Data(contentsOf: original) == Data(fixture.name.utf8),
          "A failed apply must restore permissions and bytes for \(fixture.name)")
        check(renamed == url || !files.fileExists(atPath: renamed.path), "A failed apply left a new destination")
      }
      try files.removeItem(at: url)
    }
    try renameStore.apply(reviewedRevision: renameRevision)
    for fixture in modes {
      let original = renameSource.appendingPathComponent(fixture.name)
      let renamed = renameSource.appendingPathComponent("new-" + fixture.name)
      check(try fileMode(renamed) == fixture.expected && Data(contentsOf: renamed) == Data(fixture.name.utf8)
        && !files.fileExists(atPath: original.path), "A rename lost the source access rights or bytes: \(fixture.name)")
    }
    let attributeSource = root.appendingPathComponent("attribute-source")
    try files.createDirectory(at: attributeSource, withIntermediateDirectories: true)
    let raw = Data("Windows line endings\r\nSecond line\r\n".utf8)
    try raw.write(to: attributeSource.appendingPathComponent("café.txt"))
    try Data("* text\n*.txt export-ignore\n".utf8).write(
      to: attributeSource.appendingPathComponent(".gitattributes"))
    let attributeStore = AgentProjectStore(
      workspace: root.appendingPathComponent("attribute-task/files"))
    try attributeStore.importProject(attributeSource)
    let attributeRevision = try attributeStore.versions().first!.id
    check(
      try attributeStore.contents("café.txt", revision: attributeRevision) == raw,
      "Git attributes transformed the saved file bytes")
    let zip = root.appendingPathComponent("attribute.zip")
    try attributeStore.exportZIP(zip, revision: attributeRevision)
    let extraction = root.appendingPathComponent("unzipped")
    let unzip = Process()
    unzip.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
    unzip.arguments = ["-x", "-k", zip.path, extraction.path]
    try unzip.run()
    unzip.waitUntilExit()
    check(
      try unzip.terminationStatus == 0
        && Data(contentsOf: extraction.appendingPathComponent("café.txt")) == raw,
      "Project attributes hid or transformed a ZIP file")
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("ui/history.json"), isPreview: true,
      preferences: prefs)
    let chatID = studio.selectedConversationID
    await studio.openAgentProject(source)
    check(studio.selectedMode == .chat && studio.selectedConversationID == chatID
      && studio.conversations.count == 1 && studio.attachedAgentFolder == nil,
      "Attaching a folder must not turn a Chat into an Agent task")
    studio.selectMode(.agent)
    let taskID = studio.selectedConversationID!
    let uiWorkspace = studio.agentWorkspaceURL!
    let taskStore = AgentProjectStore(workspace: uiWorkspace)
    let earlier = ChatMessage(role: .user, content: "Keep my work in this thread")
    let taskIndex = studio.conversations.firstIndex(where: { $0.id == taskID })!
    studio.conversations[taskIndex].messages.append(earlier)
    studio.draft = "Review the selected project"
    await studio.importChatFiles([source.appendingPathComponent("blague.txt")])
    let attachmentID = studio.pendingAttachments.first!.id
    try files.createDirectory(at: uiWorkspace, withIntermediateDirectories: true)
    try AgentProjectStore.write(Data("my note".utf8), path: "notes.md", root: uiWorkspace)
    try taskStore.checkpoint(messageID: earlier.id)
    let alias = root.appendingPathComponent("workspace-alias")
    try files.createSymbolicLink(at: alias, withDestinationURL: uiWorkspace.deletingLastPathComponent())
    do {
      try taskStore.importProject(alias)
      preconditionFailure("A task was allowed to attach its own private storage")
    } catch {
      check((error as NSError).code == CocoaError.fileReadNoPermission.rawValue, "Wrong self-import error")
    }
    let emptySource = root.appendingPathComponent("blank-project")
    try files.createDirectory(at: emptySource, withIntermediateDirectories: true)
    let blankStore = AgentProjectStore(workspace: root.appendingPathComponent("blank-task/files"))
    try blankStore.importProject(emptySource)
    check(try blankStore.state().baseline != nil && blankStore.paths(at: blankStore.versions().first!.id).isEmpty,
      "A new empty project cannot be attached")
    var taskPermissions = try taskStore.state().permissions
    taskPermissions.run = false
    try taskStore.setPermissions(taskPermissions)

    // A clash after a successful copy cannot truncate either file or attach part of the project.
    let conflicting = root.appendingPathComponent("conflicting")
    try files.createDirectory(at: conflicting, withIntermediateDirectories: true)
    try files.createDirectory(at: conflicting.appendingPathComponent("first"), withIntermediateDirectories: true)
    try Data().write(to: conflicting.appendingPathComponent("first/file.txt"))
    try Data("collision".utf8).write(to: conflicting.appendingPathComponent("notes.md"))
    await studio.openAgentProject(conflicting)
    check(try studio.featureError != nil && taskStore.state().source == nil
      && !files.fileExists(atPath: uiWorkspace.appendingPathComponent("first").path)
      && Data(contentsOf: uiWorkspace.appendingPathComponent("notes.md")) == Data("my note".utf8),
      "A conflicting attach changed the task files or binding")
    studio.featureError = nil
    await studio.openAgentProject(source)
    check(studio.featureError == nil && studio.selectedConversationID == taskID
      && studio.currentConversation?.messages.first?.id == earlier.id && studio.conversations.count == 2
      && studio.draft == "Review the selected project" && studio.pendingAttachments.map(\.id) == [attachmentID]
      && studio.studioDocument == nil, "Attaching from the composer lost the task, its draft or its attachments")
    check(try taskStore.state().permissions == taskPermissions && studio.attachedAgentFolder?.resolvingSymlinksInPath().path == source.resolvingSymlinksInPath().path
      && Data(contentsOf: uiWorkspace.appendingPathComponent("notes.md")) == Data("my note".utf8),
      "Attaching a folder lost the existing files, permissions or source")
    let attachedState = try taskStore.state()
    check(try !taskStore.paths(at: attachedState.baseline!).contains("notes.md")
      && taskStore.paths(at: taskStore.versions().first!.id).contains("notes.md"),
      "The review must distinguish existing task files from the original source")
    check(studio.persistNow(), "Task history could not be saved")
    let restored = StudioModel(conversationFileURL: root.appendingPathComponent("ui/history.json"),
      isPreview: true, preferences: prefs)
    check(restored.selectedConversationID == taskID && restored.attachedAgentFolder == studio.attachedAgentFolder
      && restored.pendingAttachments.map(\.id) == [attachmentID] && restored.draft == studio.draft,
      "Reopening the Agent task lost its folder or draft")
    let earlierBranch = AgentProjectStore(workspace: root.appendingPathComponent("before-attachment/files"))
    try taskStore.fork(to: earlierBranch.workspace, at: earlier.id, retaining: [earlier.id])
    check(try earlierBranch.state().source == nil && earlierBranch.state().baseline == nil
      && earlierBranch.files().paths == ["notes.md"], "Branch before attachment inherited the new source")
    let later = UUID()
    try taskStore.checkpoint(messageID: later)
    let laterBranch = AgentProjectStore(workspace: root.appendingPathComponent("after-attachment/files"))
    try taskStore.fork(to: laterBranch.workspace, at: later, retaining: [earlier.id, later])
    check(try laterBranch.state().source == attachedState.source && laterBranch.files().paths.contains("blague.txt"),
      "Branch after attachment lost the project")
    await studio.openAgentProject(conflicting)
    check(try studio.featureError != nil && taskStore.state().source == attachedState.source,
      "Attaching a second project silently changed the source")
    studio.featureError = nil
    studio.studioDocument = StudioDocument(
      workspace: uiWorkspace, file: uiWorkspace.appendingPathComponent("blague.txt"))
    let host = NSHostingView(
      rootView: StudioView().environmentObject(studio).preferredColorScheme(.dark))
    let window = NSWindow(
      contentRect: NSRect(x: 60, y: 60, width: 1200, height: 820),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.title = "MLXL3 Workspace Check"
    window.contentView = host
    window.makeKeyAndOrderFront(nil)
    func textViews(_ view: NSView) -> [NSTextView] {
      (view as? NSTextView).map { [$0] } ?? view.subviews.flatMap(textViews)
    }
    func waitForEditor(_ host: NSView, contains expected: String) async throws -> NSTextView? {
      for _ in 0..<50 {
        host.layoutSubtreeIfNeeded()
        let editors = textViews(host).filter {
          $0.accessibilityIdentifier() == "studio.code-editor"
        }
        if editors.count == 1 && editors[0].string.contains(expected) { return editors[0] }
        try await Task.sleep(for: .milliseconds(100))
      }
      let actual = textViews(host).map {
        $0.accessibilityIdentifier() + ": " + $0.string
      }
      FileHandle.standardError.write(Data(("Editor load failed: \(actual)\n").utf8))
      return nil
    }
    let actualTXT = try String(
      contentsOf: source.appendingPathComponent("blague.txt"), encoding: .utf8)
    check(actualTXT == "External write", "Apply did not preserve the reviewed TXT contents")
    let editor = try await waitForEditor(host, contains: actualTXT)
    check(window.sheets.isEmpty, "Studio still uses a modal sheet")
    check(editor != nil, "Full production window did not show the actual TXT file")
    if let folder = ProcessInfo.processInfo.environment["MLXL3_STUDIO_UI_SNAPSHOTS"] {
      let folder = URL(fileURLWithPath: folder)
      try files.createDirectory(at: folder, withIntermediateDirectories: true)
      host.layoutSubtreeIfNeeded()
      let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
      host.cacheDisplay(in: host.bounds, to: bitmap)
      try bitmap.representation(using: .png, properties: [:])!.write(
        to: folder.appendingPathComponent("agent-code.png"))
    }
    let controller = AgentStudioModel()
    let sample =
      "<!doctype html>\n<html lang=\"fr\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>Palette de couleurs</title>\n  <style>\n    body { background: #121212; color: #e9e7df; }\n    .palette { display: flex; gap: 16px; }\n  </style>\n</head>\n<body>\n  <h1>Votre palette</h1>\n  <div class=\"palette\">Créer, explorer, affiner.</div>\n</body>\n</html>\n"
    try AgentProjectStore.write(Data(sample.utf8), path: "index.html", root: uiWorkspace)
    _ = try AgentProjectStore(workspace: uiWorkspace).checkpoint("Sample page")
    await controller.open(
      workspace: uiWorkspace, file: uiWorkspace.appendingPathComponent("index.html"))
    check(controller.tab == .preview, "An HTML file did not open in Preview")
    controller.tab = .code
    controller.code = controller.code.replacingOccurrences(
      of: "Votre palette", with: "Une nouvelle palette")
    check(await controller.save(), "Agent HTML edit failed")
    let projectHost = NSHostingView(
      rootView: AgentStudioView(
        workspace: uiWorkspace, file: uiWorkspace.appendingPathComponent("index.html"),
        model: controller
      ).environmentObject(studio).preferredColorScheme(.dark))
    let projectWindow = NSWindow(
      contentRect: NSRect(x: 80, y: 80, width: 1200, height: 820),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    projectWindow.contentView = projectHost
    projectWindow.makeKeyAndOrderFront(nil)
    for tab in [StudioTab.code, .diff, .history] {
      controller.tab = tab
      if tab != .history {
        let expected = tab == .code ? "Une nouvelle palette" : "+  <h1>Une nouvelle palette"
        check(
          try await waitForEditor(projectHost, contains: expected) != nil,
          "Agent tab did not load actual contents")
      } else {
        try await Task.sleep(for: .milliseconds(300))
      }
      projectHost.layoutSubtreeIfNeeded()
      if let folder = ProcessInfo.processInfo.environment["MLXL3_STUDIO_UI_SNAPSHOTS"] {
        let bitmap = projectHost.bitmapImageRepForCachingDisplay(in: projectHost.bounds)!
        projectHost.cacheDisplay(in: projectHost.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])!.write(
          to: URL(fileURLWithPath: folder).appendingPathComponent("agent-\(tab.rawValue)-page.png"))
      }
      if tab == .code {
        let editor = textViews(projectHost).first {
          $0.accessibilityIdentifier() == "studio.code-editor"
        }!
        check(
          editor.isEditable && editor.string.contains("Une nouvelle palette"),
          "Agent editor is not editable")
        check(
          editor.enclosingScrollView!.clipsToBounds
            && editor.enclosingScrollView!.layer?.masksToBounds == true,
          "Code gutter can paint behind file title")
        var colors = Set<String>()
        for _ in 0..<250 {
          colors.removeAll()
          editor.textStorage!.enumerateAttribute(
            .foregroundColor, in: NSRange(location: 0, length: editor.textStorage!.length)
          ) { value, _, _ in
            if let value = value as? NSColor { colors.insert(value.description) }
          }
          if colors.count >= 3 { break }
          try await Task.sleep(for: .milliseconds(20))
        }
        check(colors.count >= 3, "Code has no actual syntax colors")
      }
      if tab == .diff {
        let editor = textViews(projectHost).first {
          $0.accessibilityIdentifier() == "studio.code-editor"
        }!
        check(
          !editor.isEditable && editor.string.contains("+  <h1>Une nouvelle palette"),
          "Diff is not the actual read-only patch")
      }
    }
    projectWindow.orderOut(nil)
    count += try await checkDeliverableActions(studio: studio, root: root)
    if ProcessInfo.processInfo.environment["MLXL3_STUDIO_INTERACTIVE"] == "1" {
      NSApp.setActivationPolicy(.regular)
      NSApp.activate(ignoringOtherApps: true)
      for _ in 0..<100 {
        if !window.isVisible { break }
        try await Task.sleep(for: .seconds(1))
      }
    }
    window.orderOut(nil)
    print(
      "Workspace checks passed: \(count), CPU-only Git/controller/native full-window navigation; no model, Metal or WebGL test"
    )
  }

  @MainActor private static func checkDeliverableActions(studio: StudioModel, root: URL) async throws -> Int {
    let files = FileManager.default
    let workspace = root.appendingPathComponent("deliverable-task/files")
    let store = AgentProjectStore(workspace: workspace)
    let removed = "Discarded Draft.txt"
    let csv = "Quarterly Dataset.csv"
    let pdf = "Warehouse Graph.pdf"
    let csvBytes = Data("region,units\r\nwest,17\r\n".utf8)
    let pdfBuffer = NSMutableData()
    var mediaBox = CGRect(x: 0, y: 0, width: 144, height: 144)
    guard let consumer = CGDataConsumer(data: pdfBuffer),
      let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
    else { throw DeliverableCheckError(message: "Cannot create the PDF fixture") }
    context.beginPDFPage(nil)
    context.fill(CGRect(x: 10, y: 10, width: 124, height: 124))
    context.endPDFPage()
    context.closePDF()
    let pdfBytes = Data(referencing: pdfBuffer)
    try files.createDirectory(at: workspace, withIntermediateDirectories: true)
    try AgentProjectStore.write(Data("remove this".utf8), path: removed, root: workspace)
    let user = UUID(), assistant = UUID()
    try store.checkpoint(messageID: user, startingTurn: true)
    try AgentProjectStore.write(csvBytes, path: csv, root: workspace)
    try AgentProjectStore.write(pdfBytes, path: pdf, root: workspace)
    try files.removeItem(at: workspace.appendingPathComponent(removed))
    try store.checkpoint(messageID: assistant)
    guard let result = try store.turnResult(user: user, assistant: assistant) else {
      throw DeliverableCheckError(message: "Cannot load the completed turn")
    }
    let changed = Data("new live version".utf8)
    try AgentProjectStore.write(changed, path: csv, root: workspace)
    try files.removeItem(at: workspace.appendingPathComponent(pdf))
    _ = try store.checkpoint("Later edits to both deliverables")

    studio.setLanguage(.en)
    let originalDocument = studio.studioDocument
    defer { studio.studioDocument = originalDocument }
    var requests: [String] = []
    var destination: URL?
    let margin: CGFloat = 24
    let host = NSHostingView(rootView: AgentTurnResultRow(workspace: workspace, result: result, selectDestination: { panel in
      requests.append(panel.nameFieldStringValue)
      return destination
    }).environmentObject(studio).padding(margin).background(StudioTheme.canvas).preferredColorScheme(.dark))
    let window = NSWindow(
      contentRect: CGRect(x: 80, y: 80, width: 900, height: 420),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.makeKeyAndOrderFront(nil)
    defer { window.close() }

    func click(x: CGFloat, fromTop y: CGFloat) async throws {
      // Use the fixture's fixed width and the row's fixed-height Studio buttons.
      // Clicking the wrong action or file is detected by the expected path and bytes.
      try await Task.sleep(for: .milliseconds(125))
      host.layoutSubtreeIfNeeded()
      let y = host.isFlipped ? y : host.bounds.height - y
      let location = host.convert(CGPoint(x: x, y: y), to: nil)
      func event(_ type: NSEvent.EventType) -> NSEvent {
        NSEvent.mouseEvent(with: type, location: location, modifierFlags: [],
          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
          context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
      }
      NSApp.postEvent(event(.leftMouseUp), atStart: false)
      window.sendEvent(event(.leftMouseDown))
      if let up = NSApp.nextEvent(matching: .leftMouseUp, until: .distantPast, inMode: .default, dequeue: true) {
        window.sendEvent(up)
      }
    }
    func clickFile(_ row: Int, save: Bool) async throws {
      let top = margin + 6 + StudioPageMetrics.control / 2
      let stride = StudioPageMetrics.control + 7
      let trailing: CGFloat = save ? 66 : 156
      try await click(x: host.bounds.width - trailing, fromTop: top + CGFloat(row) * stride)
    }
    func waitFor<T>(_ message: String, _ read: () throws -> T?) async throws -> T {
      let deadline = ContinuousClock.now.advanced(by: .seconds(5))
      while ContinuousClock.now < deadline {
        if let value = try read() { return value }
        try await Task.sleep(for: .milliseconds(25))
      }
      throw DeliverableCheckError(message: message)
    }
    func quickLook(in view: NSView) -> QLPreviewView? {
      if let preview = view as? QLPreviewView,
        preview.accessibilityIdentifier() == "agent.deliverable.quicklook" { return preview }
      return view.subviews.lazy.compactMap { quickLook(in: $0) }.first
    }
    func expect(_ condition: Bool, _ message: String) throws {
      guard condition else { throw DeliverableCheckError(message: message) }
    }
    func snapshot(_ view: NSView, name: String) throws {
      guard let folder = ProcessInfo.processInfo.environment["MLXL3_STUDIO_UI_SNAPSHOTS"],
        let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
      else { return }
      view.cacheDisplay(in: view.bounds, to: bitmap)
      try bitmap.representation(using: .png, properties: [:])?.write(
        to: URL(fileURLWithPath: folder).appendingPathComponent(name))
    }

    for (row, pair) in [(csv, csvBytes), (pdf, pdfBytes)].enumerated() {
      let (name, expected) = pair
      try await clickFile(row, save: false)
      let item = try await waitFor("Preview did not open for \(name)") { () -> URL? in
        window.sheets.first?.contentView.flatMap { quickLook(in: $0)?.previewItem?.previewItemURL }
      }
      try expect(item.lastPathComponent == name && Data(contentsOf: item) == expected,
        "The preview read the live file instead of the historical \(name)")
      guard let sheet = window.sheets.first, let content = sheet.contentView else {
        throw DeliverableCheckError(message: "The file preview closed unexpectedly")
      }
      if ProcessInfo.processInfo.environment["MLXL3_STUDIO_UI_SNAPSHOTS"] != nil {
        try await Task.sleep(for: .milliseconds(400))
        try snapshot(content, name: "agent-deliverable-preview-\(item.pathExtension).png")
      }
      guard let escape = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
        timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: sheet.windowNumber,
        context: nil, characters: "\u{1B}", charactersIgnoringModifiers: "\u{1B}", isARepeat: false, keyCode: 53)
      else { throw DeliverableCheckError(message: "Cannot construct the preview's Escape shortcut") }
      sheet.sendEvent(escape)
      _ = try await waitFor("The file preview did not close") { window.sheets.isEmpty ? true : nil }

      let target = root.appendingPathComponent("Saved \(name)")
      destination = target
      try await clickFile(row, save: true)
      let saved = try await waitFor("Save did not write \(name)") {
        files.fileExists(atPath: target.path) ? try Data(contentsOf: target) : nil
      }
      try expect(saved == expected, "Save changed the historical bytes of \(name)")
    }
    destination = nil
    try await clickFile(0, save: true)
    _ = try await waitFor("The canceled Save did not reach the save panel") { requests.count == 3 ? true : nil }
    try expect(requests == [csv, pdf, csv], "Save suggested the wrong filename")
    try expect(try Data(contentsOf: workspace.appendingPathComponent(csv)) == changed
      && !files.fileExists(atPath: workspace.appendingPathComponent(pdf).path),
      "Preview, Save or cancellation changed the working copy")
    studio.studioDocument = nil
    try await click(x: margin + 55, fromTop: margin + 19 + StudioPageMetrics.control * 2.5)
    _ = try await waitFor("Review changes did not open") { studio.studioDocument }
    try expect(studio.studioDocument?.turnResult?.files.contains(where: { $0.path == removed }) == true,
      "A deleted file disappeared from the review")
    try snapshot(host, name: "agent-deliverables.png")
    print("Agent deliverables: CSV and PDF previews and exports preserve turn bytes, cancellation and deleted-file review passed")
    return 7
  }
}
