import AppKit
import Foundation
import SwiftUI

@main struct StudioWorkspaceCheck {
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
    for name in ["a.txt", "[a].txt"] {
      try Data("Initial text\n".utf8).write(to: literalSource.appendingPathComponent(name))
    }
    let literalStore = AgentProjectStore(
      workspace: root.appendingPathComponent("literal-task/files"))
    try literalStore.importProject(literalSource)
    let literalBefore = try literalStore.versions().first!.id
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
    let oldID = studio.selectedConversationID
    await studio.openAgentProject(source)
    check(
      studio.selectedConversationID != oldID && studio.currentConversation?.agent == true
        && studio.studioDocument != nil,
      "Open project did not create a new isolated Agent task")
    check(
      studio.conversations.contains(where: { $0.id == oldID }),
      "Open project discarded previous task")
    let uiWorkspace = studio.agentWorkspaceURL!
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
}
