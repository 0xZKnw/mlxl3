import AppKit
import Darwin
import Foundation
import SwiftUI

@main struct StudiosCheck {
  @MainActor static func main() async throws {
    if URL(fileURLWithPath: CommandLine.arguments[0]).lastPathComponent == "quant-fixture" {
      try quantizationFixture()
      return
    }
    let files = FileManager.default
    let temporary =
      ProcessInfo.processInfo.environment["MLXL3_STUDIOS_TMP"].map { URL(fileURLWithPath: $0) }
      ?? files.temporaryDirectory
    let root = temporary.appendingPathComponent("mlxl3-studios-\(UUID())").resolvingSymlinksInPath()
    try files.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? files.removeItem(at: root) }
    var checks = 0
    func check(_ value: Bool, _ message: String) {
      precondition(value, message)
      checks += 1
    }
    for path in ["../outside", "/absolute", "a/../b", "a//b", ".git/config", "x\ny", ""] {
      check(!AgentProjectStore.validPath(path), "Invalid relative path accepted")
    }
    for path in ["index.html", "nested/main.js", "café/été.css"] {
      check(AgentProjectStore.validPath(path), "Valid relative path refused")
    }
    let source = root.appendingPathComponent("original")
    let workspace = root.appendingPathComponent("task/files")
    try files.createDirectory(at: source, withIntermediateDirectories: true)
    try Data("<!doctype html><script src='main.js'></script><p>before</p>".utf8).write(
      to: source.appendingPathComponent("index.html"))
    try Data("window.result = 1".utf8).write(to: source.appendingPathComponent("main.js"))
    try Data("PRIVATE".utf8).write(to: source.appendingPathComponent(".env"))
    try Data("#!/bin/sh\nprintf ok".utf8).write(to: source.appendingPathComponent("test.sh"))
    try files.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: source.appendingPathComponent("test.sh").path)
    try files.createSymbolicLink(
      at: source.appendingPathComponent("alias.txt"),
      withDestinationURL: source.appendingPathComponent(".env"))
    let store = AgentProjectStore(workspace: workspace)
    try store.importProject(source)
    let initial = try store.state()
    check(initial.skippedFiles == 2, "Private file and alias must be reported as skipped")
    check(
      !files.fileExists(atPath: workspace.appendingPathComponent(".env").path),
      "Private source copied")
    check(try store.versions().count == 1, "Actual Git checkpoint missing")
    check(
      (try files.attributesOfItem(atPath: workspace.appendingPathComponent("test.sh").path)[
        .posixPermissions] as? NSNumber)?.intValue == 0o755, "Executable mode lost on import")
    let first = try store.versions()[0].id
    let boundary = UUID()
    try store.checkpoint(messageID: boundary)
    var restricted = try store.state().permissions
    restricted.commandSeconds = 2
    restricted.run = false
    restricted.write = false
    try store.setPermissions(restricted)
    try AgentProjectStore.write(
      Data("<!doctype html><p>after</p>".utf8), path: "index.html", root: workspace)
    try AgentProjectStore.write(Data("window.result = 2".utf8), path: "main.js", root: workspace)
    try AgentProjectStore.write(Data("new file".utf8), path: "nested/new.txt", root: workspace)
    check(
      try String(contentsOf: source.appendingPathComponent("index.html"), encoding: .utf8).contains(
        "before"), "Agent changed originals before review")
    let second = try store.checkpoint("Edit files")
    let branchURL = root.appendingPathComponent("historical-branch/files")
    try store.fork(to: branchURL, at: boundary, retaining: [boundary])
    let historical = AgentProjectStore(workspace: branchURL)
    check(try historical.contents("main.js", revision: historical.versions()[0].id)
      == Data("window.result = 1".utf8), "Fork copied latest files instead of historical boundary")
    check(try historical.state().permissions == restricted, "Fork relaxed tool permissions")
    check(try initial.source != nil && historical.state().source == initial.source,
      "Fork lost original project")
    check(!files.fileExists(atPath: branchURL.appendingPathComponent("nested/new.txt").path),
      "Historical fork contains later file")
    check((try files.attributesOfItem(atPath: branchURL.appendingPathComponent("test.sh").path)[.posixPermissions]
      as? NSNumber)?.intValue == 0o755, "Fork lost executable mode")
    try AgentProjectStore.write(Data("branch only".utf8), path: "main.js", root: branchURL)
    check(try Data(contentsOf: workspace.appendingPathComponent("main.js")) == Data("window.result = 2".utf8),
      "Fork mutation leaked to parent")
    do {
      try store.fork(to: root.appendingPathComponent("missing/files"), at: UUID(), retaining: [])
      preconditionFailure("Missing historical checkpoint silently used current workspace")
    } catch { checks += 1 }
    let diff = try store.diff(from: first, to: second)
    check(
      diff.contains("-window.result = 1") && diff.contains("+window.result = 2"),
      "Diff misses actual source change")
    let previous = try store.materialize(first)
    check(
      try String(contentsOf: previous.appendingPathComponent("main.js"), encoding: .utf8)
        == "window.result = 1", "Old preview uses current JS")
    let zip = root.appendingPathComponent("project.zip")
    try store.exportZIP(zip, revision: second)
    let zipBytes = try Data(contentsOf: zip)
    check(zipBytes.starts(with: [0x50, 0x4b]), "Export is not an actual ZIP")
    check(
      !String(decoding: zipBytes, as: UTF8.self).contains("history.git"),
      "Private Git history exported")
    try store.apply(reviewedRevision: try store.versions().first!.id)
    check(
      try String(contentsOf: source.appendingPathComponent("main.js"), encoding: .utf8)
        == "window.result = 2", "Reviewed changes not applied")
    check(
      try String(contentsOf: source.appendingPathComponent("nested/new.txt"), encoding: .utf8)
        == "new file", "New source file missing")
    check(
      try String(contentsOf: source.appendingPathComponent(".env"), encoding: .utf8) == "PRIVATE",
      "Excluded private file changed")
    // Exercise apply's actual tree/mode path, not a source-text checklist.
    try AgentProjectStore.write(Data("window.result = 3".utf8), path: "main.js", root: workspace, mode: 0o755)
    try AgentProjectStore.write(Data("#!/bin/sh\nprintf ok".utf8), path: "test.sh", root: workspace, mode: 0o644)
    try AgentProjectStore.write(Data("#!/bin/sh\ntrue".utf8), path: "new.sh", root: workspace, mode: 0o755)
    let modeRevision = try store.checkpoint("Reviewed executable changes")
    try store.apply(reviewedRevision: modeRevision)
    for (path, mode) in [("main.js", 0o755), ("test.sh", 0o644), ("new.sh", 0o755)] {
      check((try files.attributesOfItem(atPath: source.appendingPathComponent(path).path)[.posixPermissions]
        as? NSNumber)?.intValue == mode, "Apply ignored reviewed executable mode: \(path)")
    }
    try AgentProjectStore.write(Data("changed".utf8), path: "new.sh", root: workspace, mode: 0o755)
    let conflictingMode = try store.checkpoint("Content with external chmod")
    try files.setAttributes([.posixPermissions: 0o644], ofItemAtPath: source.appendingPathComponent("new.sh").path)
    do {
      try store.apply(reviewedRevision: conflictingMode)
      preconditionFailure("Concurrent original chmod was overwritten")
    } catch { checks += 1 }
    try files.setAttributes([.posixPermissions: 0o755], ofItemAtPath: source.appendingPathComponent("new.sh").path)
    try store.apply(reviewedRevision: conflictingMode)
    // A parent-path collision fails only during writes, after a deletion and chmod.
    try files.removeItem(at: workspace.appendingPathComponent("main.js"))
    try AgentProjectStore.write(Data("changed again".utf8), path: "new.sh", root: workspace, mode: 0o644)
    try AgentProjectStore.write(Data("child".utf8), path: "zz-blocked/child", root: workspace)
    try Data("not a directory".utf8).write(to: source.appendingPathComponent("zz-blocked"))
    let failedApply = try store.checkpoint("Rollback bytes and modes")
    do {
      try store.apply(reviewedRevision: failedApply)
      preconditionFailure("Blocked parent path unexpectedly applied")
    } catch { checks += 1 }
    check(try Data(contentsOf: source.appendingPathComponent("main.js")) == Data("window.result = 3".utf8),
      "Rollback failed to restore deleted executable bytes")
    for path in ["main.js", "new.sh"] {
      check((try files.attributesOfItem(atPath: source.appendingPathComponent(path).path)[.posixPermissions]
        as? NSNumber)?.intValue == 0o755, "Rollback lost executable mode")
    }
    check(try store.state().baseline == conflictingMode, "Failed apply advanced baseline")
    try store.restore(conflictingMode)
    try AgentProjectStore.write(Data("pending".utf8), path: "index.html", root: workspace)
    try AgentProjectStore.write(Data("pending".utf8), path: "main.js", root: workspace)
    _ = try store.checkpoint("Conflicting edits")
    try Data("USER-EDIT".utf8).write(to: source.appendingPathComponent("main.js"))
    do {
      try store.apply(reviewedRevision: try store.versions().first!.id)
      preconditionFailure("Source conflict was overwritten")
    } catch { checks += 1 }
    check(
      try String(contentsOf: source.appendingPathComponent("index.html"), encoding: .utf8).contains(
        "after"), "Conflict check partially applied earlier files")
    try store.restore(first)
    check(
      try String(contentsOf: workspace.appendingPathComponent("main.js"), encoding: .utf8)
        == "window.result = 1", "Restore failed")
    check(try store.versions().contains(where: { $0.id == second }), "Restore erased later history")
    try AgentProjectStore.write(Data("*.txt".utf8), path: ".gitignore", root: workspace)
    try AgentProjectStore.write(Data("visible".utf8), path: "ignored.txt", root: workspace)
    let ignored = try store.checkpoint("Ignore rules cannot hide files")
    check(
      try store.contents("ignored.txt", revision: ignored) == Data("visible".utf8),
      "Agent hid a changed file with .gitignore")
    try files.createSymbolicLink(
      at: workspace.appendingPathComponent("escape"), withDestinationURL: source)
    do {
      try AgentProjectStore.write(Data("BAD".utf8), path: "escape/main.js", root: workspace)
      preconditionFailure("Descriptor-relative write followed a symlink")
    } catch { checks += 1 }
    check(
      try String(contentsOf: source.appendingPathComponent("main.js"), encoding: .utf8)
        == "USER-EDIT", "Symlink write escaped")
    do {
      _ = try store.checkpoint("Unsafe alias")
      preconditionFailure("Checkpoint accepted a symlink")
    } catch { checks += 1 }
    let denied = try AgentSandbox.execute(root: workspace, command: "cat ../studio/project.json")
    check(denied.status != 0, "Model read private checkpoint/permission metadata")
    let readonly = try AgentSandbox.execute(
      root: workspace, command: "printf BAD > index.html", writable: false)
    check(readonly.status != 0, "Run bypassed disabled writes")
    check(
      GeneratedHTML.documents("```html\n<p>run</p>\n```").count == 1
        && GeneratedHTML.documents("```js\nalert(1)\n```").isEmpty,
      "Generated artifact extraction is incorrect")
    let worker = root.appendingPathComponent("quant-fixture")
    try files.copyItem(at: URL(fileURLWithPath: CommandLine.arguments[0]), to: worker)
    let quantizer = QuantizationModel(
      worker: worker, jobsDirectory: root.appendingPathComponent("jobs"))
    func wait(_ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<350 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("Quantization controller did not finish within 7 seconds")
    }
    for malformed in ["array", "incomplete", "invalid"] {
      quantizer.input = "malformed/" + malformed
      quantizer.inspect()
      try await wait { !quantizer.inspecting }
      check(quantizer.error != nil && quantizer.source == nil, "Malformed probe did not fail gracefully")
    }
    quantizer.input = "fixture/model"
    quantizer.inspect()
    try await wait { !quantizer.inspecting }
    check(
      quantizer.source?.conversionSupported == true, "Real worker probe did not reach controller")
    quantizer.input = "changed/model"
    check(quantizer.source == nil, "Editing the source left an old model selected")
    quantizer.inspect()
    try await wait { !quantizer.inspecting }
    quantizer.settings.bits = 3.5
    var prepared = false
    let completed = root.appendingPathComponent("converted")
    quantizer.convert(to: completed) { prepared = true }
    try await wait { !quantizer.converting }
    check(
      prepared && quantizer.output == completed && quantizer.progress == 1,
      "Controller lost prepare/settings/completion boundary")
    let failing = root.appendingPathComponent("output-fail")
    quantizer.convert(to: failing) {}
    try await wait { !quantizer.converting }
    check(
      quantizer.output == nil && quantizer.error?.contains("fixture conversion failure") == true,
      "Failed conversion appeared installable or lost its error")
    let hanging = root.appendingPathComponent("output-hang")
    quantizer.convert(to: hanging) {}
    let pidFile = root.appendingPathComponent("worker.pid")
    try await wait { files.fileExists(atPath: pidFile.path) }
    let pid = Int32(try String(contentsOf: pidFile, encoding: .utf8))!
    quantizer.cancel()
    check(quantizer.converting, "GPU ownership released before worker exit")
    try await wait { !quantizer.converting }
    check(
      kill(pid, 0) != 0 && errno == ESRCH && quantizer.output == nil,
      "Ignored SIGTERM worker survived Stop or became installable")
    if let path = ProcessInfo.processInfo.environment["MLXL3_STUDIOS_UI_SNAPSHOTS"] {
      _ = NSApplication.shared
      let prefsName = "io.mlxl3.studios-ui." + UUID().uuidString
      let prefs = UserDefaults(suiteName: prefsName)!
      defer { prefs.removePersistentDomain(forName: prefsName) }
      let studio = StudioModel(
        conversationFileURL: root.appendingPathComponent("ui-history.json"), isPreview: true,
        preferences: prefs)
      let cleanSource = root.appendingPathComponent("ui-source")
      try files.createDirectory(at: cleanSource, withIntermediateDirectories: true)
      try Data(
        "<!doctype html><html><body><h1>Agent project</h1><p>Isolated project preview.</p></body></html>"
          .utf8
      ).write(to: cleanSource.appendingPathComponent("index.html"))
      let clean = AgentProjectStore(workspace: root.appendingPathComponent("ui-project/files"))
      try files.createDirectory(at: clean.workspace, withIntermediateDirectories: true)
      _ = try clean.importProject(cleanSource)
      let views: [(String, AnyView)] = [
        ("agent", AnyView(AgentStudioView(workspace: clean.workspace))),
        ("quantization", AnyView(QuantizationStudioView(quantizer: quantizer))),
      ]
      let folder = URL(fileURLWithPath: path)
      try files.createDirectory(at: folder, withIntermediateDirectories: true)
      for (name, view) in views {
        let host = NSHostingView(
          rootView: view.environmentObject(studio).background(StudioTheme.canvas)
            .preferredColorScheme(.dark))
        let window = NSWindow(
          contentRect: NSRect(x: 70, y: 70, width: 1100, height: 720), styleMask: [.titled],
          backing: .buffered, defer: false)
        window.contentView = host
        window.makeKeyAndOrderFront(nil)
        try await Task.sleep(for: .milliseconds(700))
        host.layoutSubtreeIfNeeded()
        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])!.write(
          to: folder.appendingPathComponent(name + ".png"))
        window.orderOut(nil)
      }
    }
    print(
      "Studios checks passed: \(checks), actual Git, safe copy/edit/conflicts/restore, historical assets, ZIP, permissions and generated HTML extraction"
    )
  }

  @MainActor private static func quantizationFixture() throws {
    let request =
      try JSONSerialization.jsonObject(with: FileHandle.standardInput.readDataToEndOfFile())
      as! [String: Any]
    func emit(_ value: [String: Any]) throws {
      var data = try JSONSerialization.data(withJSONObject: value)
      data.append(10)
      try FileHandle.standardOutput.write(contentsOf: data)
    }
    if request["operation"] as? String == "probe" {
      if let source = request["source"] as? String, source.hasPrefix("malformed/") {
        let response = source.hasSuffix("array") ? "[]\n" : source.hasSuffix("incomplete") ? "{}\n" : "not-json\n"
        try FileHandle.standardOutput.write(contentsOf: Data(response.utf8))
        return
      }
      try emit([
        "type": "result",
        "result": [
          "source": request["source"]!, "revision": "local", "fingerprint": "fixture",
          "model_type": "lfm2", "inference_supported": true, "conversion_supported": true,
          "reason": "fixture", "source_bytes": 1234, "parameters": 1024,
          "modules": ["model.layers.0"], "header_checked": true, "tied_head": true,
        ],
      ])
    } else {
      let output = URL(fileURLWithPath: request["output"] as! String)
      if output.lastPathComponent.contains("fail") {
        try emit(["type": "error", "text": "fixture conversion failure"])
        exit(1)
      }
      if output.lastPathComponent.contains("hang") {
        signal(SIGTERM, SIG_IGN)
        try Data(String(getpid()).utf8).write(
          to: output.deletingLastPathComponent().appendingPathComponent("worker.pid"))
        while true { sleep(1) }
      }
      let settings = request["settings"] as! [String: Any]
      precondition(settings["bits"] as? Double == 3.5)
      try emit(["type": "progress", "phase": "convert", "completed": 1, "total": 2])
      try emit(["type": "result", "result": ["status": "complete", "output": output.path]])
    }
  }
}
