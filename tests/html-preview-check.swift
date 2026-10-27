import AppKit
import Darwin
import SwiftUI
import WebKit

@main struct HTMLPreviewCheck {
  @MainActor static func main() async throws {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.accessory)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-html-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let metadata = root.appendingPathComponent("server.json")
    let server = Process()
    server.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
    server.arguments = [CommandLine.arguments[2], metadata.path]
    try server.run()
    defer {
      if server.isRunning { server.terminate() }
      server.waitUntilExit()
    }
    for _ in 0..<1000
    where !FileManager.default.fileExists(atPath: metadata.path) && server.isRunning {
      try await Task.sleep(for: .milliseconds(10))
    }
    let info = try JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as! [String: Any]
    let base = "http://127.0.0.1:\(info["port"]!)"
    var count = 0
    var diagnosticView: WKWebView?
    func check(_ value: @autoclosure () -> Bool, _ text: String) {
      precondition(value(), text)
      count += 1
    }
    func wait(_ attempts: Int = 300, _ predicate: @MainActor () async throws -> Bool) async throws {
      for _ in 0..<attempts {
        if try await predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      if let view = diagnosticView {
        let state = try? await view.evaluateJavaScript(
          "JSON.stringify({url:document.URL,state:document.readyState,result:window.result,error:window.lastError,body:document.body?.innerHTML.slice(0,1800)})"
        )
        FileHandle.standardError.write(
          Data(
            "WebKit diagnostic: \(String(describing: state)); nativeURL \(String(describing: view.url))\n"
              .utf8))
      }
      preconditionFailure("WebKit fixture timed out")
    }
    for language in ["html", "HTML", "htm", "text/html", nil] {
      check(
        HTMLPreview.canRun(language: language, source: "<!doctype html><p>hello", completed: true),
        "HTML recognized")
      check(
        !HTMLPreview.canRun(language: language, source: "<p>hello", completed: false),
        "Streaming never executes")
    }
    check(
      !HTMLPreview.canRun(language: "js", source: "<html></html>", completed: true),
      "JS is not an HTML artifact")
    check(
      !HTMLPreview.canRun(
        language: "html", source: String(repeating: "x", count: HTMLPreview.maximumBytes + 1),
        completed: true), "Preview source is bounded")
    let workspace = root.appendingPathComponent("workspace")
    try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
    let html = workspace.appendingPathComponent("index.html")
    try "<!doctype html><h1>artifact</h1>".write(to: html, atomically: true, encoding: .utf8)
    let outside = root.appendingPathComponent("secret.html")
    try "PRIVATE-ORACLE".write(to: outside, atomically: true, encoding: .utf8)
    let link = workspace.appendingPathComponent("escape.html")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)
    let artifacts = HTMLPreview.artifactURLs(in: workspace)
    check(
      artifacts.map(\.lastPathComponent) == ["index.html"],
      "Agent artifacts exclude symlinks: \(artifacts)")
    let actual = try HTMLPreview.source(at: html, in: workspace)
    check(actual.contains("artifact"), "Real Agent HTML opens")
    let hardlink = workspace.appendingPathComponent("hardlink.html")
    try FileManager.default.linkItem(at: outside, to: hardlink)
    do {
      _ = try HTMLPreview.source(at: hardlink, in: workspace)
      preconditionFailure("Hardlink exposed host content")
    } catch { count += 1 }
    check(
      !HTMLPreview.artifactURLs(in: workspace).contains(hardlink),
      "Unsafe aliases have no preview button")
    let boundary = workspace.appendingPathComponent("limit.html")
    try Data(repeating: 97, count: HTMLPreview.maximumBytes).write(to: boundary)
    let bounded = try HTMLPreview.data(at: boundary, in: workspace)
    check(
      bounded.count == HTMLPreview.maximumBytes && bounded.first == 97 && bounded.last == 97,
      "Descriptor reads the complete maximum-size regular file")
    try Data(repeating: 97, count: HTMLPreview.maximumBytes + 1).write(to: boundary)
    do {
      _ = try HTMLPreview.data(at: boundary, in: workspace)
      preconditionFailure("Oversized resource accepted")
    } catch { count += 1 }
    let fifo = workspace.appendingPathComponent("pipe.html")
    precondition(mkfifo(fifo.path, 0o600) == 0)
    do {
      _ = try HTMLPreview.data(at: fifo, in: workspace)
      preconditionFailure("FIFO accepted as a resource")
    } catch { count += 1 }
    for invalid in [outside, link] {
      do {
        _ = try HTMLPreview.source(at: invalid, in: workspace)
        preconditionFailure("Host file exposed")
      } catch { count += 1 }
    }
    let source = #"""
      <!doctype html><meta charset="utf-8">
      <meta http-equiv="Content-Security-Policy" content="default-src * data: blob: 'unsafe-inline' 'unsafe-eval'">
      <style>body { color: rgb(12, 34, 56); }</style>
      <button id="counter">0</button><a id="escape" href="__PROBE__/navigation">escape</a>
      <script type="importmap">{"imports":{"fixture":"data:text/javascript,export%20const%20value%3D42"}}</script>
      <script>
      window.onerror = msg => window.lastError = msg;
      window.result = {inline: true, module: 0, fetch: 'pending', file: 'pending'};
      document.querySelector('#counter').onclick = e => e.target.textContent = Number(e.target.textContent) + 1;
      try { localStorage.setItem('secret', 'x'); result.storage = 'allowed'; } catch(e) { result.storage = 'blocked'; }
      try { document.cookie = 'x=1'; result.cookie = document.cookie.includes('x=1') ? 'allowed' : 'blocked'; } catch(e) { result.cookie = 'blocked'; }
      fetch('__PROBE__/fetch', {mode:'no-cors'}).then(() => result.fetch='allowed').catch(() => result.fetch='blocked');
      fetch('__SECRET__').then(() => result.file='allowed').catch(() => result.file='blocked');
      try { result.popup = window.open('__PROBE__/popup') ? 'allowed' : 'blocked'; } catch(e) { result.popup = 'blocked'; }
      </script>
      <script type="module">import {value} from 'fixture'; result.module = value;</script>
      <img src="__PROBE__/image"><iframe src="__SECRET__"></iframe>
      """#.replacingOccurrences(of: "__PROBE__", with: base).replacingOccurrences(
        of: "__SECRET__", with: outside.absoluteString)
    let view = try await HTMLPreview.makeWebView(source: source)
    diagnosticView = view
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 800, height: 600), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.contentView = view
    window.orderFront(nil)
    defer {
      view.navigationDelegate = nil
      view.stopLoading()
      view.loadHTMLString("", baseURL: nil)
      window.contentView = nil
      window.orderOut(nil)
    }
    try await wait { (try? await view.evaluateJavaScript("window.result?.module")) as? Int == 42 }
    try await wait {
      (try? await view.evaluateJavaScript("window.result?.fetch")) as? String == "blocked"
    }
    let serialized = try await view.evaluateJavaScript("JSON.stringify(result)") as! String
    let result = try JSONSerialization.jsonObject(with: Data(serialized.utf8)) as! [String: Any]
    check(
      result["inline"] as? Bool == true && result["module"] as? Int == 42,
      "Inline JS, ES module and import map execute")
    for field in ["fetch", "file", "storage", "cookie", "popup"] {
      check(result[field] as? String == "blocked", "Isolation: \(field)")
    }
    let clicked =
      try await view.evaluateJavaScript(
        "document.querySelector('#counter').click(); document.querySelector('#counter').textContent"
      ) as? String
    check(clicked == "1", "HTML remains interactive")
    let color =
      try await view.evaluateJavaScript("getComputedStyle(document.body).color") as? String
    check(color == "rgb(12, 34, 56)", "Inline CSS renders")
    _ = try await view.evaluateJavaScript("document.querySelector('#escape').click(); undefined")
    try await Task.sleep(for: .milliseconds(200))
    check(view.url == HTMLPreview.pageURL, "Top navigation remains inside preview")
    let (requestData, _) = try await URLSession.shared.data(from: URL(string: base + "/requests")!)
    let paths = try JSONSerialization.jsonObject(with: requestData) as! [String]
    check(
      paths.isEmpty, "Independent server oracle: no fetch, image, popup or navigation reached LAN")
    check(
      !view.configuration.websiteDataStore.isPersistent, "Preview has no persistent browser store")
    let project = workspace.appendingPathComponent("project")
    try FileManager.default.createDirectory(at: project, withIntermediateDirectories: true)
    let projectHTML = project.appendingPathComponent("index.html")
    let localSource = #"""
      <!doctype html><link rel="stylesheet" href="./style.css"><body>
      <script type="module" src="./main.mjs"></script>
      <script>fetch('./escape.json').then(r => r.text()).then(t => document.body.dataset.escape=t).catch(() => document.body.dataset.escape='blocked');</script>
      """#
    try localSource.write(to: projectHTML, atomically: true, encoding: .utf8)
    try "import {value} from './utils.mjs'; document.body.dataset.module = value;".write(
      to: project.appendingPathComponent("main.mjs"), atomically: true, encoding: .utf8)
    try "export const value=73;".write(
      to: project.appendingPathComponent("utils.mjs"), atomically: true, encoding: .utf8)
    try "body { color: rgb(77, 88, 99); }".write(
      to: project.appendingPathComponent("style.css"), atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(
      at: project.appendingPathComponent("escape.json"), withDestinationURL: outside)
    let local = try await HTMLPreview.makeWebView(
      source: localSource, workspace: workspace, file: projectHTML)
    window.contentView = local
    diagnosticView = local
    defer {
      local.navigationDelegate = nil
      local.stopLoading()
      local.loadHTMLString("", baseURL: nil)
    }
    try await wait {
      (try? await local.evaluateJavaScript("document.body?.dataset.module")) as? String == "73"
    }
    try await wait {
      (try? await local.evaluateJavaScript("document.body?.dataset.escape")) as? String == "blocked"
    }
    let localColor =
      try await local.evaluateJavaScript("getComputedStyle(document.body).color") as? String
    check(
      localColor == "rgb(77, 88, 99)",
      "Sibling CSS and transitive ES modules work in an Agent folder")
    count += 1
    if let generated = ProcessInfo.processInfo.environment["MLXL3_HTML_GENERATED_FILE"] {
      let file = workspace.appendingPathComponent("qwen.html")
      try FileManager.default.copyItem(at: URL(fileURLWithPath: generated), to: file)
      let source = try HTMLPreview.source(at: file, in: workspace)
      let generatedView = try await HTMLPreview.makeWebView(
        source: source, workspace: workspace, file: file)
      window.contentView = generatedView
      diagnosticView = generatedView
      defer {
        generatedView.navigationDelegate = nil
        generatedView.stopLoading()
        generatedView.loadHTMLString("", baseURL: nil)
      }
      try await wait {
        (try? await generatedView.evaluateJavaScript(
          "document.querySelector('button')?.textContent")) as? String == "MLXL3-QWEN-OK"
      }
      let clicked =
        try await generatedView.evaluateJavaScript(
          "document.querySelector('button').click(); document.querySelector('button').textContent")
        as? String
      check(
        clicked == "42",
        "HTML created by the actual Qwen Agent executes inside the production preview")
    }
    if ProcessInfo.processInfo.environment["MLXL3_HTML_CDN_CHECK"] == "1" {
      let cdn = try await HTMLPreview.makeWebView(
        source: #"""
          <!doctype html><body><script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.179.1/build/three.module.js"}}</script>
          <script type="module">
          import * as THREE from 'three';
          const v = new THREE.Vector3(1,2,3);
          const renderer = new THREE.WebGLRenderer({antialias:false}); renderer.setSize(32,32);
          document.body.appendChild(renderer.domElement);
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(50,1,0.1,10); camera.position.z=2;
          const geometry = new THREE.BoxGeometry(1,1,1);
          const material = new THREE.MeshBasicMaterial({color:0x00ff00});
          scene.add(new THREE.Mesh(geometry,material)); renderer.render(scene,camera);
          const gl=renderer.getContext(), pixel=new Uint8Array(4);
          gl.readPixels(16,16,1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);
          document.body.dataset.result = THREE.REVISION + ':' + v.lengthSq() + ':' + (pixel[0]<15 && pixel[1]>200 && pixel[2]<15 && pixel[3]===255 && gl.getError()===0);
          geometry.dispose(); material.dispose(); renderer.dispose();
          </script>
          """#)
      window.contentView = cdn
      diagnosticView = cdn
      defer {
        cdn.navigationDelegate = nil
        cdn.stopLoading()
        cdn.loadHTMLString("", baseURL: nil)
      }
      try await wait(1500) {
        (try? await cdn.evaluateJavaScript("document.body?.dataset.result")) as? String
          == "179:14:true"
      }
      count += 1
    }
    print(
      "HTML preview checks passed: \(count); CDN \(ProcessInfo.processInfo.environment["MLXL3_HTML_CDN_CHECK"] == "1" ? "tested" : "not requested")"
    )
  }
}
