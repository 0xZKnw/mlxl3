// Concatenated with MarkdownResponseView.swift to exercise the real table renderer.
import AppKit
import SwiftUI

@MainActor enum MarkdownTableRegressionCheck {
  static func run() async throws {
    func require(_ condition: Bool, _ message: String) throws {
      guard condition else {
        throw NSError(
          domain: "MarkdownTableRegression", code: 1,
          userInfo: [NSLocalizedDescriptionKey: message])
      }
    }
    let details = [
      "**ALPHA first line**", "BRAVO café 👩🏽‍💻",
      "CHARLIE " + String(repeating: "wrapped content ", count: 8),
      "DELTA fourth line", "ECHO fifth line", "FOXTROT sixth line", "GOLF seventh line",
      "HOTEL last line $x^2$",
    ]
    func source(_ pieces: [InlineMathPiece]) -> String {
      pieces.map { piece in
        switch piece {
        case .prose(let text): return text
        case .math(_, let raw, _): return raw
        }
      }.joined()
    }
    let inlineCases = [
      ("<br>", "\n"), ("a<br>b<br/>c<br />d<BR>E", "a\nb\nc\nd\nE"),
      ("<Br\t/>end<BR >", "\nend\n"), ("**a<br>b**", "**a\nb**"),
      (#"`<br>`<br>after"#, "`<br>`\nafter"),
      (#"``a`<br>``<br/>b"#, "``a`<br>``\nb"),
      (#"\<br>literal<br>é👩🏽‍💻"#, "\\<br>literal\né👩🏽‍💻"),
      (#"$\text{<br>}$<br>x"#, "$\\text{<br>}$\nx"),
      (#"\(<br>\)<br>\[<br>\]"#, "\\(<br>\\)\n\\[<br>\\]"),
      ("&lt;br&gt;<br>next", "&lt;br&gt;\nnext"),
      ("$5<br>$10", "$5\n$10"), ("`unclosed<br>next", "`unclosed\nnext"),
      (
        "<break><bravo><br class='x'><br / ></br><br", "<break><bravo><br class='x'><br / ></br><br"
      ),
      ("line\nline", "line\nline"), ("", ""),
    ]
    for (input, expected) in inlineCases {
      try require(
        source(InlineMathParser.parse(input, htmlLineBreaks: true)) == expected,
        "Table break/code/math boundary changed: \(input)")
      try require(source(InlineMathParser.parse(input)) == input, "Non-table source changed")
    }
    func pixels(header: String, body: String, width: CGFloat, scale: CGFloat, streaming: Bool)
      async throws -> (Int, Int, Data)
    {
      let markdown =
        "| Category | \(header) | Sources |\n| --- | :---: | ---: |\n"
        + "| First row | \(body) | **Reference** |\n| NEXT ROW | No overlap | `literal` |"
      // The parser must preserve raw cells; rendering must not rewrite stored/copied Markdown.
      let blocks = MarkdownParser.parse(markdown)
      guard blocks.count == 1,
        case .table(let headers, let alignments, let rows) = blocks[0].kind
      else {
        throw NSError(domain: "MarkdownTableRegression", code: 2)
      }
      try require(rows.count == 2 && rows[0][1] == body, "Parser changed table source")
      return try await nativePixels(
        MarkdownTableView(
          header: headers, alignments: alignments, rows: rows,
          streaming: streaming, textScale: scale), width: width,
        label: "table-\(Int(width))-\(scale)-\(streaming)-html")
    }

    var cases = 0
    for width: CGFloat in [360, 800, 1200, 2000] {
      for scale: CGFloat in [0.85, 1, 1.4, 1.8] {
        for streaming in [false, true] {
          let actual = try await pixels(
            header: "Details<br />more", body: details.joined(separator: "<br>"),
            width: width, scale: scale, streaming: streaming)
          // Independent oracle: real newlines, supplied without the HTML replacement path.
          // Newlines cannot be put in a Markdown row, so use the actual parsed table data below.
          let reference = try await referencePixels(
            width: width, scale: scale, streaming: streaming,
            header: "Details\nmore", details: details.joined(separator: "\n"))
          print(
            "Table \(width)/\(scale)/\(streaming): actual=\(actual.1), reference=\(reference.1)")
          try require(actual.1 >= Int(120 * scale), "Multiline table height collapsed: \(actual.1)")
          try require(actual == reference, "HTML breaks differ from native newline cells")
          cases += 1
        }
      }
    }
    print(
      "Table rendering checks passed: \(cases) native multiline/scale/streaming cases, \(inlineCases.count) inline boundaries"
    )
  }

  private static func referencePixels(
    width: CGFloat, scale: CGFloat, streaming: Bool,
    header: String, details: String
  ) async throws -> (Int, Int, Data) {
    try await nativePixels(
      MarkdownTableView(
        header: ["Category", header, "Sources"],
        alignments: [.leading, .center, .trailing],
        rows: [["First row", details, "**Reference**"], ["NEXT ROW", "No overlap", "`literal`"]],
        streaming: streaming, textScale: scale), width: width,
      label: "table-\(Int(width))-\(scale)-\(streaming)-reference")
  }

  private static func nativePixels<V: View>(_ table: V, width: CGFloat, label: String)
    async throws -> (Int, Int, Data)
  {
    let view = table.frame(width: width).fixedSize(horizontal: false, vertical: true)
      .padding(16).background(Color.black).preferredColorScheme(.dark)
    let host = NSHostingView(rootView: view)
    host.setFrameSize(NSSize(width: width + 32, height: 1000))
    let size = host.fittingSize
    let window = NSWindow(
      contentRect: NSRect(x: 30, y: 30, width: size.width, height: size.height),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = host
    window.orderFront(nil)
    defer {
      window.contentView = nil
      window.orderOut(nil)
    }
    try await Task.sleep(for: .milliseconds(200))
    host.layoutSubtreeIfNeeded()
    guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
      throw NSError(domain: "MarkdownTableRegression", code: 4)
    }
    host.cacheDisplay(in: host.bounds, to: bitmap)
    guard let image = bitmap.cgImage else {
      throw NSError(domain: "MarkdownTableRegression", code: 5)
    }
    if label == "table-1200-1.4-false-html" {
      let text = try ResponseUICheck.recognizedCPU(image).uppercased()
      guard ["ALPHA", "HOTEL", "NEXT ROW"].allSatisfy(text.contains) else {
        throw NSError(
          domain: "MarkdownTableRegression", code: 7,
          userInfo: [NSLocalizedDescriptionKey: "Clipped table rows: \(text)"])
      }
    }
    if let folder = ProcessInfo.processInfo.environment["MLXL3_TABLE_SNAPSHOTS"] {
      try bitmap.representation(using: .png, properties: [:])!.write(
        to: URL(fileURLWithPath: folder).appendingPathComponent(label + ".png"))
    }
    var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
    let context = CGContext(
      data: &bytes, width: image.width, height: image.height,
      bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    let ink = stride(from: 0, to: bytes.count, by: 4).filter {
      Int(bytes[$0]) + Int(bytes[$0 + 1]) + Int(bytes[$0 + 2]) > 300
    }.count
    guard ink > 100 else {
      throw NSError(
        domain: "MarkdownTableRegression", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Empty table capture: \(label), ink=\(ink)"])
    }
    if label.contains("-1200-1.4-") || label.contains("-2000-1.4-") {
      // Sample padding, away from text: every column must share every horizontal border.
      let columns = 3
      let pixelScale = CGFloat(image.width) / (width + 32)
      let xs = (0..<columns).map { Int((27 + CGFloat($0) * width / CGFloat(columns)) * pixelScale) }
      let delta = Int(2 * pixelScale)
      func brightness(_ x: Int, _ y: Int) -> Int {
        let index = (y * image.width + x) * 4
        return Int(bytes[index]) + Int(bytes[index + 1]) + Int(bytes[index + 2])
      }
      var boundaries = 0
      for y in Int(30 * pixelScale)..<(image.height - Int(30 * pixelScale)) {
        let edges = xs.map { x in
          brightness(x, y) - min(brightness(x, y - delta), brightness(x, y + delta)) > 12
        }
        if edges.contains(true) {
          guard edges.allSatisfy({ $0 }) else {
            throw NSError(domain: "MarkdownTableRegression", code: 8,
              userInfo: [NSLocalizedDescriptionKey: "Staggered row border: \(label), y=\(y), edges=\(edges)"])
          }
          boundaries += 1
        }
      }
      guard boundaries >= 2 else {
        throw NSError(domain: "MarkdownTableRegression", code: 9,
          userInfo: [NSLocalizedDescriptionKey: "No full-width row borders: \(label)"])
      }
    }
    return (image.width, image.height, Data(bytes))
  }
}
