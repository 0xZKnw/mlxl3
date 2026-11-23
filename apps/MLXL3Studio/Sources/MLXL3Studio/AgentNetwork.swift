import Darwin
import Foundation

/// `fetch` and `download` of Agent mode, made by the host: HTTPS GET to hosts the user allowed,
/// each hop resolved, checked and pinned to a public address. With Internet on, commands also
/// reach the network directly (`AgentSandbox.profile`); the JavaScript runtime never does.
enum AgentNetwork {
  /// Hosts of a new task: all public hosts. Commands have the network directly when Internet
  /// is on, so a narrower list would only restrict `fetch`; it can still be narrowed per task.
  static let defaultHosts = ["*"]
  static let maximumFetchBytes = 2 << 20
  static let maximumDownloadBytes = 100 << 20
  static let maximumRedirects = 5
  static let curl = "/usr/bin/curl"

  struct Refusal: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
  }

  /// One request, pinned: `url` is fetched from `address`, whatever `host` resolves to later.
  struct Target: Equatable, Sendable {
    let url: String
    let host: String
    let address: String
  }

  struct Response: Sendable {
    var status = 0
    var type = ""
    /// Absolute URL of a redirect (not followed by the transport).
    var redirect = ""
    var bytes = 0
    var error: String?
  }

  struct Fetched: Equatable, Sendable {
    let status: Int
    let url: String
    let type: String
    let bytes: Int
  }

  /// Writes the body of `target` to `output`, at most `limit` bytes, within `seconds`.
  typealias Transport = @Sendable (Target, URL, Int, Double) -> Response

  // MARK: - Policy

  /// Entries: `github.com` (that host), `*.github.com` (its subdomains), `*` (any host).
  static func hostAllowed(_ host: String, list: [String]) -> Bool {
    let host = host.lowercased()
    return list.contains { raw in
      let entry = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      if entry == "*" { return true }
      if entry.hasPrefix("*."), entry.count > 2 {
        let suffix = String(entry.dropFirst())
        return host.hasSuffix(suffix) && host.count > suffix.count
      }
      return !entry.isEmpty && host == entry
    }
  }

  /// The host list as the user typed it: commas, spaces or lines, lowercased, unique.
  static func parseHosts(_ text: String) -> [String] {
    var seen = Set<String>()
    return text.split(whereSeparator: { $0 == "," || $0.isWhitespace }).compactMap { part in
      var entry = String(part).lowercased()
      if entry.hasSuffix(".") { entry.removeLast() }
      guard !entry.isEmpty, entry.count <= 253,
        entry.unicodeScalars.allSatisfy({
          CharacterSet.alphanumerics.contains($0) || ".-*".unicodeScalars.contains($0)
        }), seen.insert(entry).inserted
      else { return nil }
      return entry
    }
  }

  static func isPublic(ipv4 bytes: [UInt8]) -> Bool {
    guard bytes.count == 4 else { return false }
    let (a, b, c) = (bytes[0], bytes[1], bytes[2])
    switch a {
    case 0, 10, 127: return false  // this network, private, loopback
    case 100 where (64...127).contains(b): return false  // carrier-grade NAT
    case 169 where b == 254: return false  // link-local, cloud metadata
    case 172 where (16...31).contains(b): return false  // private
    case 192 where b == 0 && (c == 0 || c == 2): return false  // IETF, TEST-NET-1
    case 192 where b == 88 && c == 99: return false  // 6to4 relay
    case 192 where b == 168: return false  // private
    case 198 where b == 18 || b == 19: return false  // benchmarking
    case 198 where b == 51 && c == 100: return false  // TEST-NET-2
    case 203 where b == 0 && c == 113: return false  // TEST-NET-3
    case 224...255: return false  // multicast, reserved, broadcast
    default: return true
    }
  }

  static func isPublic(ipv6 bytes: [UInt8]) -> Bool {
    guard bytes.count == 16 else { return false }
    let embedded = Array(bytes[12..<16])
    if bytes[0..<10].allSatisfy({ $0 == 0 }) {
      // ::, ::1, IPv4-compatible (deprecated) and IPv4-mapped: judged by the IPv4 inside.
      if bytes[10] == 0xFF && bytes[11] == 0xFF { return isPublic(ipv4: embedded) }
      return false
    }
    if Array(bytes[0..<12]) == [0, 0x64, 0xFF, 0x9B, 0, 0, 0, 0, 0, 0, 0, 0] {
      return isPublic(ipv4: embedded)  // NAT64
    }
    if bytes[0] == 0x20 && bytes[1] == 0x02 { return isPublic(ipv4: Array(bytes[2..<6])) }  // 6to4
    if bytes[0] == 0x20 && bytes[1] == 0x01 && bytes[2] < 0x02 { return false }  // 2001::/23
    if bytes[0] == 0x20 && bytes[1] == 0x01 && bytes[2] == 0x0D && bytes[3] == 0xB8 { return false }  // docs
    if bytes[0] == 0x01 && bytes[1] == 0 && bytes[2..<8].allSatisfy({ $0 == 0 }) { return false }  // discard
    if bytes[0] & 0xFE == 0xFC { return false }  // unique local
    if bytes[0] == 0xFE && bytes[1] & 0xC0 != 0 { return false }  // link-local, site-local
    if bytes[0] == 0xFF { return false }  // multicast
    return bytes[0] & 0xE0 == 0x20  // global unicast 2000::/3 only
  }

  /// Numeric address text, IPv4 or IPv6 (no brackets).
  static func isPublic(address: String) -> Bool {
    var v4 = in_addr()
    if inet_pton(AF_INET, address, &v4) == 1 {
      return isPublic(ipv4: withUnsafeBytes(of: v4) { Array($0) })
    }
    var v6 = in6_addr()
    if inet_pton(AF_INET6, address, &v6) == 1 {
      return isPublic(ipv6: withUnsafeBytes(of: v6) { Array($0) })
    }
    return false
  }

  /// An HTTPS URL on port 443, without credentials, to an allowed host.
  static func check(_ text: String, hosts: [String]) throws -> (url: URL, host: String) {
    guard text.utf8.count <= 4096, let url = URL(string: text), let scheme = url.scheme?.lowercased()
    else { throw Refusal("not a URL: \(text.prefix(120))") }
    guard scheme == "https" else { throw Refusal("only https:// URLs are allowed") }
    guard url.user == nil, url.password == nil else { throw Refusal("credentials in URLs are refused") }
    guard url.port == nil || url.port == 443 else { throw Refusal("only port 443 is allowed") }
    guard let host = url.host?.lowercased(), !host.isEmpty else { throw Refusal("the URL has no host") }
    // `curl --resolve` pins `host` exactly: `github.com.` would escape the pinned address.
    guard !host.hasSuffix(".") else { throw Refusal("remove the trailing dot from \(host)") }
    guard hostAllowed(host, list: hosts) else {
      throw Refusal("\(host) is not in the allowed hosts (Sandbox, Internet)")
    }
    return (url, host)
  }

  /// The address to connect to: every address of `host` must be public.
  static func resolve(_ host: String) throws -> String {
    var hints = addrinfo()
    hints.ai_family = AF_UNSPEC
    hints.ai_socktype = SOCK_STREAM
    var list: UnsafeMutablePointer<addrinfo>?
    let status = getaddrinfo(host, "443", &hints, &list)
    guard status == 0, let first = list else {
      throw Refusal("cannot resolve \(host): \(String(cString: gai_strerror(status)))")
    }
    defer { freeaddrinfo(first) }
    var addresses: [String] = []
    var node: UnsafeMutablePointer<addrinfo>? = first
    while let current = node {
      var name = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      if getnameinfo(
        current.pointee.ai_addr, current.pointee.ai_addrlen, &name, socklen_t(name.count), nil, 0,
        NI_NUMERICHOST) == 0
      {
        var text = String(decoding: name.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        if let zone = text.firstIndex(of: "%") { text = String(text[..<zone]) }
        addresses.append(text)
      }
      node = current.pointee.ai_next
    }
    guard let chosen = addresses.first else { throw Refusal("cannot resolve \(host)") }
    // One private answer is enough to refuse: a rebinding name may alternate.
    guard addresses.allSatisfy(isPublic(address:)) else {
      throw Refusal("\(host) resolves to a local or private address")
    }
    return chosen
  }

  // MARK: - Requests

  /// GET `text`, following at most `maximumRedirects` redirects, each one checked again.
  static func get(
    _ text: String, hosts: [String], deadline: Double, limit: Int, output: URL,
    transport: Transport = curlTransport, resolver: (String) throws -> String = resolve
  ) throws -> Fetched {
    var current = text
    for _ in 0...maximumRedirects {
      let (url, host) = try check(current, hosts: hosts)
      let address = try resolver(host)
      let remaining = deadline - ProcessInfo.processInfo.systemUptime
      guard remaining > 0.5 else { throw Refusal("no time left in this call") }
      let response = transport(
        Target(url: url.absoluteString, host: host, address: address), output, limit, remaining - 0.3)
      if let error = response.error { throw Refusal(error) }
      if (300...399).contains(response.status), !response.redirect.isEmpty {
        current = response.redirect
        continue
      }
      return Fetched(status: response.status, url: url.absoluteString, type: response.type, bytes: response.bytes)
    }
    throw Refusal("more than \(maximumRedirects) redirects")
  }

  /// Downloads into the project: the body lands at the root under a random name, then a
  /// sandboxed command moves it, so a symbolic link out of the project fails at the kernel.
  static func download(
    _ text: String, to path: String, root: URL, hosts: [String], deadline: Double,
    transport: Transport = curlTransport, resolver: (String) throws -> String = resolve
  ) throws -> Fetched {
    guard AgentSandbox.validPath(path) else { throw Refusal("use a relative path inside the project") }
    let root = try AgentSandbox.canonicalRoot(root)
    let name = ".mlxl3-download-\(UUID().uuidString)"
    let temporary = root.appendingPathComponent(name)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let fetched = try get(
      text, hosts: hosts, deadline: deadline - 1, limit: maximumDownloadBytes, output: temporary,
      transport: transport, resolver: resolver)
    guard (200...299).contains(fetched.status) else {
      throw Refusal("HTTP \(fetched.status) for \(fetched.url); nothing was saved")
    }
    guard FileManager.default.fileExists(atPath: temporary.path) else { throw Refusal("nothing was downloaded") }
    func quote(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'" }
    let parent = (path as NSString).deletingLastPathComponent
    let remaining = deadline - ProcessInfo.processInfo.systemUptime
    guard remaining > 0.2 else { throw Refusal("no time left to save \(path)") }
    let moved = try AgentSandbox.execute(
      root: root,
      command: (parent.isEmpty ? "" : "mkdir -p -- \(quote(parent)) && ") + "mv -f -- \(quote(name)) \(quote(path))",
      seconds: min(5, remaining), writable: true, limit: 4096)
    guard moved.status == 0, !moved.interrupted else {
      throw Refusal("cannot write \(path): \(moved.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(200))")
    }
    return fetched
  }

  /// `/usr/bin/curl` pinned to the checked address: no `.curlrc`, proxy, cookies or
  /// redirects; the body is watched so that neither size nor time can run over.
  static let curlTransport: Transport = { target, output, limit, seconds in
    let pinned = target.address.contains(":") ? "[\(target.address)]" : target.address
    let process = Process()
    process.executableURL = URL(fileURLWithPath: curl)
    process.arguments = [
      "-q", "--silent", "--show-error", "--globoff", "--proto", "=https", "--max-redirs", "0",
      "--resolve", "\(target.host):443:\(pinned)", "--connect-timeout", "5",
      "--max-time", String(format: "%.1f", max(0.5, seconds)), "--max-filesize", String(limit),
      "--noproxy", "*", "--user-agent", "MLXL3-Agent/1", "--output", output.path,
      "--write-out", "%{http_code}\n%{content_type}\n%{redirect_url}\n%{size_download}", "--", target.url,
    ]
    process.environment = ["PATH": "/usr/bin:/bin"]
    let standardOutput = Pipe()
    let standardError = Pipe()
    process.standardOutput = standardOutput
    process.standardError = standardError
    process.standardInput = FileHandle.nullDevice
    do { try process.run() } catch { return Response(error: "curl did not start") }
    let stop = ProcessInfo.processInfo.systemUptime + seconds + 1
    var overrun: String?
    while process.isRunning {
      let size = (try? FileManager.default.attributesOfItem(atPath: output.path)[.size] as? Int) ?? 0
      if size > limit {
        overrun = "larger than \(limit >> 20) MiB"
      } else if ProcessInfo.processInfo.systemUptime > stop {
        overrun = "timed out"
      }
      if overrun != nil {
        process.terminate()
        break
      }
      usleep(20_000)
    }
    process.waitUntilExit()
    let fields = String(decoding: standardOutput.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
      .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    if let overrun { return Response(error: "\(target.host): \(overrun)") }
    guard process.terminationStatus == 0 else {
      let detail = String(decoding: standardError.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines)
      switch process.terminationStatus {
      case 63: return Response(error: "\(target.host): larger than \(limit >> 20) MiB")
      case 28: return Response(error: "\(target.host): timed out")
      default: return Response(error: detail.isEmpty ? "\(target.host): request failed" : String(detail.prefix(300)))
      }
    }
    return Response(
      status: fields.count > 0 ? Int(fields[0]) ?? 0 : 0, type: fields.count > 1 ? fields[1] : "",
      redirect: fields.count > 2 ? fields[2] : "", bytes: fields.count > 3 ? Int(fields[3]) ?? 0 : 0)
  }

  /// Text the program can read: a textual type, or bytes that are valid UTF-8.
  static func text(_ data: Data, type: String) -> String? {
    let type = type.lowercased()
    let textual =
      type.hasPrefix("text/") || type.contains("json") || type.contains("xml")
      || type.contains("javascript") || type.contains("yaml") || type.contains("toml")
    if let decoded = String(data: data, encoding: .utf8), !decoded.contains("\0") { return decoded }
    return textual ? String(decoding: data, as: UTF8.self) : nil
  }
}
