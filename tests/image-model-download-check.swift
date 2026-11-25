import Foundation
import Synchronization

private struct CheckError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

private final class FixtureProtocol: URLProtocol, @unchecked Sendable {
    enum Reply: Sendable {
        case bytes(Int, Data)
        case hold
        case partial(Data, Int)
    }
    struct State: Sendable {
        var replies: [String: Reply] = [:]
        var requests: [String] = []
    }
    static let state = Mutex(State())
    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "fixture.invalid" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let reply = Self.state.withLock { state in
            state.requests.append(url.lastPathComponent)
            return state.replies[url.lastPathComponent]
        }
        switch reply {
        case let .bytes(status, data):
            let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Length": String(data.count)])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        case let .partial(data, expected):
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Length": String(expected)])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
        case .hold: break
        case nil: client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
        }
    }
    override func stopLoading() {}
}

@main
struct ImageModelDownloadCheck {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw CheckError(message) }
    }
    static func rejects(_ operation: @Sendable () async throws -> Void) async throws {
        do { try await operation() }
        catch { return }
        throw CheckError("Expected installation to fail")
    }
    static func file(_ name: String, _ size: Int64, _ sha: String) -> ImageModelDownloadFile {
        .init(filename: name, url: URL(string: "https://fixture.invalid/files/")!.appendingPathComponent(name),
              size: size, sha256: sha)
    }
    private static func setReplies(_ replies: [String: FixtureProtocol.Reply]) {
        FixtureProtocol.state.withLock { $0 = .init(replies: replies) }
    }
    static var requests: [String] { FixtureProtocol.state.withLock { $0.requests } }

    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data(("\(error)\n").utf8))
            exit(EXIT_FAILURE)
        }
    }

    static func run() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("image-model-download-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FixtureProtocol.self]
        config.timeoutIntervalForRequest = 4
        config.timeoutIntervalForResource = 6
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let service = ImageModelDownload(session: session)
        let abc = Data("abc".utf8)
        let hello = Data("hello".utf8)
        let abcSHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        let helloSHA = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        let a = file("shard", 3, abcSHA)
        let b = file("manifest.json", 5, helloSHA)
        let files = [a, b]
        let replies: [String: FixtureProtocol.Reply] = [a.filename: .bytes(200, abc), b.filename: .bytes(200, hello)]
        let destination = root.appendingPathComponent("Model", isDirectory: true)

        setReplies(replies)
        let progress = Mutex<[Int64]>([])
        try await service.install(files: files, to: destination) { count, total in
            progress.withLock { $0.append(total == 8 ? count : -1) }
        }
        let seen = progress.withLock { $0 }
        try require(seen.first == 0 && seen.last == 8 && seen.sorted() == seen, "Progress must cover both files")
        try require(try Data(contentsOf: destination.appendingPathComponent(a.filename)) == abc, "Missing payload")
        try require(try Data(contentsOf: destination.appendingPathComponent(b.filename)) == hello, "Missing manifest")
        try require(requests == files.map(\.filename), "Unexpected request order")
        try await service.install(files: files, to: destination)
        try require(requests == files.map(\.filename), "Repeated installation downloaded existing files")
        try require(try fm.contentsOfDirectory(atPath: root.path) == ["Model"], "An extra installation copy remains")

        // A wrong response (including one of the correct length) cannot publish an incomplete model.
        for (index, bad) in [FixtureProtocol.Reply.bytes(200, Data("HELLO".utf8)),
                             .bytes(200, Data("hi".utf8)), .bytes(404, hello)].enumerated() {
            let name = "Retry\(index)"
            let dest = root.appendingPathComponent(name)
            let stage = root.appendingPathComponent("." + name + ".partial")
            setReplies([a.filename: .bytes(200, abc), b.filename: bad])
            try await rejects { try await service.install(files: files, to: dest) }
            try require(!fm.fileExists(atPath: dest.path), "Invalid checkpoint was published")
            try require(try fm.contentsOfDirectory(atPath: stage.path) == [a.filename], "A corrupt response was kept")
            setReplies(replies)
            try await service.install(files: files, to: dest)
            try require(requests == [b.filename], "A verified file was requested twice")
            try require(!fm.fileExists(atPath: stage.path), "Resume stage was not removed")
        }

        // An interrupted process can leave an unverified response. An external
        // install might also finish a model while verified chunks remain staged.
        let obsolete = root.appendingPathComponent(".Model.partial")
        try fm.createDirectory(at: obsolete, withIntermediateDirectories: false)
        try hello.write(to: obsolete.appendingPathComponent(".incoming"))
        try abc.write(to: obsolete.appendingPathComponent(a.filename))
        setReplies([:])
        try await service.install(files: files, to: destination)
        try require(requests.isEmpty && !fm.fileExists(atPath: obsolete.path), "Duplicate staging files remain after local reuse")

        let extra = destination.appendingPathComponent("user-notes")
        try abc.write(to: extra)
        try Data("HELLO".utf8).write(to: destination.appendingPathComponent(b.filename))
        setReplies(replies)
        try await service.install(files: files, to: destination)
        try require(requests == [b.filename] && (try Data(contentsOf: extra)) == abc, "Repair lost unrelated data or redownloaded a valid file")

        let later = root.appendingPathComponent("Cancelled")
        setReplies([a.filename: .bytes(200, abc), b.filename: .hold])
        let transfer = Task { try await service.install(files: files, to: later) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !requests.contains(b.filename) && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        guard requests.contains(b.filename) else { transfer.cancel(); throw CheckError("Fixture did not start the delayed transfer") }
        try await rejects { try await service.install(files: files, to: later) }
        transfer.cancel()
        do { try await transfer.value; throw CheckError("Cancelled install succeeded") }
        catch is CancellationError {}
        try require(!fm.fileExists(atPath: later.path), "Cancelled download was published")
        setReplies(replies)
        try await service.install(files: files, to: later)
        try require(requests == [b.filename], "Cancelled install lost its verified shard")

        let writing = file("large-shard", 524288, abcSHA)
        setReplies([writing.filename: .partial(Data(repeating: 0, count: 262144), 524288)])
        let written = Mutex<Int64>(0)
        let incomplete = root.appendingPathComponent("InFlight")
        let active = Task {
            try await service.install(files: [writing], to: incomplete) { count, _ in written.withLock { $0 = count } }
        }
        let progressDeadline = ContinuousClock.now.advanced(by: .seconds(3))
        while written.withLock({ $0 }) == 0 && ContinuousClock.now < progressDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        active.cancel()
        do { try await active.value; throw CheckError("Interrupted transfer succeeded") }
        catch is CancellationError {}
        try require(written.withLock { $0 } == 262144 && !fm.fileExists(atPath: incomplete.path), "No live byte progress before EOF")

        let unsafe = root.appendingPathComponent("Unsafe")
        setReplies(replies)
        for invalid in [[file("../shard", 3, abcSHA)], [a, file("SHARD", 3, abcSHA)],
                        [file("weight", Int64.max, abcSHA), b], [file("weight", 0, abcSHA)],
                        [file("weight", 3, "corrupt")], []] {
            try await rejects { try await service.install(files: invalid, to: unsafe) }
        }
        let outside = root.appendingPathComponent("Outside")
        try fm.createDirectory(at: outside, withIntermediateDirectories: false)
        try fm.createSymbolicLink(at: root.appendingPathComponent(".Unsafe.partial"), withDestinationURL: outside)
        try await rejects { try await service.install(files: files, to: unsafe) }
        try require(requests.isEmpty && (try fm.contentsOfDirectory(atPath: outside.path)).isEmpty, "Unsafe catalog or staging directory was used")
        print("Image model download passed: install, offline reuse, three invalid responses and resume, repair, cancellation and concurrent request, live byte progress, unsafe paths and catalog")
    }
}
