import Darwin
import Foundation

@main
struct AgentSandboxCheck {
  static func main() throws {
    let args = CommandLine.arguments
    if (args.count == 4 || args.count == 6) && args[1] == "--agent-command" {
      AgentSandbox.commandMode(root: args[2], command: args[3], seconds: args.count == 6 ? Double(args[4]) ?? 0 : 5, writable: args.count == 4 || args[5] == "write")
    }
    if (args.count == 3 || args.count == 4) && args[1] == "--agent-tools" { AgentSandbox.serve(root: args[2], permissionsPath: args.count == 4 ? args[3] : nil) }
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-agent-\(UUID())"
    ).resolvingSymlinksInPath()
    let root = temporary.appendingPathComponent("workspace \"quoted\"")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let secret = temporary.appendingPathComponent("private.txt")
    try Data("PRIVATE-CONTENT-DO-NOT-LEAK".utf8).write(to: secret)
    func quote(_ value: String) -> String {
      "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
    var count = 0
    func check(_ condition: Bool, _ message: String) {
      guard condition else { fatalError(message) }
      count += 1
    }
    func run(_ command: String, seconds: Double = 5) throws -> AgentSandbox.Result {
      try AgentSandbox.execute(root: root, command: command, seconds: seconds)
    }
    let success = try run("printf hello > greeting.txt; cat greeting.txt")
    check(
      success.status == 0 && !success.interrupted && success.output == "hello",
      "real sandbox must execute and round-trip a nonempty file: \(success)")
    let read = try run("cat \(quote(secret.path))")
    check(
      read.status != 0 && !read.output.contains("PRIVATE-CONTENT"), "absolute read escaped sandbox")
    let write = try run("printf changed > \(quote(secret.path))")
    let hardlink = try run("ln \(quote(secret.path)) hardlink; cat hardlink")
    check(
      hardlink.status != 0 && !hardlink.output.contains("PRIVATE-CONTENT")
        && !FileManager.default.fileExists(atPath: root.appendingPathComponent("hardlink").path),
      "hardlink source escaped sandbox")
    let unchanged = String(decoding: try Data(contentsOf: secret), as: UTF8.self)
    check(
      write.status != 0 && unchanged == "PRIVATE-CONTENT-DO-NOT-LEAK",
      "absolute write escaped sandbox")
    try FileManager.default.createSymbolicLink(
      at: root.appendingPathComponent("escape"), withDestinationURL: secret)
    let symlink = try run("cat escape; printf changed > escape")
    check(
      symlink.status != 0 && !symlink.output.contains("PRIVATE-CONTENT"), "symlink escaped sandbox")
    let network = try run("/usr/bin/curl --max-time 1 --noproxy '*' http://127.0.0.1:9")
    check(
      network.status != 0 && network.output.localizedCaseInsensitiveContains("permitted"),
      "network should be refused by Seatbelt: \(network.output)")
    let inherited = try run("/bin/sh -c \(quote("cat " + quote(secret.path)))")
    check(
      inherited.status != 0 && !inherited.output.contains("PRIVATE-CONTENT"),
      "child process escaped sandbox")
    let environment = try run("/usr/bin/env")
    let canonicalRoot = try AgentSandbox.canonicalRoot(root)
    check(
      !environment.output.contains("MLXL3_EXECUTABLE=") && !environment.output.contains("TOKEN=")
        && environment.output.contains("HOME=\(canonicalRoot.path)"), "host environment leaked")
    let timeout = try run("sleep 30", seconds: 0.2)
    check(timeout.interrupted, "silent process did not time out")
    let large = try run("yes data")
    check(
      large.interrupted && large.output.utf8.count <= AgentSandbox.maximumOutput,
      "output limit failed")
    let descendants = try run("sleep 30 & echo $! > child.pid; wait", seconds: 0.2)
    check(descendants.interrupted, "background child did not time out")
    let pid = Int32(
      String(
        decoding: try Data(contentsOf: root.appendingPathComponent("child.pid")), as: UTF8.self
      ).trimmingCharacters(in: .whitespacesAndNewlines))!
    usleep(100_000)
    check(kill(pid, 0) != 0, "background child survived cleanup")
    let sessions = try run(
      "/usr/bin/perl -MPOSIX -e 'my $a=POSIX::setsid(); my $b=POSIX::setpgid(0,0); print \"$a \", (defined($b) ? $b : -1), \"\\n\"'"
    )
    check(
      sessions.status == 0 && sessions.output == "-1 -1\n",
      "descendants can escape the process group: \(sessions)")
    let spawn = try run("/usr/bin/perl -e 'print syscall(244,0,0,0,0,0,0), \" $!\\n\"'")
    check(
      spawn.output == "-1 Operation not permitted\n",
      "posix_spawn group attributes remain reachable: \(spawn)")
    let limits = try run("ulimit -t 1000; ulimit -f 100000; ulimit -n 1000")
    check(
      limits.status != 0 && limits.output.contains("permitted"),
      "child can raise resource hard limits")
    for command in [
      "", String(repeating: "a", count: AgentSandbox.maximumCommand + 1), "echo\0oops",
    ] {
      do {
        _ = try run(command)
        fatalError("invalid command accepted")
      } catch { count += 1 }
    }
    for seconds in [0.0, -1, 30.1, .infinity, .nan] {
      do {
        _ = try run("true", seconds: seconds)
        fatalError("invalid timeout accepted")
      } catch { count += 1 }
    }
    do {
      _ = try AgentSandbox.profile(root: URL(fileURLWithPath: "/"))
      fatalError("root filesystem accepted")
    } catch { count += 1 }
    let profile = try AgentSandbox.profile(root: root)
    check(
      profile.contains("(deny default)") && !profile.contains("(allow network"),
      "policy is not deny-default")
    print(
      "Agent sandbox checks passed: \(count), real commands, file/network/child isolation, environment, limits, process cleanup and invalid input"
    )
  }
}
