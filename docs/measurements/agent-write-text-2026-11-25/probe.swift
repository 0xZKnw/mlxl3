import Foundation

@main
struct Probe {
  static func main() throws {
    var permissions = AgentPermissions()
    permissions.harness = .javascript
    let description = AgentJavaScript.toolDescription(permissions, wall: 33, persistent: true)
    let schema = try JSONSerialization.data(withJSONObject: AgentJavaScript.inputSchema, options: [.sortedKeys])
    let out: [String: String] = ["description": description, "schema": String(decoding: schema, as: UTF8.self)]
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]))
  }
}
