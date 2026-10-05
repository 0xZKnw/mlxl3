import Foundation

@main struct MCPConfigurationCheck {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-mcp-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("nested/mcp.json")
        try MCPConfiguration.ensureExa(at: file)
        let first = try Data(contentsOf: file)
        let object = try JSONSerialization.jsonObject(with: first) as! [String: Any]
        let servers = object["mcpServers"] as! [String: Any]
        let exa = servers["exa"] as! [String: Any]
        precondition(exa["url"] as? String == "https://mcp.exa.ai/mcp" && exa["enabled"] as? Bool == true)
        try MCPConfiguration.ensureExa(at: file)
        let second = try Data(contentsOf: file)
        precondition(first == second, "Repeated startup rewrote the configuration")
        let old = #"{"version":1,"custom":"preserved","mcpServers":{"local":{"command":"fixture","args":["test"],"enabled":false}}}"#
        try Data(old.utf8).write(to: file)
        try MCPConfiguration.ensureExa(at: file)
        let migrated = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        precondition(migrated["custom"] as? String == "preserved")
        let saved = migrated["mcpServers"] as! [String: Any]
        precondition((saved["local"] as! [String: Any])["enabled"] as? Bool == false && saved["exa"] != nil)
        for existing in [#"{"url":"https://custom.invalid/mcp","enabled":false}"#, #"{"command":"fixture","args":["exa"],"enabled":true}"#] {
            let data = Data(("{\"version\":1,\"mcpServers\":{\"exa\":" + existing + "}}").utf8)
            try data.write(to: file)
            try MCPConfiguration.ensureExa(at: file)
            let after = try Data(contentsOf: file)
            precondition(after == data, "Customized or disabled Exa was replaced")
        }
        for invalid in ["invalid", "[]", "{}", #"{"version":2,"mcpServers":{}}"#, #"{"version":true,"mcpServers":{}}"#, #"{"version":1,"mcpServers":[]}"#] {
            let data = Data(invalid.utf8)
            try data.write(to: file)
            do { try MCPConfiguration.ensureExa(at: file); preconditionFailure("Invalid configuration was accepted") }
            catch { }
            let after = try Data(contentsOf: file)
            precondition(after == data, "Invalid configuration was overwritten")
        }
        print("MCP configuration checks passed: fresh install, migration, idempotence, custom/disabled Exa, malformed files preserved")
    }
}
