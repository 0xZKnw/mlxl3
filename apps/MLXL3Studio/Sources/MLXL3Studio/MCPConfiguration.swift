import Foundation

enum MCPConfiguration {
    private struct Header: Decodable { let version: Int }

    static func ensureExa(at url: URL) throws {
        var object: [String: Any]
        if FileManager.default.fileExists(atPath: url.path) {
            let data = try Data(contentsOf: url)
            guard try JSONDecoder().decode(Header.self, from: data).version == 1,
                  let existing = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  existing["mcpServers"] is [String: Any] else {
                throw NSError(domain: "MLXL3.MCP", code: 1, userInfo: [NSLocalizedDescriptionKey:
                    L("Configuration MCP invalide : le fichier a été conservé.", "Invalid MCP configuration: the file was preserved.")])
            }
            object = existing
        } else {
            object = ["version": 1, "mcpServers": [String: Any]()]
        }
        var servers = object["mcpServers"] as! [String: Any]
        guard servers["exa"] == nil else { return }
        servers["exa"] = ["url": "https://mcp.exa.ai/mcp", "enabled": true]
        object["mcpServers"] = servers
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}
