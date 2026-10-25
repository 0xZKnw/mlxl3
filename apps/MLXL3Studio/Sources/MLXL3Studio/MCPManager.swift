import CoreFoundation
import Darwin
import Foundation

struct MCPToolDescriptor: Decodable, Identifiable, Sendable {
    struct Function: Decodable, Sendable {
        let name: String
        let description: String?
    }
    let function: Function
    var id: String { function.name }
}

extension MCPConfiguration {
    static func servers(at url: URL) throws -> [String: [String: Any]] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard object?["version"] as? Int == 1, let servers = object?["mcpServers"] as? [String: [String: Any]] else {
            throw MCPSettingsError.invalid
        }
        return servers
    }

    static func saveServer(name: String, json: String?, at url: URL) throws {
        guard !name.isEmpty, name.utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [46, 95, 45].contains($0) }) else {
            throw MCPSettingsError.invalid
        }
        var server: [String: Any]?
        if let json {
            guard let value = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any] else { throw MCPSettingsError.invalid }
            for key in ["command", "url", "cwd"] {
                if let field = value[key], !(field is String) { throw MCPSettingsError.invalid }
            }
            let command = value["command"] as? String ?? ""
            if let endpoint = value["url"] as? String {
                guard command.isEmpty, let parsed = URL(string: endpoint), let host = parsed.host, !host.isEmpty,
                    parsed.user == nil, parsed.password == nil,
                    parsed.scheme == "https" || (parsed.scheme == "http" && ["localhost", "127.0.0.1", "::1", "[::1]"].contains(host)) else { throw MCPSettingsError.invalid }
            } else if command.isEmpty { throw MCPSettingsError.invalid }
            if let args = value["args"], !(args is [String]) { throw MCPSettingsError.invalid }
            if let env = value["env"], !(env is [String: String]) { throw MCPSettingsError.invalid }
            if let enabled = value["enabled"], CFGetTypeID(enabled as CFTypeRef) != CFBooleanGetTypeID() { throw MCPSettingsError.invalid }
            if let headers = value["headers"] {
                guard let strings = headers as? [String: String], strings.allSatisfy({ !$0.key.unicodeScalars.contains(where: { $0 == "\r" || $0 == "\n" || $0 == ":" }) && !$0.value.unicodeScalars.contains(where: { $0 == "\r" || $0 == "\n" }) }) else { throw MCPSettingsError.invalid }
            }
            server = value
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = open(url.deletingPathExtension().appendingPathExtension("lock").path, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw MCPSettingsError.invalid }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw MCPSettingsError.invalid }
        defer { flock(descriptor, LOCK_UN) }
        var object: [String: Any] = ["version": 1, "mcpServers": [String: Any]()]
        if FileManager.default.fileExists(atPath: url.path) {
            guard let existing = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any], existing["version"] as? Int == 1,
                existing["mcpServers"] is [String: [String: Any]] else { throw MCPSettingsError.invalid }
            object = existing
        }
        var servers = object["mcpServers"] as! [String: [String: Any]]
        servers[name] = server
        object["mcpServers"] = servers
        // Keep credentials private even before the atomic replacement.
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".mcp-\(UUID()).tmp")
        let output = open(temporary.path, O_CREAT | O_EXCL | O_WRONLY, S_IRUSR | S_IWUSR)
        guard output >= 0 else { throw CocoaError(.fileWriteUnknown) }
        let handle = FileHandle(fileDescriptor: output, closeOnDealloc: true)
        defer { try? handle.close(); unlink(temporary.path) }
        try handle.write(contentsOf: JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]))
        try handle.synchronize()
        try handle.close()
        guard rename(temporary.path, url.path) == 0 else { throw CocoaError(.fileWriteUnknown) }
    }
}

enum MCPSettingsError: LocalizedError {
    case invalid
    var errorDescription: String? { L("Configuration MCP invalide. Utilisez une URL HTTPS ou une commande avec des arguments JSON valides.", "Invalid MCP configuration. Use an HTTPS URL or a command with valid JSON arguments.") }
}
