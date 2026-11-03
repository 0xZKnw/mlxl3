import SwiftUI

private struct MCPCheckResponse: Decodable {
    let tools: [MCPToolDescriptor]
    let errors: [String: String]
}

struct MCPManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.dismiss) private var dismiss
    @State private var names: [String] = []
    @State private var selected = ""
    @State private var name = ""
    @State private var configuration = "{\n  \"url\": \"https://example.com/mcp\",\n  \"enabled\": true\n}"
    @State private var error: String?
    @State private var checking = false
    @State private var checkTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(L("Serveurs et permissions MCP", "MCP servers and permissions")).font(.title2)
                Spacer()
                Button(L("Fermer", "Close")) { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text(L("Tester lance les serveurs locaux avec vos permissions et contacte les serveurs distants activés.", "Testing launches local servers with your permissions and contacts enabled remote servers.")).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading) {
                    List(names, id: \.self, selection: $selected) { Text($0).tag($0) }.frame(width: 165)
                    Button(L("Nouveau serveur", "New server")) { selected = ""; name = ""; configuration = "{\n  \"command\": \"\",\n  \"args\": [],\n  \"enabled\": true\n}" }
                }
                VStack(alignment: .leading, spacing: 10) {
                    TextField(L("Nom du serveur", "Server name"), text: $name).disabled(!selected.isEmpty)
                    Picker(L("Transport", "Transport"), selection: Binding(get: { object?["url"] != nil }, set: { remote in
                        var value = object ?? [:]
                        if remote { value.removeValue(forKey: "command"); value.removeValue(forKey: "args"); value["url"] = "https://" }
                        else { value.removeValue(forKey: "url"); value["command"] = ""; value["args"] = [String]() }
                        setObject(value)
                    })) { Text("HTTPS").tag(true); Text(L("Processus local", "Local process")).tag(false) }
                    if object?["url"] != nil {
                        TextField("URL", text: stringField("url"))
                    } else {
                        TextField(L("Exécutable", "Executable"), text: stringField("command"))
                        TextField(L("Arguments : un par ligne", "Arguments: one per line"), text: Binding(get: { (object?["args"] as? [String] ?? []).joined(separator: "\n") }, set: { value in
                            var server = object ?? [:]; server["args"] = value.isEmpty ? [] : value.components(separatedBy: "\n"); setObject(server)
                        }), axis: .vertical).lineLimit(1...3)
                    }
                    Toggle(L("Serveur activé", "Server enabled"), isOn: Binding(get: { object?["enabled"] as? Bool ?? true }, set: { value in
                        var server = object ?? [:]; server["enabled"] = value; setObject(server)
                    }))
                    DisclosureGroup(L("Avancé : environnement, en-têtes et JSON", "Advanced: environment, headers and JSON")) {
                        TextEditor(text: $configuration).font(.system(.caption, design: .monospaced)).frame(height: 90)
                    }
                    HStack {
                        Button(L("Enregistrer", "Save")) { save() }.disabled(name.isEmpty)
                        Button(L("Retirer", "Remove"), role: .destructive) { remove() }.disabled(selected.isEmpty)
                        Spacer()
                        Button(L("Tester les connexions", "Test connections")) { check() }.disabled(checking)
                        if checking { ProgressView().controlSize(.small) }
                    }
                    Divider()
                    Text(studio.currentProject.map { L("Outils autorisés pour ", "Allowed tools for ") + $0.name } ?? L("Choisissez un projet pour limiter ses outils.", "Choose a project to restrict its tools.")).font(.headline)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(studio.mcpTools) { tool in
                                Toggle(isOn: Binding(get: { studio.currentProject?.allowedTools.contains(tool.id) == true }, set: { allowed in
                                    guard var project = studio.currentProject else { return }
                                    project.allowedTools.removeAll { $0 == tool.id }
                                    if allowed { project.allowedTools.append(tool.id) }
                                    studio.updateProject(project)
                                })) {
                                    VStack(alignment: .leading) { Text(tool.id); Text(tool.function.description ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(2) }
                                }.disabled(studio.currentProject == nil)
                            }
                            if studio.mcpTools.isEmpty { Text(L("Testez une connexion pour afficher ses outils.", "Test a connection to list its tools.")).foregroundStyle(.secondary) }
                        }
                    }
                }
            }.disabled(studio.isGenerating)
            if let error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled).lineLimit(4) }
            Text(L("Un nouveau projet n’autorise aucun outil. Le bouton MCP du chat reste nécessaire pour les activer.", "A new project allows no tools. The chat MCP switch must also be enabled.")).font(.caption).foregroundStyle(.secondary)
        }.padding(24).frame(width: 850, height: 620)
        .background(StudioTheme.canvas).foregroundStyle(StudioTheme.ink)
        .task { reload() }
        .onChange(of: selected) { loadSelection() }
        .onDisappear { checkTask?.cancel() }
    }

    private var object: [String: Any]? { (try? JSONSerialization.jsonObject(with: Data(configuration.utf8))) as? [String: Any] }
    private func setObject(_ value: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) { configuration = String(decoding: data, as: UTF8.self) }
    }
    private func stringField(_ key: String) -> Binding<String> {
        Binding(get: { object?[key] as? String ?? "" }, set: { value in var server = object ?? [:]; server[key] = value; setObject(server) })
    }

    private func reload() {
        do { names = try MCPConfiguration.servers(at: studio.mcpConfigurationURL).keys.sorted() }
        catch { self.error = error.localizedDescription }
    }
    private func loadSelection() {
        guard !selected.isEmpty else { return }
        do {
            guard let server = try MCPConfiguration.servers(at: studio.mcpConfigurationURL)[selected] else { return }
            name = selected
            configuration = String(decoding: try JSONSerialization.data(withJSONObject: server, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self)
        } catch { self.error = error.localizedDescription }
    }
    private func save() {
        do {
            guard selected.isEmpty || name == selected else { throw MCPSettingsError.invalid }
            try MCPConfiguration.saveServer(name: name, json: configuration, at: studio.mcpConfigurationURL)
            reload(); selected = name; error = nil
        } catch { self.error = error.localizedDescription }
    }
    private func remove() {
        do { try MCPConfiguration.saveServer(name: selected, json: nil, at: studio.mcpConfigurationURL); selected = ""; name = ""; reload() }
        catch { self.error = error.localizedDescription }
    }
    private func check() {
        checking = true; error = nil
        checkTask = Task {
            defer { checking = false }
            do {
                let data = try await withThrowingTaskGroup(of: Data.self) { group in
                    group.addTask { try await CLICommand().output(["mcp", "check", "--json"]) }
                    group.addTask { try await Task.sleep(for: .seconds(30)); throw URLError(.timedOut) }
                    defer { group.cancelAll() }
                    return try await group.next()!
                }
                try Task.checkCancellation()
                let response = try JSONDecoder().decode(MCPCheckResponse.self, from: data)
                studio.mcpTools = response.tools
                error = response.errors.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)" }.joined(separator: "\n")
                if studio.mcpEnabled { studio.reloadMCPServers() }
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription }
        }
    }
}
