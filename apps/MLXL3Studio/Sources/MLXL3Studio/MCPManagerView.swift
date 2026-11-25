import SwiftUI

private struct MCPCheckResponse: Decodable {
    let tools: [MCPToolDescriptor]
    let errors: [String: String]
}

struct MCPManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var names: [String] = []
    @State private var selected = ""
    @State private var name = ""
    @State private var configuration = "{\n  \"url\": \"\",\n  \"enabled\": true\n}"
    @State private var error: String?
    @State private var checking = false
    @State private var checkTask: Task<Void, Never>?

    private func startNewServer() {
        selected = ""
        name = ""
        configuration = "{\n  \"url\": \"\",\n  \"enabled\": true\n}"
    }

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: "MCP", onBack: { studio.showMCPManager = false }) {
                ScrollView {
                  VStack(alignment: .leading, spacing: 0) {
                    StudioNavList {
                        ForEach(names, id: \.self) { server in
                            StudioNavRow(icon: "server.rack", title: server, selected: selected == server) {
                                selected = server
                            }
                        }
                        StudioNavRow(icon: "plus", title: L("Nouveau serveur", "New server"), selected: selected.isEmpty,
                                     action: startNewServer)
                    }
                    if names.isEmpty {
                        Text(L("Ajoutez un serveur pour connecter ses outils.", "Add a server to connect its tools."))
                            .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 18).padding(.top, 10)
                    }
                  }
                }
                .scrollIndicators(.hidden)
                .disabled(studio.isGenerating)
            }
            StudioPageColumn(spacing: 22) {
                StudioPageHeader(title: L("Serveurs MCP", "MCP servers"),
                                 detail: L("Outils connectés et autorisations par projet.", "Connected tools and project permissions.")) {
                    if checking { ProgressView().controlSize(.small) }
                    Button(L("Tester les connexions", "Test connections")) { check() }
                        .buttonStyle(.studio).disabled(checking)
                        .help(L(
                            "Tester lance les serveurs locaux avec vos permissions et contacte les serveurs distants activés.",
                            "Testing launches local servers with your permissions and contacts enabled remote servers."))
                }
                if studio.isGenerating {
                    StudioNotice(text: L(
                        "Les modifications seront disponibles à la fin de la génération.",
                        "Editing becomes available when generation finishes."))
                }
                SettingSection(
                    title: selected.isEmpty ? L("Nouveau serveur", "New server") : selected, icon: "server.rack"
                ) {
                    SettingRow(title: L("Nom du serveur", "Server name"),
                               detail: selected.isEmpty ? nil : L("Le nom d’un serveur enregistré ne change pas.", "A saved server keeps its name.")) {
                        TextField(L("Un nom unique pour ce serveur", "A unique name for this server"), text: $name)
                            .studioField().frame(width: 280).disabled(!selected.isEmpty)
                            .accessibilityLabel(L("Nom du serveur", "Server name"))
                    }
                    SettingRow(title: L("Transport", "Transport")) {
                        Picker(
                            L("Transport", "Transport"),
                            selection: Binding(
                                get: { object?["url"] != nil },
                                set: { remote in
                                    var value = object ?? [:]
                                    if remote {
                                        value.removeValue(forKey: "command")
                                        value.removeValue(forKey: "args")
                                        value["url"] = ""
                                    } else {
                                        value.removeValue(forKey: "url")
                                        value["command"] = ""
                                        value["args"] = [String]()
                                    }
                                    setObject(value)
                                })
                        ) {
                            Text("HTTPS").tag(true)
                            Text(L("Processus local", "Local process")).tag(false)
                        }.labelsHidden().pickerStyle(.segmented).fixedSize()
                    }
                    if object?["url"] != nil {
                        SettingRow(title: "URL") {
                            TextField("https://…", text: stringField("url"))
                                .studioField(monospaced: true).frame(width: 360).accessibilityLabel("URL")
                        }
                    } else {
                        SettingRow(title: L("Exécutable", "Executable")) {
                            TextField(
                                L("Chemin ou nom de l’exécutable", "Executable path or name"),
                                text: stringField("command")
                            )
                            .studioField(monospaced: true).frame(width: 360)
                            .accessibilityLabel(L("Exécutable", "Executable"))
                        }
                        SettingRow(title: L("Arguments", "Arguments"), detail: L("Un par ligne.", "One per line.")) {
                            TextField(
                                L("Arguments : un par ligne", "Arguments: one per line"),
                                text: Binding(
                                    get: {
                                        (object?["args"] as? [String] ?? []).joined(separator: "\n")
                                    },
                                    set: { value in
                                        var server = object ?? [:]
                                        server["args"] = value.isEmpty ? [] : value.components(separatedBy: "\n")
                                        setObject(server)
                                    }), axis: .vertical
                            ).lineLimit(1...4).studioField(monospaced: true).frame(width: 360)
                        }
                    }
                    SettingToggleRow(
                        title: L("Serveur activé", "Server enabled"),
                        isOn: Binding(
                            get: { object?["enabled"] as? Bool ?? true },
                            set: { value in
                                var server = object ?? [:]
                                server["enabled"] = value
                                setObject(server)
                            })
                    )
                    DisclosureGroup(
                        L("Avancé : environnement, en-têtes et JSON", "Advanced: environment, headers and JSON")
                    ) {
                        TextEditor(text: $configuration).font(.system(size: 12.5, design: .monospaced))
                            .scrollContentBackground(.hidden).frame(height: 150).padding(8)
                            .background(StudioTheme.contrast.opacity(0.045), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay { RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false) }
                            .padding(.top, 8)
                            .accessibilityLabel(L("Configuration JSON", "JSON configuration"))
                    }
                    .font(.system(size: 13, weight: .medium))
                    HStack(spacing: 8) {
                        Button(L("Enregistrer", "Save")) { save() }
                            .buttonStyle(.studio(.primary)).disabled(name.isEmpty)
                        Spacer()
                        Button(L("Retirer", "Remove"), role: .destructive) { remove() }
                            .buttonStyle(.studio(.destructive)).disabled(selected.isEmpty)
                    }
                }
                if let error, !error.isEmpty {
                    StudioNotice(kind: .warning, text: error)
                }
                SettingSection(
                    title: L("Permissions du projet", "Project permissions"), icon: "lock.shield",
                    footer: L(
                        "Un nouveau projet n’autorise aucun outil. Le bouton MCP du chat reste nécessaire pour les activer.",
                        "A new project allows no tools. The chat MCP switch must also be enabled.")
                ) {
                    Text(
                        studio.currentProject.map {
                            L("Outils autorisés pour ", "Allowed tools for ") + $0.name
                        }
                            ?? L(
                                "Choisissez un projet pour limiter ses outils.",
                                "Choose a project to restrict its tools.")
                    )
                    .font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                    ForEach(studio.mcpTools) { tool in
                        SettingToggleRow(
                            title: tool.id, detail: tool.function.description,
                            isOn: Binding(
                                get: { studio.currentProject?.allowedTools.contains(tool.id) == true },
                                set: { allowed in
                                    guard var project = studio.currentProject else { return }
                                    project.allowedTools.removeAll { $0 == tool.id }
                                    if allowed { project.allowedTools.append(tool.id) }
                                    studio.updateProject(project)
                                })
                        )
                        .disabled(studio.currentProject == nil)
                    }
                    if studio.mcpTools.isEmpty {
                        Text(
                            L(
                                "Testez une connexion pour afficher ses outils.",
                                "Test a connection to list its tools.")
                        )
                        .font(.system(size: 12.5)).foregroundStyle(StudioTheme.quiet)
                    }
                }
            }
            .disabled(studio.isGenerating)
        }
        .font(.system(size: 13))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background { StudioAurora(animated: true).background(StudioTheme.canvas) }.foregroundStyle(StudioTheme.ink)
        .onExitCommand { studio.showMCPManager = false }
        .task {
            reload()
            if let first = names.first { selected = first }
        }
        .onChange(of: selected) { loadSelection() }
        .onDisappear { checkTask?.cancel() }
    }

    private var object: [String: Any]? {
        (try? JSONSerialization.jsonObject(with: Data(configuration.utf8))) as? [String: Any]
    }
    private func setObject(_ value: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) {
            configuration = String(decoding: data, as: UTF8.self)
        }
    }
    private func stringField(_ key: String) -> Binding<String> {
        Binding(
            get: { object?[key] as? String ?? "" },
            set: { value in
                var server = object ?? [:]
                server[key] = value
                setObject(server)
            })
    }

    private func reload() {
        do {
            names =
                FileManager.default.fileExists(atPath: studio.mcpConfigurationURL.path)
                ? try MCPConfiguration.servers(at: studio.mcpConfigurationURL).keys.sorted() : []
        } catch { self.error = error.localizedDescription }
    }
    private func loadSelection() {
        guard !selected.isEmpty else { return }
        do {
            guard let server = try MCPConfiguration.servers(at: studio.mcpConfigurationURL)[selected] else { return }
            name = selected
            configuration = String(
                decoding: try JSONSerialization.data(withJSONObject: server, options: [.prettyPrinted, .sortedKeys]),
                as: UTF8.self)
        } catch { self.error = error.localizedDescription }
    }
    private func save() {
        do {
            guard selected.isEmpty || name == selected else { throw MCPSettingsError.invalid }
            try MCPConfiguration.saveServer(name: name, json: configuration, at: studio.mcpConfigurationURL)
            reload()
            selected = name
            error = nil
        } catch { self.error = error.localizedDescription }
    }
    private func remove() {
        do {
            try MCPConfiguration.saveServer(name: selected, json: nil, at: studio.mcpConfigurationURL)
            selected = ""
            name = ""
            reload()
        } catch { self.error = error.localizedDescription }
    }
    private func check() {
        guard !studio.isPreview else {
            error = L(
                "Aperçu UI : les connexions réelles sont désactivées. La configuration de test reste isolée.",
                "UI preview: real connections are disabled. Test configuration stays isolated.")
            return
        }
        checking = true
        error = nil
        checkTask = Task {
            defer { checking = false }
            do {
                let data = try await withThrowingTaskGroup(of: Data.self) { group in
                    group.addTask { try await CLICommand().output(["mcp", "check", "--json"]) }
                    group.addTask {
                        try await Task.sleep(for: .seconds(30))
                        throw URLError(.timedOut)
                    }
                    defer { group.cancelAll() }
                    return try await group.next()!
                }
                try Task.checkCancellation()
                let response = try JSONDecoder().decode(MCPCheckResponse.self, from: data)
                studio.mcpTools = response.tools
                error = response.errors.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)" }.joined(
                    separator: "\n")
                if studio.mcpEnabled { studio.reloadMCPServers() }
            } catch is CancellationError {} catch { self.error = error.localizedDescription }
        }
    }
}
