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

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                StudioSidebarHeader(title: "MCP")
                List(names, id: \.self, selection: $selected) { server in
                    Label(server, systemImage: "server.rack").tag(server).padding(.vertical, 6)
                }
                .listStyle(.sidebar).scrollContentBackground(.hidden).disabled(studio.isGenerating)
                if names.isEmpty {
                    Text(L("Ajoutez un serveur pour connecter ses outils.", "Add a server to connect its tools."))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                        .padding(.horizontal, 22).padding(.bottom, 18)
                }
                Button {
                    selected = ""
                    name = ""
                    configuration = "{\n  \"url\": \"\",\n  \"enabled\": true\n}"
                } label: {
                    Label(L("Nouveau serveur", "New server"), systemImage: "plus")
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 12).frame(height: 36)
                }.buttonStyle(StudioToolbarStyle(materialOpacity: 0.76))
                    .disabled(studio.isGenerating)
                    .padding(.horizontal, 18).padding(.bottom, 18)
                StudioBackButton(action: { studio.showMCPManager = false })
                    .padding(.horizontal, 18).padding(.bottom, 18)
            }
            .frame(width: studio.sidebarWidth.points, alignment: .leading)
            .modifier(StudioSidebarSurface())
                ScrollView {
                    VStack(alignment: .leading, spacing: 25) {
                        StudioPanelHeader(title: L("Serveurs MCP", "MCP servers"), detail: L("Outils connectés et autorisations par projet.", "Connected tools and project permissions."))
                        SettingCard(
                            title: selected.isEmpty ? L("Nouveau serveur", "New server") : selected, icon: "server.rack"
                        ) {
                            Text(L("Nom du serveur", "Server name")).fontWeight(.medium)
                            TextField(L("Un nom unique pour ce serveur", "A unique name for this server"), text: $name)
                                .textFieldStyle(.roundedBorder).disabled(!selected.isEmpty)
                            Text(L("Transport", "Transport")).fontWeight(.medium)
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
                            }.labelsHidden().pickerStyle(.segmented)
                            if object?["url"] != nil {
                                Text("URL").fontWeight(.medium)
                                TextField("https://…", text: stringField("url"))
                                    .textFieldStyle(.roundedBorder).accessibilityLabel("URL")
                            } else {
                                Text(L("Exécutable", "Executable")).fontWeight(.medium)
                                TextField(
                                    L("Chemin ou nom de l’exécutable", "Executable path or name"),
                                    text: stringField("command")
                                )
                                .textFieldStyle(.roundedBorder)
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
                                ).lineLimit(1...4).textFieldStyle(.roundedBorder)
                            }
                            Toggle(
                                L("Serveur activé", "Server enabled"),
                                isOn: Binding(
                                    get: { object?["enabled"] as? Bool ?? true },
                                    set: { value in
                                        var server = object ?? [:]
                                        server["enabled"] = value
                                        setObject(server)
                                    })
                            ).toggleStyle(.switch)
                            DisclosureGroup(
                                L("Avancé : environnement, en-têtes et JSON", "Advanced: environment, headers and JSON")
                            ) {
                                TextEditor(text: $configuration).font(.system(size: 13, design: .monospaced))
                                    .scrollContentBackground(.hidden).frame(height: 140).padding(10)
                                    .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
                            }
                            HStack(spacing: 12) {
                                Button {
                                    save()
                                } label: {
                                    Text(L("Enregistrer", "Save")).padding(.horizontal, 16).padding(.vertical, 10)
                                }.buttonStyle(PrimaryGlassButtonStyle()).disabled(name.isEmpty)
                                Button(L("Retirer", "Remove"), role: .destructive) { remove() }.disabled(
                                    selected.isEmpty)
                                Spacer()
                                Button(L("Tester les connexions", "Test connections")) { check() }.disabled(checking)
                                if checking { ProgressView().controlSize(.small) }
                            }
                            Text(
                                L(
                                    "Tester lance les serveurs locaux avec vos permissions et contacte les serveurs distants activés.",
                                    "Testing launches local servers with your permissions and contacts enabled remote servers."
                                )
                            )
                            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                        }
                        if let error, !error.isEmpty {
                            Text(error).foregroundStyle(.orange).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        SettingCard(title: L("Permissions du projet", "Project permissions"), icon: "lock.shield") {
                            Text(
                                studio.currentProject.map {
                                    L("Outils autorisés pour ", "Allowed tools for ") + $0.name
                                }
                                    ?? L(
                                        "Choisissez un projet pour limiter ses outils.",
                                        "Choose a project to restrict its tools.")
                            )
                            .foregroundStyle(StudioTheme.secondary)
                            ForEach(studio.mcpTools) { tool in
                                Toggle(
                                    isOn: Binding(
                                        get: { studio.currentProject?.allowedTools.contains(tool.id) == true },
                                        set: { allowed in
                                            guard var project = studio.currentProject else { return }
                                            project.allowedTools.removeAll { $0 == tool.id }
                                            if allowed { project.allowedTools.append(tool.id) }
                                            studio.updateProject(project)
                                        })
                                ) {
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(tool.id)
                                        Text(tool.function.description ?? "").font(.system(size: 12))
                                            .foregroundStyle(StudioTheme.secondary).fixedSize(
                                                horizontal: false, vertical: true)
                                    }
                                }.toggleStyle(.switch).disabled(studio.currentProject == nil)
                            }
                            if studio.mcpTools.isEmpty {
                                Text(
                                    L(
                                        "Testez une connexion pour afficher ses outils.",
                                        "Test a connection to list its tools.")
                                )
                                .foregroundStyle(StudioTheme.secondary)
                            }
                            Text(
                                L(
                                    "Un nouveau projet n’autorise aucun outil. Le bouton MCP du chat reste nécessaire pour les activer.",
                                    "A new project allows no tools. The chat MCP switch must also be enabled.")
                            )
                            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                        }
                    }
                    .padding(.vertical, 36).frame(maxWidth: 760, alignment: .leading)
                    .padding(.horizontal, 28).frame(maxWidth: .infinity)
                    .disabled(studio.isGenerating)
                }
        }
        .font(.system(size: 13)).controlSize(.large)
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
