import Foundation

/// The Agent tasks of the sidebar that work in one folder.
struct AgentFolderGroup: Identifiable, Equatable {
    /// The folder's path; nil: tasks that have their own private copy in the app's data.
    let folder: String?
    let title: String
    let conversations: [UUID]
    var id: String { folder ?? "" }

    /// Groups `tasks` (in sidebar order). A group stands where its first task does, so the
    /// folder worked in last comes first; tasks without a folder always come last. Two
    /// folders with the same name also show their parent folder.
    static func groups(_ tasks: [(id: UUID, folder: String?)]) -> [AgentFolderGroup] {
        var order: [String] = []
        var members: [String: [UUID]] = [:]
        var loose: [UUID] = []
        for task in tasks {
            guard let folder = task.folder else { loose.append(task.id); continue }
            if members[folder] == nil { order.append(folder) }
            members[folder, default: []].append(task.id)
        }
        let names = Dictionary(grouping: order) { ($0 as NSString).lastPathComponent }
        var groups = order.map { folder -> AgentFolderGroup in
            let name = (folder as NSString).lastPathComponent
            let parent = ((folder as NSString).deletingLastPathComponent as NSString).lastPathComponent
            let shared = (names[name]?.count ?? 0) > 1 && !parent.isEmpty && parent != "/"
            return AgentFolderGroup(folder: folder, title: shared ? parent + "/" + name : name,
                                    conversations: members[folder] ?? [])
        }
        if !loose.isEmpty {
            groups.append(AgentFolderGroup(folder: nil, title: L("Sans dossier", "No folder"), conversations: loose))
        }
        return groups
    }
}
