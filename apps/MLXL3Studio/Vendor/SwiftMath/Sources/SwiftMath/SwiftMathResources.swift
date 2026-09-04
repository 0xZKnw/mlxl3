import Foundation

enum SwiftMathResources {
    static let fontBundle: Bundle = {
        let packaged = Bundle.main.resourceURL?
            .appending(path: "mathFonts.bundle", directoryHint: .isDirectory)
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "mathFonts.bundle", directoryHint: .isDirectory)

        for candidate in [packaged, source].compactMap({ $0 }) {
            if let bundle = Bundle(url: candidate) {
                return bundle
            }
        }
        fatalError("SwiftMath mathFonts.bundle is missing")
    }()
}
