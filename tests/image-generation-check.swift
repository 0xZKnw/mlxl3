import Foundation

@main
struct ImageGenerationCheck {
    @MainActor static func main() async throws {
        AppLocalization.set(.en)
        let engine = ImageGeneration()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for _ in 0..<2 {
            precondition(!engine.running)
            engine.start(model: root, prompt: "teapot", size: 512, steps: 1, seed: 371)
            precondition(engine.running && engine.error == nil && engine.result == nil)
            // A second click must not replace the in-flight request.
            engine.start(model: root, prompt: "duplicate", size: 512, steps: 1, seed: 372)
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while engine.running && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            precondition(!engine.running && engine.result == nil && engine.status.isEmpty)
            precondition(engine.error == "Choose the EXL3 folder containing manifest.json.")
        }
        print("Image generation: overlapping request, missing model and retry checked")
    }
}
