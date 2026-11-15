import AppKit
import SpriteKit
import SwiftUI

/// Renders the real AuroraScene off screen with a reference shader (argument 1, a file)
/// and with the shipped `AuroraScene.shaderSource`, then requires byte-identical pixels.
@main struct AuroraShaderCheck {
    @MainActor static func main() throws {
        struct Failure: Error, CustomStringConvertible { let description: String }
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else { throw Failure(description: "usage: aurora-shader-check <reference.fsh> [candidate.fsh]") }
        let reference = try String(contentsOfFile: arguments[1], encoding: .utf8)
        let candidate = arguments.count >= 3 ? try String(contentsOfFile: arguments[2], encoding: .utf8) : AuroraScene.shaderSource
        guard reference.contains("gl_FragColor"), candidate.contains("gl_FragColor") else {
            throw Failure(description: "Reference and candidate must both be fragment shaders")
        }

        let view = SKView(frame: CGRect(x: 0, y: 0, width: 960, height: 640))
        func pixels(_ scene: AuroraScene, width: Int, height: Int) throws -> [UInt8] {
            guard let texture = view.texture(from: scene) else { throw Failure(description: "SpriteKit returned no texture") }
            let image = texture.cgImage()
            guard image.width > 0, image.height > 0 else { throw Failure(description: "Empty render") }
            var data = [UInt8](repeating: 0, count: width * height * 4)
            let drawn = data.withUnsafeMutableBytes { buffer -> Bool in
                guard let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
                else { return false }
                context.interpolationQuality = .none
                context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
                return true
            }
            guard drawn else { throw Failure(description: "Bitmap context unavailable") }
            return data
        }

        var compared = 0, transparent = 0, lit = 0, differing = 0, maxDelta = 0
        let layouts: [(CGSize, Bool)] = [(CGSize(width: 960, height: 640), true), (CGSize(width: 720, height: 420), false)]
        let phases: [Float] = [0, 0.37, 5.5, 41.3, 1234.5]
        for (size, fullHeight) in layouts {
            let renderer = ImageRenderer(content: StudioAurora.prints(size: size, fullHeight: fullHeight, image: StudioAurora.artwork.light))
            renderer.scale = 1
            guard let light = renderer.cgImage else { throw Failure(description: "Light print did not render") }
            let scene = AuroraScene(size: size)
            scene.backgroundColor = .clear
            scene.scaleMode = .resizeFill
            scene.cells.vectorFloat2Value = SIMD2(Float(size.width / 5), Float(size.height / 8))
            scene.sheet.texture = SKTexture(cgImage: light)
            scene.sheet.size = size
            scene.sheet.position = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
            let width = Int(size.width), height = Int(size.height)
            // A shader that fails to compile falls back to the plain texture; if both
            // did, they would match each other, so the reference must visibly differ.
            scene.sheet.shader = nil
            let plain = try pixels(scene, width: width, height: height)
            scene.phase.floatValue = phases[0]
            scene.sheet.shader = scene.makeShader(source: reference)
            guard try pixels(scene, width: width, height: height) != plain else {
                throw Failure(description: "Reference shader had no effect (did it compile?)")
            }
            for phase in phases {
                scene.phase.floatValue = phase
                scene.sheet.shader = scene.makeShader(source: reference)
                let before = try pixels(scene, width: width, height: height)
                scene.sheet.shader = scene.makeShader(source: candidate)
                let after = try pixels(scene, width: width, height: height)
                guard before.count == after.count, before.count == width * height * 4 else {
                    throw Failure(description: "Pixel count mismatch")
                }
                for index in stride(from: 0, to: before.count, by: 4) {
                    compared += 1
                    if before[index + 3] == 0 { transparent += 1 } else if before[index + 3] > 32 { lit += 1 }
                    var same = true
                    for channel in 0..<4 {
                        let delta = abs(Int(before[index + channel]) - Int(after[index + channel]))
                        maxDelta = max(maxDelta, delta)
                        if delta != 0 { same = false }
                    }
                    if !same { differing += 1 }
                }
            }
        }
        let summary = "compared=\(compared) transparent=\(transparent) lit=\(lit) differing=\(differing) maxDelta=\(maxDelta)"
        guard compared == layouts.reduce(0, { $0 + Int($1.0.width * $1.0.height) }) * phases.count else {
            throw Failure(description: "Not every pixel was compared: " + summary)
        }
        guard lit > compared / 20, transparent > 0 else { throw Failure(description: "Render is trivial: " + summary) }
        guard differing == 0 else { throw Failure(description: "Shader output changed: " + summary) }
        print("Aurora shader parity passed: " + summary)
    }
}
