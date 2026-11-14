import SwiftUI
import SpriteKit

struct StudioAurora: View {
  @Environment(\.colorScheme) private var scheme
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @Environment(\.colorSchemeContrast) private var contrast
  @Environment(\.scenePhase) private var scenePhase
  var busy = false
  var animated = false
  var fullHeight = false

  var body: some View {
    GeometryReader { geometry in
      let size = geometry.size
      if animated && !reduceMotion && scenePhase == .active && size.width > 0 && size.height > 0 {
        AuroraSurface(size: size, fullHeight: fullHeight, busy: busy)
      } else {
        Self.prints(size: size, fullHeight: fullHeight)
      }
    }
    .opacity(contrast == .increased ? 0.18 : (scheme == .dark ? (fullHeight ? 0.58 : 0.76) : 0.33))
    .clipped()
    .allowsHitTesting(false)
    .accessibilityHidden(true)
  }

  fileprivate static func prints(size: CGSize, fullHeight: Bool, image: CGImage = artwork.still) -> some View {
    let height = fullHeight ? size.height * 0.79 : min(size.height * 0.48, size.width * 0.42)
    return ZStack(alignment: .topLeading) {
      ForEach(curtains.indices, id: \.self) { index in
        let curtain = curtains[index]
        let width = min(size.width * (fullHeight ? 0.66 : 0.68), height * 2.1) * curtain.scale
        let depth = height * curtain.scale
        Image(decorative: image, scale: 2)
          .resizable().frame(width: width, height: depth)
          .hueRotation(.degrees(curtain.hue))
          .opacity(curtain.strength)
          .position(
            x: size.width * (curtain.x + 0.026 * sin(Double(index))),
            y: depth * ((fullHeight ? curtain.y : 0.31) + 0.035 * sin(Double(index))))
      }
    }
    .frame(width: size.width, height: size.height, alignment: .topLeading)
  }

  private static let curtains:
    [(x: Double, scale: Double, hue: Double, strength: Double, y: Double)] = [
      (0.045, 0.91, -38, 0.61, 0.64), (0.77, 1.08, 0, 1, 0.64),
      (1.05, 0.93, 9, 0.61, 0.64), (0.34, 0.67, -26, 0.52, -0.03),
    ]

  // The still print and the animated alphabet use the same Menlo typeface.
  fileprivate static let artwork: (still: CGImage, light: CGImage, alphabet: CGImage) = {
    let width = 740
    let height = 620
    let colors = CGColorSpace(name: CGColorSpace.sRGB)!
    func bitmap(_ w: Int, _ h: Int) -> CGContext {
      CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: colors, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    }

    // The curved foot is white-hot at the centre. The violet column fans out above it.
    func light(_ x: Double, _ y: Double) -> (SIMD3<Double>, Double) {
      let foot = 0.70 + 0.20 * x * x + 0.027 * sin(4.5 * x)
      let d = foot - y
      let span = 0.25 + 0.16 * max(0, d)
      let column = exp(-pow(x / span, 2) - pow((d - 0.22) / 0.43, 2))
      let rim = exp(-pow(x / 0.64, 2) - pow(d / 0.064, 2))
      let haze = exp(-pow(x / 0.66, 2) - pow((d - 0.17) / 0.46, 2))
      let hot = exp(-pow(x / 0.32, 2) - pow((d - 0.055) / 0.10, 2))
      let pink = exp(-pow((d - 0.14) / 0.19, 2))
      let rgb = SIMD3(
        0.20 + 0.78 * pink + 0.64 * rim + 0.30 * hot,
        0.022 + 0.10 * max(0, d) + 0.61 * hot,
        0.98 - 0.55 * rim - 0.61 * hot)
      let fade =
        min(1, max(0, y * 8)) * min(1, max(0, (1 - abs(x)) * 8))
        * min(1, max(0, (1 - y) * 9))
      return (
        rgb.clamped(lowerBound: .zero, upperBound: .one),
        min(1, 0.91 * column + 0.94 * rim + 0.24 * haze) * fade
          * exp(-pow(max(0, -d) / 0.09, 2))
      )
    }

    let glow = bitmap(width / 2, height / 2)
    let pixels = glow.data!.assumingMemoryBound(to: UInt8.self)
    for row in 0..<glow.height {
      for col in 0..<glow.width {
        let (rgb, a) = light(
          2 * (Double(col) + 0.5) / Double(glow.width) - 1,
          (Double(row) + 0.5) / Double(glow.height))
        let i = row * glow.bytesPerRow + 4 * col
        for k in 0..<3 { pixels[i + k] = UInt8(rgb[k] * a * 255) }
        pixels[i + 3] = UInt8(a * 255)
      }
    }

    let plate = bitmap(width * 2, height * 2)
    plate.scaleBy(x: 2, y: 2)
    plate.interpolationQuality = .high
    let lightImage = glow.makeImage()!
    plate.draw(lightImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    let font = CTFontCreateWithName("Menlo" as CFString, 7, nil)
    let alphabet = Array(".:-=+xX".utf16)
    var glyphs = [CGGlyph](repeating: 0, count: alphabet.count)
    CTFontGetGlyphsForCharacters(font, alphabet, &glyphs, glyphs.count)
    // Four texels per point leave clear margins around each 5 × 8 pt cell.
    let atlas = bitmap(20 * glyphs.count, 32)
    atlas.scaleBy(x: 4, y: 4)
    atlas.setFillColor(gray: 1, alpha: 1)
    for (index, value) in glyphs.enumerated() {
      var glyph = value
      var origin = CGPoint(x: Double(index * 5) + 0.4, y: 1.6)
      CTFontDrawGlyphs(font, &glyph, &origin, 1, atlas)
    }
    for row in 0..<(height / 8) {
      for col in 0..<(width / 5) {
        let x = Double(col * 5 + 2)
        let y = Double(row * 8 + 5)
        let (rgb, a) = light(2 * x / Double(width) - 1, y / Double(height))
        guard a > 0.13 else { continue }
        let grain = sin(Double(col * 17 + row * 31)) * 0.06
        let density = min(1, max(0, a + 0.13 * sin(y * 0.09) + grain))
        var glyph = glyphs[Int(density * Double(glyphs.count - 1))]
        var position = CGPoint(x: x, y: Double(height) - y - 2)
        plate.setFillColor(
          red: rgb.x * 0.60 + 0.40, green: rgb.y * 0.58 + 0.42,
          blue: rgb.z * 0.62 + 0.38, alpha: a * 0.47)
        CTFontDrawGlyphs(font, &glyph, &position, 1, plate)
      }
    }
    return (plate.makeImage()!, lightImage, atlas.makeImage()!)
  }()
}

/// The light travels across fixed type cells. Its intensity selects the glyph in each cell.
private struct AuroraSurface: NSViewRepresentable {
  var size: CGSize
  var fullHeight: Bool
  var busy: Bool

  func makeNSView(context: Context) -> SKView {
    let view = SKView()
    view.allowsTransparency = true
    view.preferredFramesPerSecond = 60
    let scene = AuroraScene(size: size)
    scene.backgroundColor = .clear
    scene.scaleMode = .resizeFill
    view.presentScene(scene)
    return view
  }

  func updateNSView(_ view: SKView, context: Context) {
    guard let scene = view.scene as? AuroraScene else { return }
    let scale = min(2, view.window?.backingScaleFactor ?? 2, 3072 / max(size.width, size.height))
    if scene.printSize != size || scene.fullHeight != fullHeight || scene.printScale != scale {
      let renderer = ImageRenderer(content: StudioAurora.prints(size: size, fullHeight: fullHeight, image: StudioAurora.artwork.light))
      renderer.scale = min(scale, 1)
      if let image = renderer.cgImage {
        scene.cells.vectorFloat2Value = SIMD2(Float(size.width / 5), Float(size.height / 8))
        scene.printSize = size
        scene.printScale = scale
        scene.fullHeight = fullHeight
        scene.size = size
        scene.sheet.texture = SKTexture(cgImage: image)
        scene.sheet.size = size
        scene.sheet.position = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
      }
    }
    scene.playback.setGenerating(busy, at: scene.elapsed)
  }

  static func dismantleNSView(_ view: SKView, coordinator: ()) {
    view.isPaused = true
    view.presentScene(nil)
  }
}

private final class AuroraScene: SKScene {
  var printSize = CGSize.zero
  var printScale = CGFloat.zero
  var fullHeight = false
  var elapsed = 0.0
  var lastFrame: TimeInterval?
  var playback = AuroraPlayback(at: 0)
  let sheet = SKSpriteNode()
  let cells = SKUniform(name: "cells", vectorFloat2: .zero)
  private let phase = SKUniform(name: "phase", float: 0)

  override func sceneDidLoad() {
    super.sceneDidLoad()
    sheet.shader = SKShader(source: """
      vec2 lightUV(vec2 p, float t) {
        vec2 drift = vec2(0.014*(sin(4.0*p.y+0.44*t)-sin(4.0*p.y)),
                          0.028*(sin(5.0*p.x-0.49*t)-sin(5.0*p.x)));
        vec2 at = clamp(p+drift*smoothstep(vec2(0.0),vec2(0.12),min(p,1.0-p)),0.0,1.0);
        return vec2(at.x,1.0-at.y);
      }
      void main() {
        vec2 p = vec2(v_tex_coord.x,1.0-v_tex_coord.y);
        float t = phase;
        vec2 cell = floor(p*cells);
        vec2 center = (cell+0.5)/cells;
        vec2 local = fract(p*cells);
        float arc = p.y + 0.065*sin(5.0*p.x + 0.24*t) + 0.035*sin(10.0*p.x - 0.31*t);
        float ribbon = 0.5 + 0.5*cos(10.5*arc - 0.82*t + 1.3*p.x);
        float glint = 0.5 + 0.5*sin(18.0*arc + 3.3*p.x - 0.58*t);
        float veil = 0.5 + 0.5*cos(4.7*p.x - 2.6*p.y + 0.37*t);
        float light = 0.77 + 0.26*ribbon + 0.12*glint*glint + 0.05*veil;
        vec3 emission = vec3(light + 0.08*ribbon*glint, light, light + 0.085*(1.0-ribbon)*veil);
        vec4 lamp = texture2D(u_texture,lightUV(p,t));
        float intensity = texture2D(u_texture,lightUV(center,t)).a;
        float wave = 0.5+0.5*cos(10.5*(center.y+0.065*sin(5.0*center.x+0.24*t))-0.82*t+1.3*center.x);
        float level = 6.0*clamp(intensity*(0.65+0.60*wave),0.0,1.0);
        float index = floor(level);
        vec2 tile = vec2(local.x/7.0,1.0-local.y);
        float first = texture2D(alphabet,tile+vec2(index/7.0,0.0)).a;
        float next = texture2D(alphabet,tile+vec2(min(6.0,index+1.0)/7.0,0.0)).a;
        float mark = mix(first,next,smoothstep(0.32,0.68,fract(level)));
        float alpha = lamp.a;
        vec3 color = min(lamp.rgb*emission,vec3(alpha));
        gl_FragColor = vec4(mix(color,vec3(alpha),0.45*mark*smoothstep(0.25,1.65,level)),alpha);
      }
      """, uniforms: [phase, cells, SKUniform(name: "alphabet", texture: SKTexture(cgImage: StudioAurora.artwork.alphabet))])
    addChild(sheet)
  }

  override func update(_ currentTime: TimeInterval) {
    if let lastFrame { elapsed += min(0.05, max(0, currentTime - lastFrame)) }
    lastFrame = currentTime
    phase.floatValue = Float(playback.phase(at: elapsed))
  }
}

/// Integrate the playback rate so changing speed never resets the field's position.
struct AuroraPlayback {
  private var start: Double
  private var origin = 0.0
  private var from = 1.0
  private var target = 1.0
  private var duration = 2.0

  init(at time: Double = ProcessInfo.processInfo.systemUptime) { start = time }

  mutating func setGenerating(_ busy: Bool, at time: Double) {
    let next = busy ? 2.0 : 1
    guard next != target else { return }
    let progress = min(1, max(0, (time - start) / duration))
    origin = phase(at: time)
    from += (target - from) * curve(progress)
    start = time
    target = next
    duration = busy ? 2 : 2.6
  }

  func phase(at time: Double) -> Double {
    let elapsed = max(0, time - start)
    let p = min(1, elapsed / duration)
    return origin + duration * (from * p + (target - from) * integral(p))
      + max(0, elapsed - duration) * target
  }

  // The back curve dips by about 10%. Skip that dip when a rapid restart could reverse the field.
  private var back: Double {
    from < target && from - 0.11 * (target - from) >= 0.12 ? 1.70158 * 1.525 : 0
  }

  // Back-in/out for acceleration, quartic ease-out for coasting.
  private func curve(_ p: Double) -> Double {
    if target == 1 { return 1 - pow(1 - p, 4) }
    let c = back
    let u = p < 0.5 ? 2 * p : 2 * (1 - p)
    let value = u * u * ((c + 1) * u - c) / 2
    return p < 0.5 ? value : 1 - value
  }

  private func integral(_ p: Double) -> Double {
    if target == 1 { return p + (pow(1 - p, 5) - 1) / 5 }
    let c = back
    let u = min(p, 1 - p)
    let leading = (c + 1) * pow(u, 4) - (2 * c / 3) * pow(u, 3)
    return p <= 0.5 ? leading : p - 0.5 + leading
  }
}
