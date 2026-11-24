import SwiftUI
import SpriteKit

/// The background of the window, chosen by the style: the ASCII aurora, the soft aurora,
/// or (Graphite) a monochrome type texture on the home screen only.
struct StudioAurora: View {
  /// `ambient` sits behind whole screens (Aurora styles); `home` is the empty home
  /// screen's texture (Graphite).
  enum Placement { case ambient, home }

  @Environment(\.colorScheme) private var scheme
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @Environment(\.colorSchemeContrast) private var contrast
  @Environment(\.scenePhase) private var scenePhase
  @Environment(\.studioStyle) private var style
  var busy = false
  var animated = false
  var fullHeight = false
  var placement = Placement.ambient

  // Always a GeometryReader: callers hang the canvas colour on this view with
  // `.background`, so a style without this layer must still fill the space offered.
  var body: some View {
    switch (placement, style) {
    case (.ambient, .auroraASCII): ascii
    case (.ambient, .aurora): SoftAurora(busy: busy, animated: animated)
    case (.home, .graphite): texture
    default: GeometryReader { _ in Color.clear }.allowsHitTesting(false).accessibilityHidden(true)
    }
  }

  private var ascii: some View {
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

  /// The same cells and glyphs as the aurora, without the light: type a few percent
  /// above the paper, rising from the bottom edge like a horizon. The top half stays
  /// clear for the greeting, whatever the caller does.
  private var texture: some View {
    GeometryReader { geometry in
      let size = geometry.size
      if animated && !reduceMotion && scenePhase == .active && size.width > 0 && size.height > 0 {
        AuroraSurface(size: size, fullHeight: true, busy: busy, tint: inkComponents)
      } else {
        // The still print (Reduce Motion, inactive window) fills its glyph shapes with the
        // same ink as the shader. `colorMultiply` is a filter that some render paths skip
        // (the white glyphs then vanished on light paper); a template image is not.
        Self.homePrint(size: size, image: Self.artwork.type, ink: Color(nsColor: ink))
      }
    }
    .opacity(contrast == .increased ? 0.10 : (scheme == .dark ? 0.30 : 0.24))
    .mask(Self.homeMask)
    .clipped()
    .allowsHitTesting(false)
    .accessibilityHidden(true)
  }

  /// Clear over the top half, full at the bottom edge.
  static let homeMask = LinearGradient(
    stops: [.init(color: .clear, location: 0.5), .init(color: .black, location: 1)],
    startPoint: .top, endPoint: .bottom)

  /// Graphite ink for the current scheme, as the texture draws it.
  private var ink: NSColor { scheme == .dark ? StudioTheme.rgb(0xEAE6DF) : StudioTheme.rgb(0x1C1B19) }

  private var inkComponents: SIMD3<Float> {
    SIMD3(Float(ink.redComponent), Float(ink.greenComponent), Float(ink.blueComponent))
  }

  /// One curtain, wide, its hot foot on the bottom edge. With `ink`, the image's alpha
  /// is drawn in that colour (the type print); without, the image keeps its own colours
  /// (the light the shader samples).
  static func homePrint(size: CGSize, image: CGImage, ink: Color? = nil) -> some View {
    let width = size.width * 1.35
    let depth = width * 620 / 740
    return Image(decorative: image, scale: 2)
      .renderingMode(ink == nil ? .original : .template)
      .resizable().frame(width: width, height: depth)
      .foregroundStyle(ink ?? .white)
      .position(x: size.width * 0.5, y: size.height + depth * (0.5 - 0.74))
      .frame(width: size.width, height: size.height, alignment: .topLeading)
  }

  static func prints(size: CGSize, fullHeight: Bool, image: CGImage = artwork.still) -> some View {
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
  static let artwork: (still: CGImage, light: CGImage, alphabet: CGImage, type: CGImage) = {
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
    // The texture print: the same glyphs in white, without the light under them.
    let type = bitmap(width * 2, height * 2)
    type.scaleBy(x: 2, y: 2)
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
        // Fade in from the threshold instead of starting at a visible edge.
        let t = min(1, max(0, (a - 0.13) / 0.62))
        type.setFillColor(gray: 1, alpha: t * t * (3 - 2 * t))
        CTFontDrawGlyphs(font, &glyph, &position, 1, type)
      }
    }
    return (plate.makeImage()!, lightImage, atlas.makeImage()!, type.makeImage()!)
  }()
}

/// Stops drawing while the window cannot be seen (minimized, hidden, fully covered).
/// The scene clamps each frame step, so the field resumes where it stopped.
final class AuroraView: SKView {
  private var occlusion: NSObjectProtocol?

  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    if let occlusion { NotificationCenter.default.removeObserver(occlusion) }
    occlusion = nil
    guard let window else { return }
    occlusion = NotificationCenter.default.addObserver(
      forName: NSWindow.didChangeOcclusionStateNotification, object: window, queue: .main
    ) { [weak self] _ in
      MainActor.assumeIsolated { self?.syncVisibility() }
    }
    syncVisibility()
  }

  func syncVisibility() {
    let visible = window?.occlusionState.contains(.visible) ?? false
    if isPaused == visible { isPaused = !visible }
  }

  isolated deinit { if let occlusion { NotificationCenter.default.removeObserver(occlusion) } }
}

/// The light travels across fixed type cells. Its intensity selects the glyph in each cell.
private struct AuroraSurface: NSViewRepresentable {
  var size: CGSize
  var fullHeight: Bool
  var busy: Bool
  /// Ink colour for the Graphite texture; nil keeps the coloured light.
  var tint: SIMD3<Float>? = nil

  func makeNSView(context: Context) -> SKView {
    let view = AuroraView()
    view.allowsTransparency = true
    // The texture moves slowly: 30 fps is enough and leaves the GPU to inference.
    view.preferredFramesPerSecond = tint == nil ? 60 : 30
    let scene = AuroraScene(size: size, monochrome: tint != nil)
    scene.backgroundColor = .clear
    scene.scaleMode = .resizeFill
    view.presentScene(scene)
    return view
  }

  func updateNSView(_ view: SKView, context: Context) {
    guard let scene = view.scene as? AuroraScene else { return }
    let scale = min(2, view.window?.backingScaleFactor ?? 2, 3072 / max(size.width, size.height))
    if scene.printSize != size || scene.fullHeight != fullHeight || scene.printScale != scale {
      let renderer = ImageRenderer(content: Group {
        if tint == nil {
          StudioAurora.prints(size: size, fullHeight: fullHeight, image: StudioAurora.artwork.light)
        } else {
          StudioAurora.homePrint(size: size, image: StudioAurora.artwork.light)
        }
      })
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
    if let tint { scene.tint.vectorFloat3Value = tint }
    scene.playback.setGenerating(busy, at: scene.elapsed)
  }

  static func dismantleNSView(_ view: SKView, coordinator: ()) {
    view.isPaused = true
    view.presentScene(nil)
  }
}

final class AuroraScene: SKScene {
  var printSize = CGSize.zero
  var printScale = CGFloat.zero
  var fullHeight = false
  var elapsed = 0.0
  var lastFrame: TimeInterval?
  var playback = AuroraPlayback(at: 0)
  let sheet = SKSpriteNode()
  let cells = SKUniform(name: "cells", vectorFloat2: .zero)
  let phase = SKUniform(name: "phase", float: 0)
  let tint = SKUniform(name: "tint", vectorFloat3: SIMD3(1, 1, 1))

  /// The monochrome scene draws only the type, in `tint`, from the same light and cells.
  convenience init(size: CGSize, monochrome: Bool) {
    self.init(size: size)
    if monochrome { sheet.shader = makeShader(source: Self.monochromeShaderSource) }
  }

  override func sceneDidLoad() {
    super.sceneDidLoad()
    sheet.shader = makeShader(source: Self.shaderSource)
    addChild(sheet)
  }

  func makeShader(source: String) -> SKShader {
    SKShader(source: source, uniforms: [phase, cells, tint, SKUniform(name: "alphabet", texture: SKTexture(cgImage: StudioAurora.artwork.alphabet))])
  }

  // Same cells, glyph choice and drift as `shaderSource`; it keeps the marks and drops
  // the coloured light, so the field reads as printed type rather than a glow.
  static let monochromeShaderSource = """
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
        float intensity = texture2D(u_texture,lightUV(center,t)).a;
        if (intensity <= 0.13) {
          gl_FragColor = vec4(0.0);
        } else {
          float wave = 0.5+0.5*cos(10.5*(center.y+0.065*sin(5.0*center.x+0.24*t))-0.82*t+1.3*center.x);
          float level = 6.0*clamp(intensity*(0.65+0.60*wave),0.0,1.0);
          float index = floor(level);
          vec2 tile = vec2(local.x/7.0,1.0-local.y);
          float first = texture2D(alphabet,tile+vec2(index/7.0,0.0)).a;
          float next = texture2D(alphabet,tile+vec2(min(6.0,index+1.0)/7.0,0.0)).a;
          float mark = mix(first,next,smoothstep(0.32,0.68,fract(level)));
          float alpha = mark*smoothstep(0.13,0.75,intensity);
          gl_FragColor = vec4(tint*alpha,alpha);
        }
      }
      """

  // Transparent pixels (lamp alpha exactly 0) leave before the light math; they produce
  // (0,0,0,0) on both paths, and the rest of the program is unchanged.
  static let shaderSource = """
      vec2 lightUV(vec2 p, float t) {
        vec2 drift = vec2(0.014*(sin(4.0*p.y+0.44*t)-sin(4.0*p.y)),
                          0.028*(sin(5.0*p.x-0.49*t)-sin(5.0*p.x)));
        vec2 at = clamp(p+drift*smoothstep(vec2(0.0),vec2(0.12),min(p,1.0-p)),0.0,1.0);
        return vec2(at.x,1.0-at.y);
      }
      void main() {
        vec2 p = vec2(v_tex_coord.x,1.0-v_tex_coord.y);
        float t = phase;
        vec4 lamp = texture2D(u_texture,lightUV(p,t));
        if (lamp.a <= 0.0) {
          gl_FragColor = vec4(0.0);
        } else {
          vec2 cell = floor(p*cells);
          vec2 center = (cell+0.5)/cells;
          vec2 local = fract(p*cells);
          float arc = p.y + 0.065*sin(5.0*p.x + 0.24*t) + 0.035*sin(10.0*p.x - 0.31*t);
          float ribbon = 0.5 + 0.5*cos(10.5*arc - 0.82*t + 1.3*p.x);
          float glint = 0.5 + 0.5*sin(18.0*arc + 3.3*p.x - 0.58*t);
          float veil = 0.5 + 0.5*cos(4.7*p.x - 2.6*p.y + 0.37*t);
          float light = 0.77 + 0.26*ribbon + 0.12*glint*glint + 0.05*veil;
          vec3 emission = vec3(light + 0.08*ribbon*glint, light, light + 0.085*(1.0-ribbon)*veil);
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
      }
      """

  override func update(_ currentTime: TimeInterval) {
    if let lastFrame { elapsed += min(0.05, max(0, currentTime - lastFrame)) }
    lastFrame = currentTime
    phase.floatValue = Float(playback.phase(at: elapsed))
  }
}

/// The first aurora (7becaaba3): a mesh band and five soft lights above the workspace,
/// no characters. It descends and saturates while the model generates.
private struct SoftAurora: View {
  @Environment(\.colorScheme) private var scheme
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @Environment(\.colorSchemeContrast) private var contrast
  @Environment(\.scenePhase) private var scenePhase
  @State private var playback = AuroraPlayback()
  @State private var phase = 0.0
  var busy = false
  var animated = false

  var body: some View {
    let moving = animated && !reduceMotion && scenePhase == .active
    field(at: moving ? phase : 0)
      .task(id: moving ? (busy ? 12 : 6) : 0) {
        guard moving else { return }
        let interval: Duration = .milliseconds(busy ? 83 : 167)
        while !Task.isCancelled {
          phase = playback.phase(at: ProcessInfo.processInfo.systemUptime)
          try? await Task.sleep(for: interval, tolerance: .milliseconds(8))
        }
      }
      .onChange(of: busy, initial: true) { _, running in
        playback.setGenerating(running, at: ProcessInfo.processInfo.systemUptime)
      }
      .clipped()
      .allowsHitTesting(false)
      .accessibilityHidden(true)
  }

  private func field(at phase: Double) -> some View {
    GeometryReader { geometry in
      let dark = scheme == .dark
      let cool = dark ? Color(red: 0.29, green: 0.42, blue: 0.74) : Color(red: 0.61, green: 0.77, blue: 0.99)
      let violet = dark ? Color(red: 0.47, green: 0.36, blue: 0.64) : Color(red: 0.80, green: 0.73, blue: 0.96)
      let rose = dark ? Color(red: 0.69, green: 0.38, blue: 0.52) : Color(red: 0.98, green: 0.77, blue: 0.84)
      let amber = dark ? Color(red: 0.77, green: 0.48, blue: 0.32) : Color(red: 1, green: 0.82, blue: 0.65)

      let size = geometry.size
      let drift = Self.alternate(phase, duration: 44)
      ZStack(alignment: .topLeading) {
        MeshGradient(
          width: 4, height: 4,
          points: [
            [0, 0], [0.32, 0], [0.68, 0], [1, 0],
            [0, 0.18], [0.32, 0.42], [0.68, 0.15], [1, 0.26],
            [0, 0.52], [0.32, 0.73], [0.68, 0.40], [1, 0.59],
            [0, 1], [0.32, 1], [0.68, 1], [1, 1],
          ] as [SIMD2<Float>],
          colors: [
            cool, violet, rose, amber,
            cool, cool, violet, amber,
            cool.opacity(0.20), violet.opacity(0.28), rose.opacity(0.10), amber.opacity(0.17),
            .clear, .clear, .clear, .clear,
          ]
        )
        .scaleEffect(x: 1.12, y: 1)
        .offset(x: (drift - 0.5) * size.width * 0.10)

        ForEach(Self.lights.indices, id: \.self) { index in
          let light = Self.lights[index]
          let travel = Self.glide.value(at: Self.alternate(phase, duration: light.travelTime))
          let breath = Self.alternate(phase, duration: light.breathTime)
          let color = [cool, violet, rose, amber, amber][index]
          EllipticalGradient(
            colors: [color.opacity(0.80), color.opacity(0.32), .clear],
            center: .center, startRadiusFraction: 0, endRadiusFraction: 0.5
          )
          .frame(width: size.width * 0.49, height: size.height * 0.34)
          .opacity(light.opacity + (light.endOpacity - light.opacity) * breath)
          .scaleEffect(1 + 0.16 * travel)
          .position(
            x: size.width * (light.x + light.dx * travel),
            y: size.height * (0.035 + light.dy * travel)
          )
          .offset(y: busy ? size.height * light.descent : 0)
          .animation(
            reduceMotion
              ? nil
              : .timingCurve(0.5, -0.06, 0.25, 1.16, duration: busy ? light.settle : 2.6)
                .delay(busy ? light.delay : 0), value: busy)
        }
      }
      .frame(width: size.width, height: size.height * (busy ? 0.77 : 0.51))
      .mask(
        LinearGradient(
          stops: [
            .init(color: .black, location: 0.15),
            .init(color: .black.opacity(0.60), location: 0.48), .init(color: .clear, location: 1),
          ], startPoint: .top, endPoint: .bottom)
      )
      .hueRotation(.degrees(-7 + 13 * Self.alternate(phase, duration: 52)))
      .saturation(busy ? 1.65 : 1.03)
      .opacity(contrast == .increased ? 0.22 : (dark ? 0.72 : 0.89))
      .animation(reduceMotion ? nil : .easeInOut(duration: busy ? 2.3 : 2.6), value: busy)
    }
  }

  // Travel, breathing periods, directions and stagger of the five lobes.
  private struct Light {
    let x, dx, dy, travelTime, breathTime, opacity, endOpacity, descent, settle, delay: Double
  }

  private static let lights = [
    Light(x: 0.12, dx: 0.19, dy: 0.11, travelTime: 21, breathTime: 13, opacity: 0.95, endOpacity: 0.50,
          descent: 0.14, settle: 1.7, delay: 0.05),
    Light(x: 0.34, dx: -0.13, dy: 0.17, travelTime: 27, breathTime: 17, opacity: 0.55, endOpacity: 1,
          descent: 0.08, settle: 2.2, delay: 0.26),
    Light(x: 0.50, dx: 0.14, dy: -0.08, travelTime: 23, breathTime: 11, opacity: 0.90, endOpacity: 0.42,
          descent: 0.17, settle: 1.5, delay: 0.13),
    Light(x: 0.70, dx: -0.17, dy: 0.14, travelTime: 19, breathTime: 15, opacity: 0.60, endOpacity: 1,
          descent: 0.10, settle: 2.4, delay: 0.38),
    Light(x: 0.91, dx: -0.21, dy: -0.10, travelTime: 25, breathTime: 19, opacity: 1, endOpacity: 0.58,
          descent: 0.15, settle: 1.9, delay: 0.19),
  ]

  private static let glide = UnitCurve.bezier(
    startControlPoint: .init(x: 0.22, y: 1), endControlPoint: .init(x: 0.36, y: 1))

  private static func alternate(_ phase: Double, duration: Double) -> Double {
    (1 - cos(.pi * phase / duration)) / 2
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
