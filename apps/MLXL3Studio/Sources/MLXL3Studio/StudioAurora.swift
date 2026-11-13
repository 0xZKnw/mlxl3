import SwiftUI

/// A single band above the workspace. The reading surface below it is the user's theme.
struct StudioAurora: View {
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
      let cool =
        dark ? Color(red: 0.29, green: 0.42, blue: 0.74) : Color(red: 0.61, green: 0.77, blue: 0.99)
      let violet =
        dark ? Color(red: 0.47, green: 0.36, blue: 0.64) : Color(red: 0.80, green: 0.73, blue: 0.96)
      let rose =
        dark ? Color(red: 0.69, green: 0.38, blue: 0.52) : Color(red: 0.98, green: 0.77, blue: 0.84)
      let amber =
        dark ? Color(red: 0.77, green: 0.48, blue: 0.32) : Color(red: 1, green: 0.82, blue: 0.65)

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

  // Travel, breathing periods, directions and stagger from Locked's five aurora lobes.
  private struct Light {
    let x, dx, dy, travelTime, breathTime, opacity, endOpacity, descent, settle, delay: Double
  }

  private static let lights = [
    Light(
      x: 0.12, dx: 0.19, dy: 0.11, travelTime: 21, breathTime: 13, opacity: 0.95, endOpacity: 0.50,
      descent: 0.14, settle: 1.7, delay: 0.05),
    Light(
      x: 0.34, dx: -0.13, dy: 0.17, travelTime: 27, breathTime: 17, opacity: 0.55, endOpacity: 1,
      descent: 0.08, settle: 2.2, delay: 0.26),
    Light(
      x: 0.50, dx: 0.14, dy: -0.08, travelTime: 23, breathTime: 11, opacity: 0.90, endOpacity: 0.42,
      descent: 0.17, settle: 1.5, delay: 0.13),
    Light(
      x: 0.70, dx: -0.17, dy: 0.14, travelTime: 19, breathTime: 15, opacity: 0.60, endOpacity: 1,
      descent: 0.10, settle: 2.4, delay: 0.38),
    Light(
      x: 0.91, dx: -0.21, dy: -0.10, travelTime: 25, breathTime: 19, opacity: 1, endOpacity: 0.58,
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
    let next = busy ? 4.4 : 1
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
