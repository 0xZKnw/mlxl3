// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MLXL3Studio",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "MLXL3Studio", targets: ["MLXL3Studio"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "SwiftMath",
            path: "Vendor/SwiftMath/Sources/SwiftMath",
            exclude: ["mathFonts.bundle"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "MLXL3Studio",
            dependencies: ["SwiftMath"]
        ),
    ]
)
