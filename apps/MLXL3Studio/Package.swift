// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MLXL3Studio",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "MLXL3Studio", targets: ["MLXL3Studio"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mgriebling/SwiftMath.git", exact: "1.7.0"),
    ],
    targets: [
        .executableTarget(
            name: "MLXL3Studio",
            dependencies: ["SwiftMath"]
        ),
    ]
)
