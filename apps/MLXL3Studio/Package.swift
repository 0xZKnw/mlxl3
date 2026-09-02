// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MLXL3Studio",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "MLXL3Studio", targets: ["MLXL3Studio"]),
    ],
    targets: [
        .executableTarget(name: "MLXL3Studio"),
    ]
)
