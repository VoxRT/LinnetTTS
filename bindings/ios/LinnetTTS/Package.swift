// swift-tools-version:5.9
// LinnetTTS Swift package: a Swift API over the C API of the Rust runtime, shipped as
// LinnetTTS.xcframework (static library; ios-arm64, ios simulator arm64 + x86_64, macos-arm64 for host builds and tests only).
//
// This is the local development package: the binary target below points at the xcframework produced by
// ../build-xcframework.sh. The package consumed by URL lives in the repository root (Package.swift there) and
// references the release asset by URL and SwiftPM checksum.
import PackageDescription

let package = Package(
    name: "LinnetTTS",
    platforms: [.iOS(.v15), .macOS(.v12)],   // macOS: host builds and tests only, not a supported product
    products: [
        .library(name: "LinnetTTS", targets: ["LinnetTTS"]),
    ],
    targets: [
        .binaryTarget(name: "CLinnetTTS", path: "LinnetTTS.xcframework"),
        .target(
            name: "LinnetTTS",
            dependencies: ["CLinnetTTS"],
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
        .testTarget(
            name: "LinnetTTSTests",
            dependencies: ["LinnetTTS"],
            // Model files for runs where the host file system is out of reach (a physical
            // device): copy them into Tests/LinnetTTSTests/Assets/ before building the tests.
            resources: [.copy("Assets")]
        ),
    ]
)
