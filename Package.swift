// swift-tools-version:5.9
// LinnetTTS Swift package (iOS 15+; macOS 12+ for host builds and tests only, not a supported product): the Swift API
// in bindings/ios/LinnetTTS/Sources over the C API of the
// Rust runtime, which ships as a prebuilt static LinnetTTS.xcframework attached to each GitHub release.
//
//   .package(url: "https://github.com/VoxRT/LinnetTTS.git", from: "0.1.0")
//
// The binary target points at the release asset; the checksum below must match that asset (see build/README.md,
// "Release checklist"). To build against a locally produced xcframework (bindings/ios/build-xcframework.sh), set
// LINNET_LOCAL_XCFRAMEWORK=1 in the environment of the SwiftPM / xcodebuild invocation. The package in
// bindings/ios/LinnetTTS is the local development package with the same sources and the XCTests.
import Foundation
import PackageDescription

let version = "0.1.1"
let checksum = "83990ef6ab979857cc66dd57c903b80c180f533a55007e6cec5907d898caed64"

let binary: Target = ProcessInfo.processInfo.environment["LINNET_LOCAL_XCFRAMEWORK"] != nil
    ? .binaryTarget(name: "CLinnetTTS", path: "bindings/ios/LinnetTTS/LinnetTTS.xcframework")
    : .binaryTarget(
        name: "CLinnetTTS",
        url: "https://github.com/VoxRT/LinnetTTS/releases/download/v\(version)/LinnetTTS-\(version).xcframework.zip",
        checksum: checksum
    )

let package = Package(
    name: "LinnetTTS",
    platforms: [.iOS(.v15), .macOS(.v12)],   // macOS: host builds and tests only
    products: [
        .library(name: "LinnetTTS", targets: ["LinnetTTS"]),
    ],
    targets: [
        binary,
        .target(
            name: "LinnetTTS",
            dependencies: ["CLinnetTTS"],
            path: "bindings/ios/LinnetTTS/Sources/LinnetTTS",
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
    ]
)
