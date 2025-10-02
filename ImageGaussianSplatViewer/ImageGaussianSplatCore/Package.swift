// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ImageGaussianSplatCore",
    platforms: [
        .macOS(.v13),
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "ImageGaussianSplatCore",
            targets: ["ImageGaussianSplatCore"]
        )
    ],
    targets: [
        .target(
            name: "ImageGaussianSplatCore",
            resources: [
                .process("Shaders")
            ]
        )
    ]
)
