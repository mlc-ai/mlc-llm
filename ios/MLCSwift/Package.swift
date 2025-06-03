// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLCSwift",
    products: [
        .library(
            name: "MLCSwift",
            targets: ["MLCEngineObjC", "MLCSwift"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MLCEngineObjC",
            path: "Sources/ObjC",
            cxxSettings: [
                .headerSearchPath("../../tvm_home/include"),
                .headerSearchPath("../../tvm_home/ffi/include"),
                .headerSearchPath("../../tvm_home/3rdparty/dmlc-core/include"),
                .headerSearchPath("../../tvm_home/3rdparty/dlpack/include")
            ]
        ),
        .target(
            name: "MLCSwift",
            dependencies: ["MLCEngineObjC"],
            path: "Sources/Swift"
        )
    ],
    cxxLanguageStandard: .cxx17
)
