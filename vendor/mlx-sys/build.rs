extern crate cmake;

use bindgen::RustTarget;
use cmake::Config;
use std::{env, path::PathBuf};

/// Patched build script: instead of compiling MLX (and its Metal kernels,
/// which require the full Xcode `metal` toolchain) from source, link against
/// the prebuilt `libmlx.dylib` shipped in the official MLX Python wheel,
/// vendored at `vendor/mlx`. Only the thin `mlx-c` C API layer is compiled
/// here, which needs nothing beyond a C++ compiler.
fn build_and_link_mlx_c() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlx_root = manifest_dir
        .parent()
        .expect("vendor dir")
        .join("mlx")
        .canonicalize()
        .expect("vendor/mlx with the prebuilt MLX dylib must exist");

    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("CMAKE_BUILD_TYPE", "Release");
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");
    config.define("MLX_C_USE_SYSTEM_MLX", "ON");
    config.define("MLX_ROOT", mlx_root.to_str().unwrap());
    config.define("CMAKE_PREFIX_PATH", mlx_root.to_str().unwrap());

    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlxc");

    println!(
        "cargo:rustc-link-search=native={}",
        mlx_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlx");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Let dependents (the top-level crate) know where the dylib lives so they
    // can embed an rpath.
    println!("cargo:mlx_lib_dir={}", mlx_root.join("lib").display());
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target(RustTarget::Stable_1_73)
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
