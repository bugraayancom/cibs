use std::env;
use std::path::PathBuf;

fn main() {
    // Embed an rpath to the vendored prebuilt MLX dylib (vendor/mlx/lib) so
    // binaries run without DYLD_LIBRARY_PATH. MLX also loads mlx.metallib
    // from the directory containing libmlx.dylib, which the wheel provides.
    if cfg!(feature = "mlx") {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let lib_dir = manifest_dir.join("vendor/mlx/lib");
        if lib_dir.is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
        println!("cargo:rerun-if-changed=vendor/mlx/lib");
    }
}
