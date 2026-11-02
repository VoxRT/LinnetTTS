//! Build script: detect whether the compiler has the stable AArch64
//! binary16 conversion intrinsics (`vcvt_f32_f16` / `vcvt_high_f32_f16`,
//! stable since Rust 1.94). The NEON fp16 kernels use them when
//! available and fall back to equivalent inline assembly on older
//! toolchains, so the crate keeps its minimum supported Rust version.

use std::process::Command;

fn rustc_minor() -> Option<u32> {
    let rustc = std::env::var_os("RUSTC")?;
    let out = Command::new(rustc).arg("--version").output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    // "rustc 1.94.0 (hash date)" or "rustc 1.96.0-nightly (...)"
    let version = text.split_whitespace().nth(1)?;
    let mut parts = version.split(['.', '-']);
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    if major != 1 {
        return None;
    }
    Some(minor)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-check-cfg=cfg(linnet_f16_intrinsics)");
    if matches!(rustc_minor(), Some(minor) if minor >= 94) {
        println!("cargo:rustc-cfg=linnet_f16_intrinsics");
    }
}
