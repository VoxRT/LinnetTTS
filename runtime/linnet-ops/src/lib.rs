//! Numeric kernels for the LinnetTTS runtime.
//!
//! Every public operation in [`ops`] comes in two flavours:
//!
//! - `op(...)` — the dispatcher. On aarch64 it calls the NEON kernel,
//!   on x86_64 it calls the AVX2 + FMA kernel when the CPU supports it
//!   (runtime check via `is_x86_feature_detected!`), on wasm32 with
//!   `simd128` it calls the SIMD128 kernel; otherwise it falls back to
//!   the scalar reference.
//! - `op_scalar(...)` — the pure scalar reference. Never dispatches.
//!
//! The `force-scalar` Cargo feature compiles the SIMD modules out and
//! makes every dispatcher call its scalar sibling.
//!
//! Numerical policy: fp16 / int8 weights are always expanded to f32 in
//! registers and accumulated in f32. Element-wise kernels keep a single
//! rounding per element (bit-exact across paths unless the kernel is
//! documented as FMA-fused). Reductions (matrix products, dot products,
//! LayerNorm statistics) may differ from the scalar reference by a few
//! ULP because the summation order changes; batched variants are
//! bit-equal to their row-wise counterparts on the same dispatch path.
//!
//! Also provided: [`f16`] (binary16 → binary32), [`fft`] (radix-2 FFT,
//! inverse real FFT, periodic Hann window) and [`rng`] (xoshiro256**
//! with a Gaussian sampler) — the small infrastructure pieces the TTS
//! runtime needs alongside the kernels.

// Index-based loops are deliberate throughout: they make the summation
// order (the numerical contract) explicit.
#![allow(clippy::needless_range_loop)]

pub mod f16;
pub mod fft;
pub mod ops;
pub mod rng;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub mod ops_neon;

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
pub mod ops_avx2;

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(feature = "force-scalar")
))]
pub mod ops_wasm_simd;

pub use f16::{f16_to_f32, f32_to_f16};
pub use fft::{fft_forward, hann_periodic, irfft_real, FftPlan};
pub use rng::Xoshiro256StarStar;
