//! Numeric kernels: scalar reference implementations plus the dispatch
//! layer that routes to the NEON / AVX2 / WASM SIMD128 variants.
//!
//! Every kernel is stateless and operates on caller-provided slices;
//! streaming state (left-context FIFOs, KV caches) lives in the model
//! code above this crate. Each public `op` has a pure `op_scalar`
//! sibling that never dispatches — the scalar version is the numerical
//! reference the SIMD paths are tested against.
//!
//! Layout conventions:
//! - Dense weights are row-major `[out_features × in_features]`
//!   (PyTorch `nn.Linear.weight` order).
//! - fp16 weights are IEEE binary16 bit patterns (`u16`); int8 weights
//!   are `i8` with one f32 scale per output row. Both are expanded to
//!   f32 in registers and accumulated in f32.
//! - `layer_norm` uses a channel-major `[channels × frames]` buffer;
//!   `conv1d_depthwise_nlc` uses a frame-major `[frames × channels]`
//!   buffer. Everything else is flat or row-major as documented.

use crate::f16::f16_to_f32;

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
macro_rules! avx2_fma_available {
    () => {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    };
}

// ───────────────────────── normalisation ─────────────────────────

/// LayerNorm over the channel axis, independently per frame.
///
/// ```text
/// μ_t  = mean_c input[c, t]
/// σ²_t = mean_c (input[c, t] - μ_t)²          (biased, ÷ channels)
/// output[c, t] = gamma[c] * (input[c, t] - μ_t) / sqrt(σ²_t + eps) + beta[c]
/// ```
///
/// Layout: `input`/`output` are `[channels × frames]` channel-major,
/// `gamma`/`beta` are `[channels]`. The SIMD fast paths cover the
/// `frames == 1` case (one contiguous vector); other shapes run scalar.
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    frames: usize,
    eps: f32,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), channels * frames);
    debug_assert_eq!(output.len(), channels * frames);
    debug_assert_eq!(gamma.len(), channels);
    debug_assert_eq!(beta.len(), channels);

    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    if frames == 1 {
        crate::ops_neon::layer_norm_frames1_neon(input, gamma, beta, channels, eps, output);
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    if frames == 1 {
        crate::ops_wasm_simd::layer_norm_frames1_wasm(input, gamma, beta, channels, eps, output);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if frames == 1 && avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::layer_norm_frames1_avx2(input, gamma, beta, channels, eps, output);
        }
        return;
    }

    layer_norm_scalar(input, gamma, beta, channels, frames, eps, output);
}

/// Pure scalar [`layer_norm`]; never dispatches.
pub fn layer_norm_scalar(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    frames: usize,
    eps: f32,
    output: &mut [f32],
) {
    let inv_c = 1.0_f32 / channels as f32;
    for t in 0..frames {
        let mut sum = 0.0_f32;
        for c in 0..channels {
            sum += input[c * frames + t];
        }
        let mean = sum * inv_c;

        let mut var = 0.0_f32;
        for c in 0..channels {
            let d = input[c * frames + t] - mean;
            var += d * d;
        }
        var *= inv_c;

        let inv_std = 1.0_f32 / (var + eps).sqrt();
        for c in 0..channels {
            let x = input[c * frames + t];
            output[c * frames + t] = gamma[c] * (x - mean) * inv_std + beta[c];
        }
    }
}

/// RMS-style normalisation with an *unbiased* variance (divisor `n - 1`,
/// the `torch.var` default) and **no** mean subtraction in the output:
///
/// ```text
/// var = Σ(x - mean)² / (n - 1);   y = x · alpha / sqrt(eps + var)
/// ```
///
/// Used once per synthesis for the timestep embedding tail of the flow
/// head; scalar only.
pub fn rms_norm_unbiased(input: &[f32], alpha: &[f32], eps: f32, output: &mut [f32]) {
    rms_norm_unbiased_scalar(input, alpha, eps, output)
}

/// Pure scalar [`rms_norm_unbiased`]; never dispatches.
pub fn rms_norm_unbiased_scalar(input: &[f32], alpha: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(alpha.len(), n);
    debug_assert_eq!(output.len(), n);
    debug_assert!(n > 1);
    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n as f32 - 1.0);
    let inv = 1.0 / (eps + var).sqrt();
    for i in 0..n {
        output[i] = input[i] * alpha[i] * inv;
    }
}

/// Adaptive-LayerNorm modulation: `x = x · (1 + scale) + shift`, fused
/// in one pass.
pub fn modulate_inplace(x: &mut [f32], scale: &[f32], shift: &[f32]) {
    debug_assert_eq!(x.len(), scale.len());
    debug_assert_eq!(x.len(), shift.len());
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::modulate_inplace_neon(x, scale, shift);
        return;
    }
    #[allow(unreachable_code)]
    modulate_inplace_scalar(x, scale, shift)
}

/// Pure scalar [`modulate_inplace`]; never dispatches.
pub fn modulate_inplace_scalar(x: &mut [f32], scale: &[f32], shift: &[f32]) {
    for i in 0..x.len() {
        x[i] = x[i] * (1.0 + scale[i]) + shift[i];
    }
}

// ───────────────────────── matrix products (f32) ─────────────────────────

/// Dense matrix–vector product `output = weight · input` (no bias).
///
/// - `input`:  `[in_features]`
/// - `weight`: `[out_features × in_features]` row-major
/// - `output`: `[out_features]`
pub fn matmul_mv(
    input: &[f32],
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);

    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mv_neon(input, weight, in_features, out_features, output);
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::matmul_mv_wasm(input, weight, in_features, out_features, output);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::matmul_mv_avx2(input, weight, in_features, out_features, output);
        }
        return;
    }
    #[allow(unreachable_code)]
    matmul_mv_scalar(input, weight, in_features, out_features, output);
}

/// Pure scalar [`matmul_mv`]; never dispatches.
pub fn matmul_mv_scalar(
    input: &[f32],
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    for o in 0..out_features {
        let w_row = &weight[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0_f32;
        for k in 0..in_features {
            acc += w_row[k] * input[k];
        }
        output[o] = acc;
    }
}

/// Dot product `Σ a[i] · b[i]`.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        return crate::ops_neon::dot_product_f32_neon(a, b);
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        return unsafe { crate::ops_avx2::dot_product_f32_avx2(a, b) };
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        return crate::ops_wasm_simd::dot_product_f32_wasm(a, b);
    }
    #[allow(unreachable_code)]
    dot_product_f32_scalar(a, b)
}

/// Pure scalar [`dot_product_f32`]; never dispatches.
pub fn dot_product_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Dense matrix × matrix, `out = a · b`.
///
/// ```text
/// a:   [m × k]  row-major
/// b:   [k × n]  row-major
/// out: [m × n]  row-major (overwritten, no bias)
/// ```
pub fn matmul_mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mm_neon(a, b, m, k, n, out);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::matmul_mm_avx2(a, b, m, k, n, out);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::matmul_mm_wasm(a, b, m, k, n, out);
        return;
    }
    #[allow(unreachable_code)]
    matmul_mm_scalar(a, b, m, k, n, out);
}

/// Pure scalar [`matmul_mm`]; never dispatches.
pub fn matmul_mm_scalar(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/// Dense matrix × transposed matrix, `out = a · bᵀ`, without
/// materialising `bᵀ`.
///
/// ```text
/// a:   [m × k]  row-major
/// b:   [n × k]  row-major (each row is one column of bᵀ)
/// out: [m × n]  row-major
/// ```
///
/// Used for attention scores `Q · Kᵀ`, where K is naturally stored as
/// `[time × head_dim]`.
pub fn matmul_mm_transposed_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(out.len(), m * n);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mm_transposed_b_neon(a, b, m, k, n, out);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::matmul_mm_transposed_b_avx2(a, b, m, k, n, out);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::matmul_mm_transposed_b_wasm(a, b, m, k, n, out);
        return;
    }
    #[allow(unreachable_code)]
    matmul_mm_transposed_b_scalar(a, b, m, k, n, out);
}

/// Pure scalar [`matmul_mm_transposed_b`]; never dispatches.
pub fn matmul_mm_transposed_b_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
}

// ───────────────────────── matrix products (fp16 weights) ─────────────────────────

/// `output = W · input` with `W` stored as IEEE binary16 bit patterns,
/// row-major `[out_features × in_features]`; fp32 accumulation.
/// Halves weight traffic versus f32 storage, which is what bounds the
/// autoregressive step.
pub fn matmul_mv_f16(
    input: &[f32],
    weight: &[u16],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mv_f16_neon(input, weight, in_features, out_features, output);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() && std::is_x86_feature_detected!("f16c") {
        unsafe {
            crate::ops_avx2::matmul_mv_f16_avx2(input, weight, in_features, out_features, output);
        }
        return;
    }
    #[allow(unreachable_code)]
    matmul_mv_f16_scalar(input, weight, in_features, out_features, output)
}

/// Pure scalar [`matmul_mv_f16`]; never dispatches.
pub fn matmul_mv_f16_scalar(
    input: &[f32],
    weight: &[u16],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    for o in 0..out_features {
        let row = &weight[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(input) {
            acc += f16_to_f32(*w) * x;
        }
        output[o] = acc;
    }
}

/// Batched twin of [`matmul_mv_f16`]: `out[m × n] = a[m × k] · Wᵀ` with
/// `W` fp16 row-major `[n × k]` — the same layout and interpretation as
/// the GEMV, i.e. `m` stacked GEMVs. Each weight row is streamed once
/// for all `m` input rows, so weight traffic is 1× instead of m×
/// (prefill over a whole prompt).
///
/// The per-dot accumulation order is identical to [`matmul_mv_f16`] on
/// every dispatch path (same accumulator count, chunking and
/// reduction), so the result is **bit-equal** to `m` row-wise GEMV
/// calls on the same path.
///
/// Dispatch: on aarch64 the NEON kernel is register-blocked over
/// activation rows (blocks of 4, 2, 1) and converts each fp16 weight
/// row in register exactly once per block — no scratch panel, no
/// allocation, used on every step. On x86_64 the AVX2 + F16C kernel does
/// the same with blocks of 4 rows and a 1-row remainder, converting each
/// 16-wide weight chunk in register. wasm32 runs the scalar reference.
pub fn matmul_mm_f16(a: &[f32], weight: &[u16], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(weight.len(), n * k);
    debug_assert_eq!(out.len(), m * n);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mm_f16_neon(a, weight, m, k, n, out);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() && std::is_x86_feature_detected!("f16c") {
        unsafe {
            crate::ops_avx2::matmul_mm_f16_avx2(a, weight, m, k, n, out);
        }
        return;
    }
    #[allow(unreachable_code)]
    matmul_mm_f16_scalar(a, weight, m, k, n, out)
}

/// Pure scalar [`matmul_mm_f16`]; sequential k-accumulation per dot,
/// bit-equal to `m` calls of [`matmul_mv_f16_scalar`].
pub fn matmul_mm_f16_scalar(
    a: &[f32],
    weight: &[u16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    for j in 0..n {
        let w_row = &weight[j * k..(j + 1) * k];
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let mut acc = 0.0f32;
            for (w, x) in w_row.iter().zip(a_row) {
                acc += f16_to_f32(*w) * x;
            }
            out[i * n + j] = acc;
        }
    }
}

/// Read row `row` of an fp16 embedding table `[rows × dim]` into f32.
pub fn embed_lookup_f16(table: &[u16], dim: usize, row: usize, out: &mut [f32]) {
    embed_lookup_f16_scalar(table, dim, row, out)
}

/// Pure scalar [`embed_lookup_f16`]; never dispatches.
pub fn embed_lookup_f16_scalar(table: &[u16], dim: usize, row: usize, out: &mut [f32]) {
    debug_assert!(table.len() >= (row + 1) * dim);
    debug_assert_eq!(out.len(), dim);
    let src = &table[row * dim..(row + 1) * dim];
    for (o, h) in out.iter_mut().zip(src) {
        *o = f16_to_f32(*h);
    }
}

// ───────────────────────── matrix products (int8 weights) ─────────────────────────

/// Weight-only int8 GEMV with one scale per output row:
/// `output[o] = scale[o] · Σ_i w[o, i] · x[i]`, fp32 accumulation.
/// The scale is applied once, after the whole reduction.
pub fn matmul_mv_i8(
    input: &[f32],
    weight: &[i8],
    scale: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(scale.len(), out_features);
    debug_assert_eq!(output.len(), out_features);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mv_i8_neon(input, weight, scale, in_features, out_features, output);
        return;
    }
    #[allow(unreachable_code)]
    matmul_mv_i8_scalar(input, weight, scale, in_features, out_features, output)
}

/// Pure scalar [`matmul_mv_i8`]; never dispatches.
pub fn matmul_mv_i8_scalar(
    input: &[f32],
    weight: &[i8],
    scale: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    for o in 0..out_features {
        let row = &weight[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for (w, x) in row.iter().zip(input) {
            acc += *w as f32 * x;
        }
        output[o] = acc * scale[o];
    }
}

/// Batched twin of [`matmul_mv_i8`]: `out[t × out_f] = a[t × in_f] · Wᵀ`
/// with `W` int8 row-major `[out_f × in_f]` and per-row `scale[out_f]`.
/// Streams each weight row once for all `t` input rows.
///
/// Per `(row, o)` the reduction is literally [`matmul_mv_i8`] (int8 →
/// f32 in register, f32 accumulator, same order) and the row scale is
/// applied once, so the result is **bit-equal** to `t` GEMV calls on
/// the same dispatch path.
pub fn matmul_mm_i8(
    a: &[f32],
    w: &[i8],
    scale: &[f32],
    t: usize,
    in_f: usize,
    out_f: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(a.len(), t * in_f);
    debug_assert_eq!(w.len(), out_f * in_f);
    debug_assert_eq!(scale.len(), out_f);
    debug_assert_eq!(out.len(), t * out_f);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::matmul_mm_i8_neon(a, w, scale, t, in_f, out_f, out);
        return;
    }
    #[allow(unreachable_code)]
    matmul_mm_i8_scalar(a, w, scale, t, in_f, out_f, out)
}

/// Pure scalar [`matmul_mm_i8`]; sequential k-accumulation per dot with
/// the scale applied last, bit-equal to `t` calls of
/// [`matmul_mv_i8_scalar`].
pub fn matmul_mm_i8_scalar(
    a: &[f32],
    w: &[i8],
    scale: &[f32],
    t: usize,
    in_f: usize,
    out_f: usize,
    out: &mut [f32],
) {
    for o in 0..out_f {
        let w_row = &w[o * in_f..(o + 1) * in_f];
        let s = scale[o];
        for r in 0..t {
            let a_row = &a[r * in_f..(r + 1) * in_f];
            let mut acc = 0.0f32;
            for (wv, x) in w_row.iter().zip(a_row) {
                acc += *wv as f32 * x;
            }
            out[r * out_f + o] = acc * s;
        }
    }
}

// ───────────────────────── element-wise ─────────────────────────

/// `a[i] += b[i]`. Residual sums. Bit-exact across all dispatch paths
/// (one rounding per element).
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::add_inplace_neon(a, b);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            crate::ops_avx2::add_inplace_avx2(a, b);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::add_inplace_wasm(a, b);
        return;
    }
    #[allow(unreachable_code)]
    add_inplace_scalar(a, b);
}

/// Pure scalar [`add_inplace`]; never dispatches.
pub fn add_inplace_scalar(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// `a[i] += scale · b[i]`. The NEON and AVX2 paths use a fused
/// multiply-add (single rounding), so they may differ from the scalar
/// two-step result by ≤ 1 ULP per element; the WASM path is bit-exact.
pub fn add_scaled_inplace(a: &mut [f32], b: &[f32], scale: f32) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::add_scaled_inplace_neon(a, b, scale);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::add_scaled_inplace_avx2(a, b, scale);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::add_scaled_inplace_wasm(a, b, scale);
        return;
    }
    #[allow(unreachable_code)]
    add_scaled_inplace_scalar(a, b, scale);
}

/// Pure scalar [`add_scaled_inplace`]; never dispatches.
pub fn add_scaled_inplace_scalar(a: &mut [f32], b: &[f32], scale: f32) {
    for i in 0..a.len() {
        a[i] += scale * b[i];
    }
}

/// `a[i] += b[i] · c[i]` (element-wise multiply-add; LayerScale
/// residuals). NEON path is FMA-fused (≤ 1 ULP vs scalar).
pub fn add_mul_inplace(a: &mut [f32], b: &[f32], c: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::add_mul_inplace_neon(a, b, c);
        return;
    }
    #[allow(unreachable_code)]
    add_mul_inplace_scalar(a, b, c)
}

/// Pure scalar [`add_mul_inplace`]; never dispatches.
pub fn add_mul_inplace_scalar(a: &mut [f32], b: &[f32], c: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i] * c[i];
    }
}

// ───────────────────────── activations ─────────────────────────

/// In-place SiLU / Swish: `x · sigmoid(x)`. SIMD paths use a polynomial
/// `exp` (≤ 1e-6 abs error on the sigmoid range); the scalar path uses
/// libm.
pub fn silu_inplace(buf: &mut [f32]) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::silu_inplace_neon(buf);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::silu_inplace_avx2(buf);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::silu_inplace_wasm(buf);
        return;
    }
    #[allow(unreachable_code)]
    silu_inplace_scalar(buf);
}

/// Pure scalar [`silu_inplace`]; never dispatches.
pub fn silu_inplace_scalar(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        let s = 1.0_f32 / (1.0 + (-*x).exp());
        *x *= s;
    }
}

/// In-place erf-GELU (`torch.nn.functional.gelu`, default form). The
/// erf is the Abramowitz–Stegun 7.1.26 rational approximation,
/// |err| ≤ 1.5e-7.
pub fn gelu_inplace(buf: &mut [f32]) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::gelu_inplace_neon(buf);
        return;
    }
    #[allow(unreachable_code)]
    gelu_inplace_scalar(buf)
}

/// Pure scalar [`gelu_inplace`]; never dispatches.
pub fn gelu_inplace_scalar(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf_scalar(*v * core::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Abramowitz–Stegun 7.1.26 erf approximation (|err| ≤ 1.5e-7).
// Coefficients written out as published; they round to the intended f32.
#[allow(clippy::excessive_precision)]
pub fn erf_scalar(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    s * y
}

/// In-place tanh-GELU (`torch.nn.functional.gelu(x, approximate="tanh")`):
/// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`. Distinct from
/// [`gelu_inplace`] — the two forms differ by up to ~1e-3 per activation.
pub fn gelu_tanh_inplace(buf: &mut [f32]) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::gelu_tanh_inplace_neon(buf);
        return;
    }
    #[allow(unreachable_code)]
    gelu_tanh_inplace_scalar(buf)
}

/// Pure scalar [`gelu_tanh_inplace`]; never dispatches.
pub fn gelu_tanh_inplace_scalar(buf: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in buf.iter_mut() {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// In-place row-wise softmax over a `[rows × cols]` row-major matrix,
/// with the usual max subtraction. Rows whose maximum is not finite
/// (all `-inf`, or NaN) become all zeros instead of propagating NaN.
pub fn softmax_row_inplace(scores: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::softmax_row_inplace_neon(scores, rows, cols);
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if avx2_fma_available!() {
        unsafe {
            crate::ops_avx2::softmax_row_inplace_avx2(scores, rows, cols);
        }
        return;
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(feature = "force-scalar")
    ))]
    {
        crate::ops_wasm_simd::softmax_row_inplace_wasm(scores, rows, cols);
        return;
    }
    #[allow(unreachable_code)]
    softmax_row_inplace_scalar(scores, rows, cols);
}

/// Pure scalar [`softmax_row_inplace`]; never dispatches.
pub fn softmax_row_inplace_scalar(scores: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > m {
                m = v;
            }
        }
        if !m.is_finite() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// In-place `exp(clamp(x, lo, hi))` — the log-magnitude → magnitude
/// step of the iSTFT head. Deliberately scalar: libm `exp` matches the
/// PyTorch reference to < 1 ULP, whereas the polynomial SIMD `exp` is
/// calibrated for the softmax range and would scale its error at
/// outputs ≫ 1. The call count is tiny, so SIMD would not pay off.
pub fn exp_clamp_inplace(buf: &mut [f32], lo: f32, hi: f32) {
    exp_clamp_inplace_scalar(buf, lo, hi)
}

/// Pure scalar [`exp_clamp_inplace`]; never dispatches.
pub fn exp_clamp_inplace_scalar(buf: &mut [f32], lo: f32, hi: f32) {
    for v in buf.iter_mut() {
        *v = v.clamp(lo, hi).exp();
    }
}

/// `p → (cos p, sin p)` for every phase value, into two outputs. Scalar
/// by design for the same reason as [`exp_clamp_inplace`]: libm
/// `sin`/`cos` match the reference to < 1 ULP at any |p|.
pub fn sincos_into(phase: &[f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
    sincos_into_scalar(phase, cos_out, sin_out)
}

/// Pure scalar [`sincos_into`]; never dispatches.
pub fn sincos_into_scalar(phase: &[f32], cos_out: &mut [f32], sin_out: &mut [f32]) {
    debug_assert_eq!(phase.len(), cos_out.len());
    debug_assert_eq!(phase.len(), sin_out.len());
    for (i, &p) in phase.iter().enumerate() {
        let (s, c) = p.sin_cos();
        cos_out[i] = c;
        sin_out[i] = s;
    }
}

// ───────────────────────── convolution ─────────────────────────

/// Depthwise Conv1D over a **channels-last** (`[frames × channels]`)
/// buffer, dilation 1, stride 1. The caller supplies the left context:
/// `frames_out = frames_in - (kernel - 1)`.
///
/// - `input`:  `[frames_in × channels]`, frame-major
/// - `weight_kmajor`: `[kernel × channels]` — tap-major. Repack a
///   PyTorch `[C, 1, K]` depthwise weight as `wk[k·C + c] = w[c·K + k]`
///   at load time.
/// - `bias`:   `[channels]`
/// - `output`: `[frames_out × channels]`
///
/// The summation order over `k` is identical in the scalar and SIMD
/// kernels (vector across channels, sequential accumulator over taps),
/// so a batched call over `T` frames is bit-equal to `T` single-frame
/// calls over the sliding window on the same dispatch path — offline
/// and streaming decode agree exactly.
pub fn conv1d_depthwise_nlc(
    input: &[f32],
    weight_kmajor: &[f32],
    bias: &[f32],
    channels: usize,
    kernel: usize,
    frames_in: usize,
    output: &mut [f32],
) {
    let frames_out = frames_in
        .checked_sub(kernel - 1)
        .expect("frames_in must include left context");
    debug_assert_eq!(input.len(), frames_in * channels);
    debug_assert_eq!(weight_kmajor.len(), kernel * channels);
    debug_assert_eq!(bias.len(), channels);
    debug_assert_eq!(output.len(), frames_out * channels);

    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    {
        crate::ops_neon::conv1d_depthwise_nlc_neon(
            input,
            weight_kmajor,
            bias,
            channels,
            kernel,
            frames_in,
            output,
        );
        return;
    }
    // x86_64 / wasm32: scalar. The depthwise pass is a negligible share
    // of the decoder's work next to the pointwise matmuls.
    #[allow(unreachable_code)]
    conv1d_depthwise_nlc_scalar(
        input,
        weight_kmajor,
        bias,
        channels,
        kernel,
        frames_in,
        output,
    )
}

/// Pure scalar [`conv1d_depthwise_nlc`]; never dispatches.
pub fn conv1d_depthwise_nlc_scalar(
    input: &[f32],
    weight_kmajor: &[f32],
    bias: &[f32],
    channels: usize,
    kernel: usize,
    frames_in: usize,
    output: &mut [f32],
) {
    let frames_out = frames_in - (kernel - 1);
    for t in 0..frames_out {
        let dst = &mut output[t * channels..(t + 1) * channels];
        dst.copy_from_slice(bias);
        for k in 0..kernel {
            let w = &weight_kmajor[k * channels..(k + 1) * channels];
            let x = &input[(t + k) * channels..(t + k + 1) * channels];
            for c in 0..channels {
                dst[c] += w[c] * x[c];
            }
        }
    }
}
