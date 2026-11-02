//! aarch64 NEON kernels.
//!
//! Pattern used throughout: 4 parallel `float32x4_t` accumulators over
//! 16-wide chunks, a 4-wide tail, a horizontal reduce (`vaddvq_f32`),
//! then a scalar remainder. NEON has a fused multiply-add
//! (`vfmaq_f32`), so where the scalar reference does `mul` then `add`
//! the vector path rounds once; element-wise kernels therefore match
//! scalar to ≤ 1 ULP and reductions to a few ULP.
//!
//! NEON is a baseline feature of AArch64, so no runtime detection is
//! needed; the module is gated on `target_arch = "aarch64"`.

#![cfg(target_arch = "aarch64")]
// The exp polynomial constants below are written out in full so they
// stay bit-identical to the reference implementation.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use core::arch::aarch64::*;

/// NEON LayerNorm for a single frame (`channels` contiguous values):
/// three passes (sum, squared deviations, apply) with 4-wide chunks and
/// a scalar remainder.
pub fn layer_norm_frames1_neon(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    channels: usize,
    eps: f32,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), channels);
    debug_assert_eq!(output.len(), channels);
    debug_assert_eq!(gamma.len(), channels);
    debug_assert_eq!(beta.len(), channels);

    let chunks4 = channels / 4;
    let rem_start = chunks4 * 4;

    // Pass 1: Σ input[c]
    let mut sum_v = unsafe { vdupq_n_f32(0.0) };
    for c in 0..chunks4 {
        unsafe {
            sum_v = vaddq_f32(sum_v, vld1q_f32(input.as_ptr().add(c * 4)));
        }
    }
    let mut sum: f32 = unsafe { vaddvq_f32(sum_v) };
    for j in rem_start..channels {
        sum += input[j];
    }
    let inv_c = 1.0_f32 / channels as f32;
    let mean = sum * inv_c;

    // Pass 2: Σ (input[c] - μ)²
    let mean_v = unsafe { vdupq_n_f32(mean) };
    let mut var_v = unsafe { vdupq_n_f32(0.0) };
    for c in 0..chunks4 {
        unsafe {
            let x = vld1q_f32(input.as_ptr().add(c * 4));
            let d = vsubq_f32(x, mean_v);
            var_v = vfmaq_f32(var_v, d, d);
        }
    }
    let mut var: f32 = unsafe { vaddvq_f32(var_v) };
    for j in rem_start..channels {
        let d = input[j] - mean;
        var += d * d;
    }
    var *= inv_c;

    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let inv_std_v = unsafe { vdupq_n_f32(inv_std) };

    // Pass 3: output[c] = gamma * (input - μ) * inv_std + beta
    for c in 0..chunks4 {
        unsafe {
            let x = vld1q_f32(input.as_ptr().add(c * 4));
            let g = vld1q_f32(gamma.as_ptr().add(c * 4));
            let b = vld1q_f32(beta.as_ptr().add(c * 4));
            let centred = vsubq_f32(x, mean_v);
            let scaled = vmulq_f32(centred, inv_std_v);
            let y = vfmaq_f32(b, g, scaled);
            vst1q_f32(output.as_mut_ptr().add(c * 4), y);
        }
    }
    for j in rem_start..channels {
        output[j] = gamma[j] * (input[j] - mean) * inv_std + beta[j];
    }
}

/// NEON dense mat-vec `output[o] = Σ_k weight[o, k] · input[k]`
/// (no bias). 16-wide outer chunks with 4 accumulators, 4-wide tail,
/// scalar remainder.
pub fn matmul_mv_neon(
    input: &[f32],
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);

    let chunks16 = in_features / 16;
    let chunks4_after = (in_features % 16) / 4;
    let rem_start = chunks16 * 16 + chunks4_after * 4;

    for o in 0..out_features {
        let w_row = &weight[o * in_features..(o + 1) * in_features];

        let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
            (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            )
        };

        for c in 0..chunks16 {
            let b = c * 16;
            unsafe {
                let x0 = vld1q_f32(input.as_ptr().add(b));
                let x1 = vld1q_f32(input.as_ptr().add(b + 4));
                let x2 = vld1q_f32(input.as_ptr().add(b + 8));
                let x3 = vld1q_f32(input.as_ptr().add(b + 12));
                let w0 = vld1q_f32(w_row.as_ptr().add(b));
                let w1 = vld1q_f32(w_row.as_ptr().add(b + 4));
                let w2 = vld1q_f32(w_row.as_ptr().add(b + 8));
                let w3 = vld1q_f32(w_row.as_ptr().add(b + 12));
                acc0 = vfmaq_f32(acc0, x0, w0);
                acc1 = vfmaq_f32(acc1, x1, w1);
                acc2 = vfmaq_f32(acc2, x2, w2);
                acc3 = vfmaq_f32(acc3, x3, w3);
            }
        }

        let after16 = chunks16 * 16;
        for k in 0..chunks4_after {
            let b = after16 + k * 4;
            unsafe {
                let xv = vld1q_f32(input.as_ptr().add(b));
                let wv = vld1q_f32(w_row.as_ptr().add(b));
                acc0 = vfmaq_f32(acc0, xv, wv);
            }
        }

        let acc_v = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
        let mut acc: f32 = unsafe { vaddvq_f32(acc_v) };
        for j in rem_start..in_features {
            acc += input[j] * w_row[j];
        }
        output[o] = acc;
    }
}

/// NEON in-place `a[i] += b[i]`. Bit-exact vs scalar (one add per
/// element).
pub fn add_inplace_neon(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    for c in 0..chunks4 {
        unsafe {
            let av = vld1q_f32(a.as_ptr().add(c * 4));
            let bv = vld1q_f32(b.as_ptr().add(c * 4));
            vst1q_f32(a.as_mut_ptr().add(c * 4), vaddq_f32(av, bv));
        }
    }
    for j in rem_start..n {
        a[j] += b[j];
    }
}

/// NEON in-place `a[i] += scale · b[i]` via `vfmaq_f32`. Single rounding
/// where the scalar reference rounds twice → ≤ 1 ULP per element.
pub fn add_scaled_inplace_neon(a: &mut [f32], b: &[f32], scale: f32) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    let sv = unsafe { vdupq_n_f32(scale) };
    for c in 0..chunks4 {
        unsafe {
            let av = vld1q_f32(a.as_ptr().add(c * 4));
            let bv = vld1q_f32(b.as_ptr().add(c * 4));
            vst1q_f32(a.as_mut_ptr().add(c * 4), vfmaq_f32(av, bv, sv));
        }
    }
    for j in rem_start..n {
        a[j] += scale * b[j];
    }
}

/// NEON in-place `a[i] += b[i] · c[i]` via `vfmaq_f32` (≤ 1 ULP vs
/// scalar).
pub fn add_mul_inplace_neon(a: &mut [f32], b: &[f32], c: &[f32]) {
    let n = a.len();
    let chunks4 = n / 4;
    for i in 0..chunks4 {
        unsafe {
            let p = a.as_mut_ptr().add(i * 4);
            let va = vld1q_f32(p);
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let vc = vld1q_f32(c.as_ptr().add(i * 4));
            vst1q_f32(p, vfmaq_f32(va, vb, vc));
        }
    }
    for i in chunks4 * 4..n {
        a[i] += b[i] * c[i];
    }
}

/// NEON adaptive-LayerNorm modulation `x = x · (1 + scale) + shift`,
/// computed as `(x + x·scale) + shift` with one FMA and one add.
pub fn modulate_inplace_neon(x: &mut [f32], scale: &[f32], shift: &[f32]) {
    let n = x.len();
    let chunks4 = n / 4;
    for i in 0..chunks4 {
        unsafe {
            let p = x.as_mut_ptr().add(i * 4);
            let vx = vld1q_f32(p);
            let vs = vld1q_f32(scale.as_ptr().add(i * 4));
            let vh = vld1q_f32(shift.as_ptr().add(i * 4));
            vst1q_f32(p, vaddq_f32(vfmaq_f32(vx, vx, vs), vh));
        }
    }
    for i in chunks4 * 4..n {
        x[i] = x[i] * (1.0 + scale[i]) + shift[i];
    }
}

/// NEON dense matmul `out = a · b`. 4-column blocking: broadcast
/// `a[i, p]`, load `b[p, j..j+4]`, FMA-accumulate per lane; each lane
/// runs the same `p` sequence as the scalar reference.
pub fn matmul_mm_neon(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    let n_chunks4 = n / 4;
    let n_rem_start = n_chunks4 * 4;

    for i in 0..m {
        let a_row_off = i * k;
        let out_row_off = i * n;

        for cc in 0..n_chunks4 {
            let j0 = cc * 4;
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for p in 0..k {
                unsafe {
                    let av = vdupq_n_f32(*a.get_unchecked(a_row_off + p));
                    let bv = vld1q_f32(b.as_ptr().add(p * n + j0));
                    acc = vfmaq_f32(acc, av, bv);
                }
            }
            unsafe {
                vst1q_f32(out.as_mut_ptr().add(out_row_off + j0), acc);
            }
        }

        for j in n_rem_start..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[a_row_off + p] * b[p * n + j];
            }
            out[out_row_off + j] = acc;
        }
    }
}

/// NEON `out = a · bᵀ`. 16-wide k-loop with 4 FMA accumulators and a
/// horizontal sum per output cell (a few ULP vs scalar).
pub fn matmul_mm_transposed_b_neon(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    let chunks16 = k / 16;
    let chunks4_after = (k % 16) / 4;
    let rem_start = chunks16 * 16 + chunks4_after * 4;

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row_off = i * n;
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let (mut acc0, mut acc1, mut acc2, mut acc3) = unsafe {
                (
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                )
            };

            for c in 0..chunks16 {
                let base = c * 16;
                unsafe {
                    let a0 = vld1q_f32(a_row.as_ptr().add(base));
                    let a1 = vld1q_f32(a_row.as_ptr().add(base + 4));
                    let a2 = vld1q_f32(a_row.as_ptr().add(base + 8));
                    let a3 = vld1q_f32(a_row.as_ptr().add(base + 12));
                    let b0 = vld1q_f32(b_row.as_ptr().add(base));
                    let b1 = vld1q_f32(b_row.as_ptr().add(base + 4));
                    let b2 = vld1q_f32(b_row.as_ptr().add(base + 8));
                    let b3 = vld1q_f32(b_row.as_ptr().add(base + 12));
                    acc0 = vfmaq_f32(acc0, a0, b0);
                    acc1 = vfmaq_f32(acc1, a1, b1);
                    acc2 = vfmaq_f32(acc2, a2, b2);
                    acc3 = vfmaq_f32(acc3, a3, b3);
                }
            }

            let after16 = chunks16 * 16;
            for cc in 0..chunks4_after {
                let base = after16 + cc * 4;
                unsafe {
                    let av = vld1q_f32(a_row.as_ptr().add(base));
                    let bv = vld1q_f32(b_row.as_ptr().add(base));
                    acc0 = vfmaq_f32(acc0, av, bv);
                }
            }

            let acc_v = unsafe { vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)) };
            let mut acc: f32 = unsafe { vaddvq_f32(acc_v) };
            for p in rem_start..k {
                acc += a_row[p] * b_row[p];
            }
            out[out_row_off + j] = acc;
        }
    }
}

/// NEON dot product `Σ a[i] · b[i]`: one 4-wide FMA accumulator,
/// `vaddvq_f32`, scalar tail (a few ULP vs scalar).
pub fn dot_product_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for c in 0..chunks4 {
        unsafe {
            let av = vld1q_f32(a.as_ptr().add(c * 4));
            let bv = vld1q_f32(b.as_ptr().add(c * 4));
            acc = vfmaq_f32(acc, av, bv);
        }
    }
    let mut s = unsafe { vaddvq_f32(acc) };
    for i in rem_start..n {
        s += a[i] * b[i];
    }
    s
}

// ───────── exp / sigmoid / erf polynomial primitives ─────────
//
// Cody–Waite range reduction + degree-7 Horner polynomial (FMA, single
// rounding per step) + IEEE-754 exponent-bias trick for 2^n. Error
// ≤ 1e-6 abs vs libm `expf` on the sigmoid / softmax input range.

const EXP_LOG2_E: f32 = 1.442_695_040_888_963_4_f32;
const EXP_LN2_HI: f32 = 0.693_145_751_953_125_f32;
const EXP_LN2_LO: f32 = 1.428_606_820_309_101_e-6_f32;
const EXP_C0: f32 = 1.0;
const EXP_C1: f32 = 1.0;
const EXP_C2: f32 = 0.5;
const EXP_C3: f32 = 1.0 / 6.0;
const EXP_C4: f32 = 1.0 / 24.0;
const EXP_C5: f32 = 1.0 / 120.0;
const EXP_C6: f32 = 1.0 / 720.0;
const EXP_C7: f32 = 1.0 / 5040.0;
const EXP_CLAMP: f32 = 87.0;

#[inline]
unsafe fn expf_neon_4(x: float32x4_t) -> float32x4_t {
    let x = unsafe { vminq_f32(x, vdupq_n_f32(EXP_CLAMP)) };
    let x = unsafe { vmaxq_f32(x, vdupq_n_f32(-EXP_CLAMP)) };
    let n_f = unsafe { vrndnq_f32(vmulq_n_f32(x, EXP_LOG2_E)) };
    let n_i = unsafe { vcvtq_s32_f32(n_f) };
    let r = unsafe { vfmsq_f32(x, n_f, vdupq_n_f32(EXP_LN2_HI)) };
    let r = unsafe { vfmsq_f32(r, n_f, vdupq_n_f32(EXP_LN2_LO)) };
    let p = unsafe { vfmaq_n_f32(vdupq_n_f32(EXP_C6), r, EXP_C7) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C5), p, r) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C4), p, r) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C3), p, r) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C2), p, r) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C1), p, r) };
    let p = unsafe { vfmaq_f32(vdupq_n_f32(EXP_C0), p, r) };
    let pow2_bits = unsafe { vshlq_n_s32::<23>(vaddq_s32(n_i, vdupq_n_s32(127))) };
    let pow2 = unsafe { vreinterpretq_f32_s32(pow2_bits) };
    unsafe { vmulq_f32(p, pow2) }
}

/// 4-wide sigmoid, numerically stable on both signs:
/// `y = exp(-|x|); x ≥ 0 ? 1/(1+y) : y/(1+y)`.
#[inline]
unsafe fn sigmoid_neon_4(x: float32x4_t) -> float32x4_t {
    let abs_x = unsafe { vabsq_f32(x) };
    let neg_abs = unsafe { vnegq_f32(abs_x) };
    let y = unsafe { expf_neon_4(neg_abs) };
    let one = unsafe { vdupq_n_f32(1.0) };
    let denom = unsafe { vaddq_f32(one, y) };
    let inv_denom = unsafe { vdivq_f32(one, denom) };
    let sig_pos = inv_denom;
    let sig_neg = unsafe { vmulq_f32(y, inv_denom) };
    let mask = unsafe { vcgeq_f32(x, vdupq_n_f32(0.0)) };
    unsafe { vbslq_f32(mask, sig_pos, sig_neg) }
}

/// 4-wide erf via the Abramowitz–Stegun 7.1.26 rational form on top of
/// [`expf_neon_4`]; matches [`crate::ops::erf_scalar`] to ~1e-7.
#[inline(always)]
unsafe fn erf_neon_4(x: float32x4_t) -> float32x4_t {
    let sign_mask = vdupq_n_u32(0x8000_0000);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let ax = vabsq_f32(x);
    let t = vdivq_f32(
        vdupq_n_f32(1.0),
        vfmaq_f32(vdupq_n_f32(1.0), ax, vdupq_n_f32(0.327_591_1)),
    );
    // Horner: ((((a5·t + a4)·t + a3)·t + a2)·t + a1)·t
    let mut p = vdupq_n_f32(1.061_405_429);
    p = vfmaq_f32(vdupq_n_f32(-1.453_152_027), p, t);
    p = vfmaq_f32(vdupq_n_f32(1.421_413_741), p, t);
    p = vfmaq_f32(vdupq_n_f32(-0.284_496_736), p, t);
    p = vfmaq_f32(vdupq_n_f32(0.254_829_592), p, t);
    p = vmulq_f32(p, t);
    let e = expf_neon_4(vnegq_f32(vmulq_f32(ax, ax)));
    let y = vfmaq_f32(vdupq_n_f32(1.0), vnegq_f32(p), e); // 1 - p·e
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(y), sign))
}

/// NEON in-place SiLU: `x[i] *= sigmoid(x[i])`. Scalar (libm) tail.
pub fn silu_inplace_neon(buf: &mut [f32]) {
    let n = buf.len();
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    for c in 0..chunks4 {
        unsafe {
            let ptr = buf.as_mut_ptr().add(c * 4);
            let v = vld1q_f32(ptr);
            let s = sigmoid_neon_4(v);
            vst1q_f32(ptr, vmulq_f32(v, s));
        }
    }
    for i in rem_start..n {
        let x = buf[i];
        let s = 1.0_f32 / (1.0 + (-x).exp());
        buf[i] = x * s;
    }
}

/// NEON in-place erf-GELU: `0.5 · x · (1 + erf(x / √2))`. Scalar tail
/// uses [`crate::ops::erf_scalar`].
pub fn gelu_inplace_neon(buf: &mut [f32]) {
    let n = buf.len();
    let chunks4 = n / 4;
    let (inv_sqrt2, half, one) = unsafe {
        (
            vdupq_n_f32(core::f32::consts::FRAC_1_SQRT_2),
            vdupq_n_f32(0.5),
            vdupq_n_f32(1.0),
        )
    };
    for c in 0..chunks4 {
        unsafe {
            let ptr = buf.as_mut_ptr().add(c * 4);
            let x = vld1q_f32(ptr);
            let e = erf_neon_4(vmulq_f32(x, inv_sqrt2));
            vst1q_f32(ptr, vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, e)));
        }
    }
    for v in buf[chunks4 * 4..].iter_mut() {
        *v = 0.5 * *v * (1.0 + crate::ops::erf_scalar(*v * core::f32::consts::FRAC_1_SQRT_2));
    }
}

/// NEON in-place tanh-GELU `0.5·x·(1 + tanh(inner))`,
/// `inner = √(2/π)·(x + 0.044715·x³)`. Uses the identity
/// `tanh(u) = 2σ(2u) − 1`, so `0.5·(1 + tanh(inner)) = σ(2·inner)` and
/// `gelu_tanh(x) = x · σ(2·inner)` reuses [`sigmoid_neon_4`]. The tail
/// (< 4 elements) uses libm `tanh`, identical to the scalar reference.
pub fn gelu_tanh_inplace_neon(buf: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let n = buf.len();
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    unsafe {
        let c0 = vdupq_n_f32(SQRT_2_OVER_PI);
        let c3 = vdupq_n_f32(SQRT_2_OVER_PI * 0.044_715);
        let two = vdupq_n_f32(2.0);
        for c in 0..chunks4 {
            let ptr = buf.as_mut_ptr().add(c * 4);
            let x = vld1q_f32(ptr);
            let x2 = vmulq_f32(x, x);
            // inner = c0·x + c3·x³ = x·(c0 + c3·x²)
            let poly = vfmaq_f32(c0, c3, x2);
            let inner = vmulq_f32(x, poly);
            let s = sigmoid_neon_4(vmulq_f32(two, inner));
            vst1q_f32(ptr, vmulq_f32(x, s));
        }
    }
    for v in buf.iter_mut().take(n).skip(rem_start) {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// NEON in-place row-wise softmax. Per row: 4-wide max + `vmaxvq_f32`,
/// 4-wide `exp(v - m)` with vector sum, 4-wide multiply by `1/sum`.
/// Rows with a non-finite max become all zeros (matches scalar).
pub fn softmax_row_inplace_neon(scores: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    let chunks4 = cols / 4;
    let rem_start = chunks4 * 4;

    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];

        // 1. Row max.
        let mut m = f32::NEG_INFINITY;
        if chunks4 > 0 {
            let mut mv = unsafe { vld1q_f32(row.as_ptr()) };
            for c in 1..chunks4 {
                unsafe {
                    let v = vld1q_f32(row.as_ptr().add(c * 4));
                    mv = vmaxq_f32(mv, v);
                }
            }
            let m_chunks = unsafe { vmaxvq_f32(mv) };
            if m_chunks > m {
                m = m_chunks;
            }
        }
        for j in rem_start..cols {
            if row[j] > m {
                m = row[j];
            }
        }

        if !m.is_finite() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }

        // 2. exp(v - m), accumulate sum.
        let mv = unsafe { vdupq_n_f32(m) };
        let mut sum_v = unsafe { vdupq_n_f32(0.0) };
        for c in 0..chunks4 {
            unsafe {
                let ptr = row.as_mut_ptr().add(c * 4);
                let v = vld1q_f32(ptr);
                let e = expf_neon_4(vsubq_f32(v, mv));
                sum_v = vaddq_f32(sum_v, e);
                vst1q_f32(ptr, e);
            }
        }
        let mut sum = unsafe { vaddvq_f32(sum_v) };
        for j in rem_start..cols {
            let e = (row[j] - m).exp();
            row[j] = e;
            sum += e;
        }

        // 3. Divide by sum.
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let inv_v = unsafe { vdupq_n_f32(inv) };
        for c in 0..chunks4 {
            unsafe {
                let ptr = row.as_mut_ptr().add(c * 4);
                let v = vld1q_f32(ptr);
                vst1q_f32(ptr, vmulq_f32(v, inv_v));
            }
        }
        for j in rem_start..cols {
            row[j] *= inv;
        }
    }
}

// ───────── fp16-resident matrix products ─────────

/// Half → float conversion of 4 lanes via the "magic multiply": place
/// the fp16 exponent + mantissa bits into the f32 field positions and
/// multiply by 2^112 to re-bias. Exact for normals and subnormals (the
/// multiply performs the denormal shift); ±inf / NaN are not mapped,
/// which is fine because model weights never contain them.
#[inline(always)]
unsafe fn f16x4_to_f32x4(h: uint16x4_t) -> float32x4_t {
    let ext = vshll_n_u16(h, 0); // widen u16 → u32
    let sign = vshlq_n_u32(vandq_u32(ext, vdupq_n_u32(0x8000)), 16);
    let em = vshlq_n_u32(vandq_u32(ext, vdupq_n_u32(0x7FFF)), 13);
    let magic = vdupq_n_f32(f32::from_bits(0x7780_0000)); // 2^112
    let val = vmulq_f32(vreinterpretq_f32_u32(em), magic);
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(val), sign))
}

/// fp16-resident GEMV: fp32 accumulation, 4 accumulators over 16-wide
/// chunks, scalar tail — same skeleton as [`matmul_mv_neon`] with the
/// weights dequantised in-register.
pub fn matmul_mv_f16_neon(
    input: &[f32],
    weight: &[u16],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    let chunks16 = in_features / 16;
    let rem_start = chunks16 * 16;
    for o in 0..out_features {
        let w_row = &weight[o * in_features..(o + 1) * in_features];
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            for c in 0..chunks16 {
                let b = c * 16;
                let h0 = vld1_u16(w_row.as_ptr().add(b));
                let h1 = vld1_u16(w_row.as_ptr().add(b + 4));
                let h2 = vld1_u16(w_row.as_ptr().add(b + 8));
                let h3 = vld1_u16(w_row.as_ptr().add(b + 12));
                acc0 = vfmaq_f32(acc0, vld1q_f32(input.as_ptr().add(b)), f16x4_to_f32x4(h0));
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(input.as_ptr().add(b + 4)),
                    f16x4_to_f32x4(h1),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(input.as_ptr().add(b + 8)),
                    f16x4_to_f32x4(h2),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(input.as_ptr().add(b + 12)),
                    f16x4_to_f32x4(h3),
                );
            }
            let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            for i in rem_start..in_features {
                acc += crate::f16::f16_to_f32(w_row[i]) * input[i];
            }
            output[o] = acc;
        }
    }
}

/// Panel width of the fp16 GEMM: how many weight rows are dequantised
/// into the f32 scratch panel at a time. 32 rows × k ≤ 2048 is ≤ 256 KB,
/// which stays L2-resident on typical mobile cores while a 4-row block
/// of `a` stays L1-hot across the whole panel.
const MM_F16_PANEL: usize = 32;

/// Batched fp16 GEMM `out[m × n] = a[m × k] · Wᵀ`. Each fp16 weight row
/// is read from memory once and dequantised once (exactly — no value
/// changes) into a panel of `MM_F16_PANEL` f32 rows; the loop nest is
/// panel → 4-row block of `a` → weight row within the panel.
///
/// Numerics: per `(i, j)` dot the accumulator structure is verbatim
/// [`matmul_mv_f16_neon`] (4 accumulators × 16-wide chunks,
/// `vaddvq((a0+a1)+(a2+a3))`, scalar tail after the reduce), so the
/// result is bit-equal to `m` GEMV calls.
pub fn matmul_mm_f16_neon(
    a: &[f32],
    weight: &[u16],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    let chunks16 = k / 16;
    let rem_start = chunks16 * 16;
    let m4 = m / 4 * 4;
    // One allocation per call; the op runs once per layer during prefill,
    // not inside the per-frame loop.
    let mut wf = vec![0.0f32; MM_F16_PANEL.min(n) * k];
    let mut jp = 0;
    while jp < n {
        let jb = MM_F16_PANEL.min(n - jp);
        unsafe {
            let quads = jb * k / 4;
            let w_panel = weight.as_ptr().add(jp * k);
            for q in 0..quads {
                let h = vld1_u16(w_panel.add(q * 4));
                vst1q_f32(wf.as_mut_ptr().add(q * 4), f16x4_to_f32x4(h));
            }
            for p in quads * 4..jb * k {
                wf[p] = crate::f16::f16_to_f32(*w_panel.add(p));
            }
        }
        let mut i = 0;
        while i < m4 {
            for jj in 0..jb {
                let j = jp + jj;
                unsafe {
                    let a0 = a.as_ptr().add(i * k);
                    let a1 = a.as_ptr().add((i + 1) * k);
                    let a2 = a.as_ptr().add((i + 2) * k);
                    let a3 = a.as_ptr().add((i + 3) * k);
                    let wp = wf.as_ptr().add(jj * k);
                    let mut r0a = vdupq_n_f32(0.0);
                    let mut r0b = vdupq_n_f32(0.0);
                    let mut r0c = vdupq_n_f32(0.0);
                    let mut r0d = vdupq_n_f32(0.0);
                    let mut r1a = vdupq_n_f32(0.0);
                    let mut r1b = vdupq_n_f32(0.0);
                    let mut r1c = vdupq_n_f32(0.0);
                    let mut r1d = vdupq_n_f32(0.0);
                    let mut r2a = vdupq_n_f32(0.0);
                    let mut r2b = vdupq_n_f32(0.0);
                    let mut r2c = vdupq_n_f32(0.0);
                    let mut r2d = vdupq_n_f32(0.0);
                    let mut r3a = vdupq_n_f32(0.0);
                    let mut r3b = vdupq_n_f32(0.0);
                    let mut r3c = vdupq_n_f32(0.0);
                    let mut r3d = vdupq_n_f32(0.0);
                    for c in 0..chunks16 {
                        let b = c * 16;
                        let w0 = vld1q_f32(wp.add(b));
                        let w1 = vld1q_f32(wp.add(b + 4));
                        let w2 = vld1q_f32(wp.add(b + 8));
                        let w3 = vld1q_f32(wp.add(b + 12));
                        r0a = vfmaq_f32(r0a, vld1q_f32(a0.add(b)), w0);
                        r0b = vfmaq_f32(r0b, vld1q_f32(a0.add(b + 4)), w1);
                        r0c = vfmaq_f32(r0c, vld1q_f32(a0.add(b + 8)), w2);
                        r0d = vfmaq_f32(r0d, vld1q_f32(a0.add(b + 12)), w3);
                        r1a = vfmaq_f32(r1a, vld1q_f32(a1.add(b)), w0);
                        r1b = vfmaq_f32(r1b, vld1q_f32(a1.add(b + 4)), w1);
                        r1c = vfmaq_f32(r1c, vld1q_f32(a1.add(b + 8)), w2);
                        r1d = vfmaq_f32(r1d, vld1q_f32(a1.add(b + 12)), w3);
                        r2a = vfmaq_f32(r2a, vld1q_f32(a2.add(b)), w0);
                        r2b = vfmaq_f32(r2b, vld1q_f32(a2.add(b + 4)), w1);
                        r2c = vfmaq_f32(r2c, vld1q_f32(a2.add(b + 8)), w2);
                        r2d = vfmaq_f32(r2d, vld1q_f32(a2.add(b + 12)), w3);
                        r3a = vfmaq_f32(r3a, vld1q_f32(a3.add(b)), w0);
                        r3b = vfmaq_f32(r3b, vld1q_f32(a3.add(b + 4)), w1);
                        r3c = vfmaq_f32(r3c, vld1q_f32(a3.add(b + 8)), w2);
                        r3d = vfmaq_f32(r3d, vld1q_f32(a3.add(b + 12)), w3);
                    }
                    let mut s0 = vaddvq_f32(vaddq_f32(vaddq_f32(r0a, r0b), vaddq_f32(r0c, r0d)));
                    let mut s1 = vaddvq_f32(vaddq_f32(vaddq_f32(r1a, r1b), vaddq_f32(r1c, r1d)));
                    let mut s2 = vaddvq_f32(vaddq_f32(vaddq_f32(r2a, r2b), vaddq_f32(r2c, r2d)));
                    let mut s3 = vaddvq_f32(vaddq_f32(vaddq_f32(r3a, r3b), vaddq_f32(r3c, r3d)));
                    for p in rem_start..k {
                        let w = *wp.add(p);
                        s0 += w * *a0.add(p);
                        s1 += w * *a1.add(p);
                        s2 += w * *a2.add(p);
                        s3 += w * *a3.add(p);
                    }
                    out[i * n + j] = s0;
                    out[(i + 1) * n + j] = s1;
                    out[(i + 2) * n + j] = s2;
                    out[(i + 3) * n + j] = s3;
                }
            }
            i += 4;
        }
        // Remaining m % 4 rows, one at a time — same skeleton as the GEMV.
        while i < m {
            let a_row = &a[i * k..(i + 1) * k];
            for jj in 0..jb {
                let j = jp + jj;
                unsafe {
                    let wp = wf.as_ptr().add(jj * k);
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);
                    for c in 0..chunks16 {
                        let b = c * 16;
                        acc0 =
                            vfmaq_f32(acc0, vld1q_f32(a_row.as_ptr().add(b)), vld1q_f32(wp.add(b)));
                        acc1 = vfmaq_f32(
                            acc1,
                            vld1q_f32(a_row.as_ptr().add(b + 4)),
                            vld1q_f32(wp.add(b + 4)),
                        );
                        acc2 = vfmaq_f32(
                            acc2,
                            vld1q_f32(a_row.as_ptr().add(b + 8)),
                            vld1q_f32(wp.add(b + 8)),
                        );
                        acc3 = vfmaq_f32(
                            acc3,
                            vld1q_f32(a_row.as_ptr().add(b + 12)),
                            vld1q_f32(wp.add(b + 12)),
                        );
                    }
                    let mut acc =
                        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
                    for p in rem_start..k {
                        acc += *wp.add(p) * a_row[p];
                    }
                    out[i * n + j] = acc;
                }
            }
            i += 1;
        }
        jp += jb;
    }
}

// ───────── int8-resident matrix products ─────────

/// Weight-only int8 GEMV with per-row scales: int8 → f32 widening in
/// register (`vmovl` ×2 + `vcvtq_f32_s32`), 4 accumulators over 16-wide
/// chunks, scalar tail, scale applied once after the reduction.
pub fn matmul_mv_i8_neon(
    input: &[f32],
    weight: &[i8],
    scale: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    let chunks16 = in_features / 16;
    let rem_start = chunks16 * 16;
    for o in 0..out_features {
        let row = &weight[o * in_features..(o + 1) * in_features];
        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            for c in 0..chunks16 {
                let b = c * 16;
                let w8 = vld1q_s8(row.as_ptr().add(b));
                let lo16 = vmovl_s8(vget_low_s8(w8));
                let hi16 = vmovl_s8(vget_high_s8(w8));
                let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                acc0 = vfmaq_f32(acc0, vld1q_f32(input.as_ptr().add(b)), w0);
                acc1 = vfmaq_f32(acc1, vld1q_f32(input.as_ptr().add(b + 4)), w1);
                acc2 = vfmaq_f32(acc2, vld1q_f32(input.as_ptr().add(b + 8)), w2);
                acc3 = vfmaq_f32(acc3, vld1q_f32(input.as_ptr().add(b + 12)), w3);
            }
            let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            for i in rem_start..in_features {
                acc += row[i] as f32 * input[i];
            }
            output[o] = acc * scale[o];
        }
    }
}

/// Panel width of the int8 GEMM (same rationale as [`MM_F16_PANEL`]).
const MM_I8_PANEL: usize = 32;

/// Batched int8 GEMM `out[m × n] = a[m × k] · Wᵀ` with per-row scales.
/// Each int8 weight row is read once and widened once (exactly) into a
/// panel of `MM_I8_PANEL` f32 rows, then the same panel → 4-row block →
/// weight-row nest as [`matmul_mm_f16_neon`].
///
/// Numerics: per `(i, j)` dot is verbatim [`matmul_mv_i8_neon`]
/// (4 accumulators × 16-wide chunks, `vaddvq((a0+a1)+(a2+a3))`, scalar
/// tail *before* the scale) and `scale[j]` is applied once per output,
/// so the result is bit-equal to `m` GEMV calls.
pub fn matmul_mm_i8_neon(
    a: &[f32],
    w: &[i8],
    scale: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    let chunks16 = k / 16;
    let rem_start = chunks16 * 16;
    let m4 = m / 4 * 4;
    let mut wf = vec![0.0f32; MM_I8_PANEL.min(n) * k];
    let mut jp = 0;
    while jp < n {
        let jb = MM_I8_PANEL.min(n - jp);
        unsafe {
            // Widen the panel flat over jb*k elements; 16-wide chunks may
            // straddle row boundaries, which is harmless because the
            // int8 → f32 conversion is exact and position-independent.
            let total = jb * k;
            let c16 = total / 16;
            let w_panel = w.as_ptr().add(jp * k);
            for q in 0..c16 {
                let w8 = vld1q_s8(w_panel.add(q * 16));
                let lo16 = vmovl_s8(vget_low_s8(w8));
                let hi16 = vmovl_s8(vget_high_s8(w8));
                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(wf.as_mut_ptr().add(q * 16), f0);
                vst1q_f32(wf.as_mut_ptr().add(q * 16 + 4), f1);
                vst1q_f32(wf.as_mut_ptr().add(q * 16 + 8), f2);
                vst1q_f32(wf.as_mut_ptr().add(q * 16 + 12), f3);
            }
            for p in c16 * 16..total {
                wf[p] = *w_panel.add(p) as f32;
            }
        }
        let mut i = 0;
        while i < m4 {
            for jj in 0..jb {
                let j = jp + jj;
                let s = scale[j];
                unsafe {
                    let a0 = a.as_ptr().add(i * k);
                    let a1 = a.as_ptr().add((i + 1) * k);
                    let a2 = a.as_ptr().add((i + 2) * k);
                    let a3 = a.as_ptr().add((i + 3) * k);
                    let wp = wf.as_ptr().add(jj * k);
                    let mut r0a = vdupq_n_f32(0.0);
                    let mut r0b = vdupq_n_f32(0.0);
                    let mut r0c = vdupq_n_f32(0.0);
                    let mut r0d = vdupq_n_f32(0.0);
                    let mut r1a = vdupq_n_f32(0.0);
                    let mut r1b = vdupq_n_f32(0.0);
                    let mut r1c = vdupq_n_f32(0.0);
                    let mut r1d = vdupq_n_f32(0.0);
                    let mut r2a = vdupq_n_f32(0.0);
                    let mut r2b = vdupq_n_f32(0.0);
                    let mut r2c = vdupq_n_f32(0.0);
                    let mut r2d = vdupq_n_f32(0.0);
                    let mut r3a = vdupq_n_f32(0.0);
                    let mut r3b = vdupq_n_f32(0.0);
                    let mut r3c = vdupq_n_f32(0.0);
                    let mut r3d = vdupq_n_f32(0.0);
                    for c in 0..chunks16 {
                        let b = c * 16;
                        let w0 = vld1q_f32(wp.add(b));
                        let w1 = vld1q_f32(wp.add(b + 4));
                        let w2 = vld1q_f32(wp.add(b + 8));
                        let w3 = vld1q_f32(wp.add(b + 12));
                        r0a = vfmaq_f32(r0a, vld1q_f32(a0.add(b)), w0);
                        r0b = vfmaq_f32(r0b, vld1q_f32(a0.add(b + 4)), w1);
                        r0c = vfmaq_f32(r0c, vld1q_f32(a0.add(b + 8)), w2);
                        r0d = vfmaq_f32(r0d, vld1q_f32(a0.add(b + 12)), w3);
                        r1a = vfmaq_f32(r1a, vld1q_f32(a1.add(b)), w0);
                        r1b = vfmaq_f32(r1b, vld1q_f32(a1.add(b + 4)), w1);
                        r1c = vfmaq_f32(r1c, vld1q_f32(a1.add(b + 8)), w2);
                        r1d = vfmaq_f32(r1d, vld1q_f32(a1.add(b + 12)), w3);
                        r2a = vfmaq_f32(r2a, vld1q_f32(a2.add(b)), w0);
                        r2b = vfmaq_f32(r2b, vld1q_f32(a2.add(b + 4)), w1);
                        r2c = vfmaq_f32(r2c, vld1q_f32(a2.add(b + 8)), w2);
                        r2d = vfmaq_f32(r2d, vld1q_f32(a2.add(b + 12)), w3);
                        r3a = vfmaq_f32(r3a, vld1q_f32(a3.add(b)), w0);
                        r3b = vfmaq_f32(r3b, vld1q_f32(a3.add(b + 4)), w1);
                        r3c = vfmaq_f32(r3c, vld1q_f32(a3.add(b + 8)), w2);
                        r3d = vfmaq_f32(r3d, vld1q_f32(a3.add(b + 12)), w3);
                    }
                    let mut s0 = vaddvq_f32(vaddq_f32(vaddq_f32(r0a, r0b), vaddq_f32(r0c, r0d)));
                    let mut s1 = vaddvq_f32(vaddq_f32(vaddq_f32(r1a, r1b), vaddq_f32(r1c, r1d)));
                    let mut s2 = vaddvq_f32(vaddq_f32(vaddq_f32(r2a, r2b), vaddq_f32(r2c, r2d)));
                    let mut s3 = vaddvq_f32(vaddq_f32(vaddq_f32(r3a, r3b), vaddq_f32(r3c, r3d)));
                    for p in rem_start..k {
                        let wv = *wp.add(p);
                        s0 += wv * *a0.add(p);
                        s1 += wv * *a1.add(p);
                        s2 += wv * *a2.add(p);
                        s3 += wv * *a3.add(p);
                    }
                    out[i * n + j] = s0 * s;
                    out[(i + 1) * n + j] = s1 * s;
                    out[(i + 2) * n + j] = s2 * s;
                    out[(i + 3) * n + j] = s3 * s;
                }
            }
            i += 4;
        }
        // Remaining m % 4 rows, one at a time — same skeleton as the GEMV.
        while i < m {
            let a_row = &a[i * k..(i + 1) * k];
            for jj in 0..jb {
                let j = jp + jj;
                let s = scale[j];
                unsafe {
                    let wp = wf.as_ptr().add(jj * k);
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    let mut acc2 = vdupq_n_f32(0.0);
                    let mut acc3 = vdupq_n_f32(0.0);
                    for c in 0..chunks16 {
                        let b = c * 16;
                        acc0 =
                            vfmaq_f32(acc0, vld1q_f32(a_row.as_ptr().add(b)), vld1q_f32(wp.add(b)));
                        acc1 = vfmaq_f32(
                            acc1,
                            vld1q_f32(a_row.as_ptr().add(b + 4)),
                            vld1q_f32(wp.add(b + 4)),
                        );
                        acc2 = vfmaq_f32(
                            acc2,
                            vld1q_f32(a_row.as_ptr().add(b + 8)),
                            vld1q_f32(wp.add(b + 8)),
                        );
                        acc3 = vfmaq_f32(
                            acc3,
                            vld1q_f32(a_row.as_ptr().add(b + 12)),
                            vld1q_f32(wp.add(b + 12)),
                        );
                    }
                    let mut acc =
                        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
                    for p in rem_start..k {
                        acc += *wp.add(p) * a_row[p];
                    }
                    out[i * n + j] = acc * s;
                }
            }
            i += 1;
        }
        jp += jb;
    }
}

// ───────── convolution ─────────

/// Depthwise Conv1D over a channels-last `[frames × channels]` buffer,
/// dilation 1. Vectorised across channels (contiguous loads of the
/// input frame and of the tap-major weights) with a sequential FMA
/// accumulator over taps, so the summation order matches the scalar
/// reference and a batched call equals the stepwise sliding-window
/// result bit for bit.
pub fn conv1d_depthwise_nlc_neon(
    input: &[f32],
    weight_kmajor: &[f32],
    bias: &[f32],
    channels: usize,
    kernel: usize,
    frames_in: usize,
    output: &mut [f32],
) {
    let frames_out = frames_in - (kernel - 1);
    debug_assert_eq!(input.len(), frames_in * channels);
    debug_assert_eq!(weight_kmajor.len(), kernel * channels);
    debug_assert_eq!(bias.len(), channels);
    debug_assert_eq!(output.len(), frames_out * channels);

    let chunks4 = channels / 4;
    let rem_start = chunks4 * 4;
    for t in 0..frames_out {
        let dst = &mut output[t * channels..(t + 1) * channels];
        for cb in 0..chunks4 {
            let c = cb * 4;
            unsafe {
                let mut acc = vld1q_f32(bias.as_ptr().add(c));
                for k in 0..kernel {
                    let w = vld1q_f32(weight_kmajor.as_ptr().add(k * channels + c));
                    let x = vld1q_f32(input.as_ptr().add((t + k) * channels + c));
                    acc = vfmaq_f32(acc, w, x);
                }
                vst1q_f32(dst.as_mut_ptr().add(c), acc);
            }
        }
        for c in rem_start..channels {
            let mut acc = bias[c];
            for k in 0..kernel {
                acc += weight_kmajor[k * channels + c] * input[(t + k) * channels + c];
            }
            dst[c] = acc;
        }
    }
}
