//! WASM SIMD128 kernels.
//!
//! One-to-one ports of the aarch64 NEON kernels: same 16-wide outer +
//! 4-wide tail with 4 accumulators for the matrix products, same
//! three-pass LayerNorm. The one structural difference is that SIMD128
//! has no fused multiply-add for `f32x4`, so `vfmaq_f32(a, b, c)`
//! becomes `f32x4_add(a, f32x4_mul(b, c))` — two roundings, exactly
//! like the scalar reference, which makes the element-wise kernels
//! bit-exact against it.
//!
//! Compiled only for `target_arch = "wasm32"` with `target_feature =
//! "simd128"` (pass `-C target-feature=+simd128`).

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
// The exp polynomial constants below are written out in full so they
// stay bit-identical to the reference implementation.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use core::arch::wasm32::*;

/// Horizontal add of the 4 lanes. SIMD128 has no `vaddvq_f32`
/// equivalent, so extract-and-add; the order `(l0 + l1) + l2) + l3`
/// is part of the numerical contract.
#[inline(always)]
unsafe fn hsum_f32x4(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

/// SIMD128 LayerNorm for a single frame: three passes (sum, squared
/// deviations, apply), 4 lanes at a time, scalar remainder.
pub fn layer_norm_frames1_wasm(
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
    let mut sum_v = f32x4_splat(0.0);
    for c in 0..chunks4 {
        unsafe {
            sum_v = f32x4_add(sum_v, v128_load(input.as_ptr().add(c * 4) as *const v128));
        }
    }
    let mut sum: f32 = unsafe { hsum_f32x4(sum_v) };
    for j in rem_start..channels {
        sum += input[j];
    }
    let inv_c = 1.0_f32 / channels as f32;
    let mean = sum * inv_c;

    // Pass 2: Σ (input[c] - μ)²
    let mean_v = f32x4_splat(mean);
    let mut var_v = f32x4_splat(0.0);
    for c in 0..chunks4 {
        unsafe {
            let x = v128_load(input.as_ptr().add(c * 4) as *const v128);
            let d = f32x4_sub(x, mean_v);
            var_v = f32x4_add(var_v, f32x4_mul(d, d));
        }
    }
    let mut var: f32 = unsafe { hsum_f32x4(var_v) };
    for j in rem_start..channels {
        let d = input[j] - mean;
        var += d * d;
    }
    var *= inv_c;

    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let inv_std_v = f32x4_splat(inv_std);

    // Pass 3: output[c] = gamma * (input - μ) * inv_std + beta
    for c in 0..chunks4 {
        unsafe {
            let x = v128_load(input.as_ptr().add(c * 4) as *const v128);
            let g = v128_load(gamma.as_ptr().add(c * 4) as *const v128);
            let b = v128_load(beta.as_ptr().add(c * 4) as *const v128);
            let centred = f32x4_sub(x, mean_v);
            let scaled = f32x4_mul(centred, inv_std_v);
            let y = f32x4_add(b, f32x4_mul(g, scaled));
            v128_store(output.as_mut_ptr().add(c * 4) as *mut v128, y);
        }
    }
    for j in rem_start..channels {
        output[j] = gamma[j] * (input[j] - mean) * inv_std + beta[j];
    }
}

/// SIMD128 dense mat-vec (no bias): 16-wide chunks with 4 accumulators,
/// 4-wide tail, scalar remainder.
pub fn matmul_mv_wasm(
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

        let (mut acc0, mut acc1, mut acc2, mut acc3) = (
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
        );

        for c in 0..chunks16 {
            let b = c * 16;
            unsafe {
                let x0 = v128_load(input.as_ptr().add(b) as *const v128);
                let x1 = v128_load(input.as_ptr().add(b + 4) as *const v128);
                let x2 = v128_load(input.as_ptr().add(b + 8) as *const v128);
                let x3 = v128_load(input.as_ptr().add(b + 12) as *const v128);
                let w0 = v128_load(w_row.as_ptr().add(b) as *const v128);
                let w1 = v128_load(w_row.as_ptr().add(b + 4) as *const v128);
                let w2 = v128_load(w_row.as_ptr().add(b + 8) as *const v128);
                let w3 = v128_load(w_row.as_ptr().add(b + 12) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(x0, w0));
                acc1 = f32x4_add(acc1, f32x4_mul(x1, w1));
                acc2 = f32x4_add(acc2, f32x4_mul(x2, w2));
                acc3 = f32x4_add(acc3, f32x4_mul(x3, w3));
            }
        }

        let after16 = chunks16 * 16;
        for k in 0..chunks4_after {
            let b = after16 + k * 4;
            unsafe {
                let xv = v128_load(input.as_ptr().add(b) as *const v128);
                let wv = v128_load(w_row.as_ptr().add(b) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(xv, wv));
            }
        }

        let acc_v = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
        let mut acc: f32 = unsafe { hsum_f32x4(acc_v) };
        for j in rem_start..in_features {
            acc += input[j] * w_row[j];
        }
        output[o] = acc;
    }
}

/// SIMD128 dot product `Σ a[i] · b[i]`: one 4-wide accumulator,
/// horizontal sum, scalar tail (a few ULP vs scalar).
pub fn dot_product_f32_wasm(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    let mut acc = f32x4_splat(0.0);
    for c in 0..chunks4 {
        unsafe {
            let av = v128_load(a.as_ptr().add(c * 4) as *const v128);
            let bv = v128_load(b.as_ptr().add(c * 4) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(av, bv));
        }
    }
    let mut s = unsafe { hsum_f32x4(acc) };
    for i in rem_start..n {
        s += a[i] * b[i];
    }
    s
}

/// SIMD128 dense matmul `out = a · b`. 4-column blocking with the same
/// per-lane `p` order as the scalar reference, and no FMA → bit-exact
/// vs [`crate::ops::matmul_mm_scalar`]. Column tail (`n % 4`) is scalar.
pub fn matmul_mm_wasm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
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
            let mut acc = f32x4_splat(0.0);
            for p in 0..k {
                unsafe {
                    let av = f32x4_splat(*a.get_unchecked(a_row_off + p));
                    let bv = v128_load(b.as_ptr().add(p * n + j0) as *const v128);
                    acc = f32x4_add(acc, f32x4_mul(av, bv));
                }
            }
            unsafe {
                v128_store(out.as_mut_ptr().add(out_row_off + j0) as *mut v128, acc);
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

/// SIMD128 `out = a · bᵀ`: 16-wide k-loop with 4 accumulators and a
/// horizontal sum per output cell (reordered summation → a few ULP vs
/// scalar).
pub fn matmul_mm_transposed_b_wasm(
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
            let (mut acc0, mut acc1, mut acc2, mut acc3) = (
                f32x4_splat(0.0),
                f32x4_splat(0.0),
                f32x4_splat(0.0),
                f32x4_splat(0.0),
            );

            for c in 0..chunks16 {
                let base = c * 16;
                unsafe {
                    let a0 = v128_load(a_row.as_ptr().add(base) as *const v128);
                    let a1 = v128_load(a_row.as_ptr().add(base + 4) as *const v128);
                    let a2 = v128_load(a_row.as_ptr().add(base + 8) as *const v128);
                    let a3 = v128_load(a_row.as_ptr().add(base + 12) as *const v128);
                    let b0 = v128_load(b_row.as_ptr().add(base) as *const v128);
                    let b1 = v128_load(b_row.as_ptr().add(base + 4) as *const v128);
                    let b2 = v128_load(b_row.as_ptr().add(base + 8) as *const v128);
                    let b3 = v128_load(b_row.as_ptr().add(base + 12) as *const v128);
                    acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
                    acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
                    acc2 = f32x4_add(acc2, f32x4_mul(a2, b2));
                    acc3 = f32x4_add(acc3, f32x4_mul(a3, b3));
                }
            }

            let after16 = chunks16 * 16;
            for cc in 0..chunks4_after {
                let base = after16 + cc * 4;
                unsafe {
                    let av = v128_load(a_row.as_ptr().add(base) as *const v128);
                    let bv = v128_load(b_row.as_ptr().add(base) as *const v128);
                    acc0 = f32x4_add(acc0, f32x4_mul(av, bv));
                }
            }

            let acc_v = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
            let mut acc: f32 = unsafe { hsum_f32x4(acc_v) };
            for p in rem_start..k {
                acc += a_row[p] * b_row[p];
            }
            out[out_row_off + j] = acc;
        }
    }
}

/// SIMD128 in-place `a[i] += b[i]`. Bit-exact vs scalar.
pub fn add_inplace_wasm(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);

    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    for c in 0..chunks4 {
        unsafe {
            let av = v128_load(a.as_ptr().add(c * 4) as *const v128);
            let bv = v128_load(b.as_ptr().add(c * 4) as *const v128);
            let s = f32x4_add(av, bv);
            v128_store(a.as_mut_ptr().add(c * 4) as *mut v128, s);
        }
    }
    for j in rem_start..n {
        a[j] += b[j];
    }
}

/// SIMD128 in-place `a[i] += scale · b[i]`. Multiply then add — the same
/// two roundings as the scalar reference, so bit-exact.
pub fn add_scaled_inplace_wasm(a: &mut [f32], b: &[f32], scale: f32) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);

    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    let sv = f32x4_splat(scale);
    for c in 0..chunks4 {
        unsafe {
            let av = v128_load(a.as_ptr().add(c * 4) as *const v128);
            let bv = v128_load(b.as_ptr().add(c * 4) as *const v128);
            let p = f32x4_mul(sv, bv);
            let r = f32x4_add(av, p);
            v128_store(a.as_mut_ptr().add(c * 4) as *mut v128, r);
        }
    }
    for j in rem_start..n {
        a[j] += scale * b[j];
    }
}

// ───────── exp / sigmoid polynomial primitives ─────────
//
// Cody–Waite range reduction + degree-7 Horner polynomial (mul + add,
// no FMA) + IEEE-754 exponent-bias trick for 2^n. Error ≤ 1e-6 abs vs
// libm `expf` on the sigmoid / softmax input range.

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
unsafe fn expf_wasm_4(x: v128) -> v128 {
    // Clamp before range reduction so `n` cannot overflow the i32 lane.
    let x = f32x4_min(x, f32x4_splat(EXP_CLAMP));
    let x = f32x4_max(x, f32x4_splat(-EXP_CLAMP));

    // n = round-to-nearest(x * log2(e)) → f32 then i32.
    let n_f = f32x4_nearest(f32x4_mul(x, f32x4_splat(EXP_LOG2_E)));
    let n_i = i32x4_trunc_sat_f32x4(n_f);

    // r = x - n * LN2_HI - n * LN2_LO  (Cody–Waite high/low split).
    let r = f32x4_sub(x, f32x4_mul(n_f, f32x4_splat(EXP_LN2_HI)));
    let r = f32x4_sub(r, f32x4_mul(n_f, f32x4_splat(EXP_LN2_LO)));

    // Horner: p = (((((c7*r + c6)*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + c0
    let p = f32x4_add(f32x4_mul(f32x4_splat(EXP_C7), r), f32x4_splat(EXP_C6));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C5));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C4));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C3));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C2));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C1));
    let p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C0));

    // 2^n via IEEE-754 bias: bits_of(2^n) = (n + 127) << 23.
    let biased = i32x4_add(n_i, i32x4_splat(127));
    let pow2 = i32x4_shl(biased, 23);
    f32x4_mul(p, pow2)
}

/// 4-wide sigmoid, numerically stable on both signs:
/// `y = exp(-|x|); x ≥ 0 ? 1/(1+y) : y/(1+y)`.
#[inline]
unsafe fn sigmoid_wasm_4(x: v128) -> v128 {
    let abs_x = v128_and(x, i32x4_splat(0x7FFF_FFFF));
    let neg_abs = f32x4_neg(abs_x);
    let y = expf_wasm_4(neg_abs);
    let one = f32x4_splat(1.0);
    let denom = f32x4_add(one, y);
    let inv_denom = f32x4_div(one, denom);
    let sig_pos = inv_denom;
    let sig_neg = f32x4_mul(y, inv_denom);
    let mask = f32x4_ge(x, f32x4_splat(0.0));
    v128_bitselect(sig_pos, sig_neg, mask)
}

/// SIMD128 in-place SiLU: `x[i] *= sigmoid(x[i])`, polynomial sigmoid
/// with a scalar (libm) tail.
pub fn silu_inplace_wasm(buf: &mut [f32]) {
    let n = buf.len();
    let chunks4 = n / 4;
    let rem_start = chunks4 * 4;
    for c in 0..chunks4 {
        unsafe {
            let ptr = buf.as_mut_ptr().add(c * 4);
            let v = v128_load(ptr as *const v128);
            let s = sigmoid_wasm_4(v);
            v128_store(ptr as *mut v128, f32x4_mul(v, s));
        }
    }
    for i in rem_start..n {
        let x = buf[i];
        let s = 1.0_f32 / (1.0 + (-x).exp());
        buf[i] = x * s;
    }
}

/// SIMD128 in-place row-wise softmax: 4-wide max, 4-wide `exp(v - m)`
/// with vector sum, 4-wide multiply by `1/sum`. Rows with a non-finite
/// max become all zeros (matches scalar).
pub fn softmax_row_inplace_wasm(scores: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    let chunks4 = cols / 4;
    let rem_start = chunks4 * 4;

    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];

        // 1. Row max.
        let mut m = f32::NEG_INFINITY;
        if chunks4 > 0 {
            let mut mv = unsafe { v128_load(row.as_ptr() as *const v128) };
            for c in 1..chunks4 {
                unsafe {
                    let v = v128_load(row.as_ptr().add(c * 4) as *const v128);
                    mv = f32x4_max(mv, v);
                }
            }
            let m_chunks = f32x4_extract_lane::<0>(mv)
                .max(f32x4_extract_lane::<1>(mv))
                .max(f32x4_extract_lane::<2>(mv))
                .max(f32x4_extract_lane::<3>(mv));
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
        let mv = f32x4_splat(m);
        let mut sum_v = f32x4_splat(0.0);
        for c in 0..chunks4 {
            unsafe {
                let ptr = row.as_mut_ptr().add(c * 4);
                let v = v128_load(ptr as *const v128);
                let e = expf_wasm_4(f32x4_sub(v, mv));
                sum_v = f32x4_add(sum_v, e);
                v128_store(ptr as *mut v128, e);
            }
        }
        let mut sum = unsafe { hsum_f32x4(sum_v) };
        for j in rem_start..cols {
            let e = (row[j] - m).exp();
            row[j] = e;
            sum += e;
        }

        // 3. Divide by sum.
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let inv_v = f32x4_splat(inv);
        for c in 0..chunks4 {
            unsafe {
                let ptr = row.as_mut_ptr().add(c * 4);
                let v = v128_load(ptr as *const v128);
                v128_store(ptr as *mut v128, f32x4_mul(v, inv_v));
            }
        }
        for j in rem_start..cols {
            row[j] *= inv;
        }
    }
}
