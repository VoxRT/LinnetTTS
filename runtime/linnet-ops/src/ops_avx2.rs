//! x86_64 AVX2 + FMA kernels.
//!
//! Same algorithms as the aarch64 NEON module with the lane width
//! doubled (8 × f32): the outer chunking goes 16 → 32 and the mid-tail
//! chunking 4 → 8, with a scalar remainder after that.
//!
//! Compiled only on `target_arch = "x86_64"`. Every entry point is an
//! `unsafe fn` gated with `#[target_feature]`; the dispatchers in
//! [`crate::ops`] check `is_x86_feature_detected!("avx2")` /
//! `("fma")` (and `("f16c")` for the fp16 kernels) before calling in,
//! so CPUs without those extensions fall through to the scalar
//! reference.

#![cfg(target_arch = "x86_64")]
// Every function here is `#[target_feature]`-gated `unsafe fn` whose body
// is intrinsic calls; a block per call adds nothing.
#![allow(unsafe_op_in_unsafe_fn)]
// The exp polynomial constants below are written out in full so they
// stay bit-identical to the reference implementation.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use core::arch::x86_64::*;

/// Horizontal sum of the 8 lanes of a `__m256`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    // [a,b,c,d] → [a+b, ., c+d, .] → [a+b+c+d, ., ., .]
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// AVX2 + FMA dense mat-vec `output[o] = Σ_k weight[o, k] · input[k]`.
/// 4 × 8-lane accumulators over 32-wide chunks, 8-wide mid tail,
/// scalar remainder.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_mv_avx2(
    input: &[f32],
    weight: &[f32],
    in_features: usize,
    out_features: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_features);
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert_eq!(output.len(), out_features);

    let chunks32 = in_features / 32;
    let chunks8_after = (in_features % 32) / 8;
    let rem_start = chunks32 * 32 + chunks8_after * 8;

    for o in 0..out_features {
        let w_row = &weight[o * in_features..(o + 1) * in_features];

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for c in 0..chunks32 {
            let b = c * 32;
            let x0 = _mm256_loadu_ps(input.as_ptr().add(b));
            let x1 = _mm256_loadu_ps(input.as_ptr().add(b + 8));
            let x2 = _mm256_loadu_ps(input.as_ptr().add(b + 16));
            let x3 = _mm256_loadu_ps(input.as_ptr().add(b + 24));
            let w0 = _mm256_loadu_ps(w_row.as_ptr().add(b));
            let w1 = _mm256_loadu_ps(w_row.as_ptr().add(b + 8));
            let w2 = _mm256_loadu_ps(w_row.as_ptr().add(b + 16));
            let w3 = _mm256_loadu_ps(w_row.as_ptr().add(b + 24));
            acc0 = _mm256_fmadd_ps(x0, w0, acc0);
            acc1 = _mm256_fmadd_ps(x1, w1, acc1);
            acc2 = _mm256_fmadd_ps(x2, w2, acc2);
            acc3 = _mm256_fmadd_ps(x3, w3, acc3);
        }

        let after32 = chunks32 * 32;
        for k in 0..chunks8_after {
            let b = after32 + k * 8;
            let xv = _mm256_loadu_ps(input.as_ptr().add(b));
            let wv = _mm256_loadu_ps(w_row.as_ptr().add(b));
            acc0 = _mm256_fmadd_ps(xv, wv, acc0);
        }

        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let mut acc = hsum256(_mm256_add_ps(sum01, sum23));

        for j in rem_start..in_features {
            acc += input[j] * w_row[j];
        }
        output[o] = acc;
    }
}

/// AVX2 LayerNorm for a single frame (`channels` contiguous values).
/// Three passes: mean → variance → apply, 8 lanes at a time.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_norm_frames1_avx2(
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

    let chunks8 = channels / 8;
    let rem_start = chunks8 * 8;

    // Pass 1: sum.
    let mut sum_v = _mm256_setzero_ps();
    for c in 0..chunks8 {
        sum_v = _mm256_add_ps(sum_v, _mm256_loadu_ps(input.as_ptr().add(c * 8)));
    }
    let mut sum = hsum256(sum_v);
    for j in rem_start..channels {
        sum += input[j];
    }
    let inv_c = 1.0_f32 / channels as f32;
    let mean = sum * inv_c;

    // Pass 2: sum of squared deviations.
    let mean_v = _mm256_set1_ps(mean);
    let mut var_v = _mm256_setzero_ps();
    for c in 0..chunks8 {
        let x = _mm256_loadu_ps(input.as_ptr().add(c * 8));
        let d = _mm256_sub_ps(x, mean_v);
        var_v = _mm256_fmadd_ps(d, d, var_v);
    }
    let mut var = hsum256(var_v);
    for j in rem_start..channels {
        let d = input[j] - mean;
        var += d * d;
    }
    var *= inv_c;

    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let inv_std_v = _mm256_set1_ps(inv_std);

    // Pass 3: gamma * (x - μ) * inv_std + beta.
    for c in 0..chunks8 {
        let x = _mm256_loadu_ps(input.as_ptr().add(c * 8));
        let g = _mm256_loadu_ps(gamma.as_ptr().add(c * 8));
        let b_ = _mm256_loadu_ps(beta.as_ptr().add(c * 8));
        let centred = _mm256_sub_ps(x, mean_v);
        let scaled = _mm256_mul_ps(centred, inv_std_v);
        let y = _mm256_fmadd_ps(g, scaled, b_);
        _mm256_storeu_ps(output.as_mut_ptr().add(c * 8), y);
    }
    for j in rem_start..channels {
        output[j] = gamma[j] * (input[j] - mean) * inv_std + beta[j];
    }
}

/// AVX2 in-place `a[i] += b[i]`. Bit-exact vs scalar (one rounding per
/// element).
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn add_inplace_avx2(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let chunks8 = n / 8;
    let rem_start = chunks8 * 8;
    for c in 0..chunks8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(c * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(c * 8));
        _mm256_storeu_ps(a.as_mut_ptr().add(c * 8), _mm256_add_ps(av, bv));
    }
    for j in rem_start..n {
        a[j] += b[j];
    }
}

/// AVX2 in-place `a[i] += scale · b[i]` via `_mm256_fmadd_ps`. The fused
/// multiply-add rounds once where the scalar reference rounds twice, so
/// results may differ by ≤ 1 ULP per element.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_scaled_inplace_avx2(a: &mut [f32], b: &[f32], scale: f32) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let chunks8 = n / 8;
    let rem_start = chunks8 * 8;
    let sv = _mm256_set1_ps(scale);
    for c in 0..chunks8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(c * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(c * 8));
        _mm256_storeu_ps(a.as_mut_ptr().add(c * 8), _mm256_fmadd_ps(bv, sv, av));
    }
    for j in rem_start..n {
        a[j] += scale * b[j];
    }
}

/// AVX2 dot product `Σ a[i] · b[i]`: one 8-lane FMA accumulator,
/// horizontal reduce, scalar tail. Reordered summation → a few ULP vs
/// scalar.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks8 = n / 8;
    let rem_start = chunks8 * 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks8 {
        let av = _mm256_loadu_ps(a.as_ptr().add(c * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(c * 8));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    let mut s = hsum256(acc);
    for i in rem_start..n {
        s += a[i] * b[i];
    }
    s
}

/// AVX2 dense matmul `out = a · b`. 8-column blocking: broadcast
/// `a[i, p]`, load `b[p, j..j+8]`, FMA-accumulate per lane. Each lane
/// runs the same `p` sequence as the scalar reference (FMA single
/// rounding vs scalar mul+add is the only difference).
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_mm_avx2(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    let n_chunks8 = n / 8;
    let n_rem_start = n_chunks8 * 8;

    for i in 0..m {
        let a_row_off = i * k;
        let out_row_off = i * n;

        for cc in 0..n_chunks8 {
            let j0 = cc * 8;
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let av = _mm256_set1_ps(*a.get_unchecked(a_row_off + p));
                let bv = _mm256_loadu_ps(b.as_ptr().add(p * n + j0));
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(out_row_off + j0), acc);
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

/// AVX2 `out = a · bᵀ`. 32-wide k-loop with 4 FMA accumulators and a
/// horizontal sum per output cell. Reordered summation → a few ULP vs
/// scalar.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_mm_transposed_b_avx2(
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

    let chunks32 = k / 32;
    let chunks8_after = (k % 32) / 8;
    let rem_start = chunks32 * 32 + chunks8_after * 8;

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row_off = i * n;
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            for c in 0..chunks32 {
                let base = c * 32;
                let a0 = _mm256_loadu_ps(a_row.as_ptr().add(base));
                let a1 = _mm256_loadu_ps(a_row.as_ptr().add(base + 8));
                let a2 = _mm256_loadu_ps(a_row.as_ptr().add(base + 16));
                let a3 = _mm256_loadu_ps(a_row.as_ptr().add(base + 24));
                let b0 = _mm256_loadu_ps(b_row.as_ptr().add(base));
                let b1 = _mm256_loadu_ps(b_row.as_ptr().add(base + 8));
                let b2 = _mm256_loadu_ps(b_row.as_ptr().add(base + 16));
                let b3 = _mm256_loadu_ps(b_row.as_ptr().add(base + 24));
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                acc3 = _mm256_fmadd_ps(a3, b3, acc3);
            }

            let after32 = chunks32 * 32;
            for cc in 0..chunks8_after {
                let base = after32 + cc * 8;
                let av = _mm256_loadu_ps(a_row.as_ptr().add(base));
                let bv = _mm256_loadu_ps(b_row.as_ptr().add(base));
                acc0 = _mm256_fmadd_ps(av, bv, acc0);
            }

            let sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut acc = hsum256(sum);
            for p in rem_start..k {
                acc += a_row[p] * b_row[p];
            }
            out[out_row_off + j] = acc;
        }
    }
}

// ───────── exp / sigmoid — 8-wide polynomial approximations ─────────
//
// Cody–Waite range reduction + degree-7 Horner polynomial + IEEE-754
// exponent-bias trick for 2^n. Error ≤ 1e-6 abs vs libm `expf` on the
// sigmoid / softmax input range.

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

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn expf_avx2_8(x: __m256) -> __m256 {
    let x = _mm256_min_ps(x, _mm256_set1_ps(EXP_CLAMP));
    let x = _mm256_max_ps(x, _mm256_set1_ps(-EXP_CLAMP));
    let n_f = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(EXP_LOG2_E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let n_i = _mm256_cvtps_epi32(n_f);
    let r = _mm256_fnmadd_ps(n_f, _mm256_set1_ps(EXP_LN2_HI), x);
    let r = _mm256_fnmadd_ps(n_f, _mm256_set1_ps(EXP_LN2_LO), r);
    let p = _mm256_fmadd_ps(_mm256_set1_ps(EXP_C7), r, _mm256_set1_ps(EXP_C6));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C5));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C4));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C3));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C2));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C1));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP_C0));
    let biased = _mm256_add_epi32(n_i, _mm256_set1_epi32(127));
    let pow2_bits = _mm256_slli_epi32::<23>(biased);
    let pow2 = _mm256_castsi256_ps(pow2_bits);
    _mm256_mul_ps(p, pow2)
}

/// 8-wide sigmoid, numerically stable on both signs:
/// `y = exp(-|x|); x ≥ 0 ? 1/(1+y) : y/(1+y)`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sigmoid_avx2_8(x: __m256) -> __m256 {
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let abs_x = _mm256_and_ps(x, sign_mask);
    let neg_abs = _mm256_sub_ps(_mm256_setzero_ps(), abs_x);
    let y = expf_avx2_8(neg_abs);
    let one = _mm256_set1_ps(1.0);
    let denom = _mm256_add_ps(one, y);
    let inv_denom = _mm256_div_ps(one, denom);
    let sig_pos = inv_denom;
    let sig_neg = _mm256_mul_ps(y, inv_denom);
    let mask = _mm256_cmp_ps::<{ _CMP_GE_OQ }>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(sig_neg, sig_pos, mask)
}

/// AVX2 in-place SiLU: `x[i] *= sigmoid(x[i])`. Scalar (libm) tail.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_inplace_avx2(buf: &mut [f32]) {
    let n = buf.len();
    let chunks8 = n / 8;
    let rem_start = chunks8 * 8;
    for c in 0..chunks8 {
        let ptr = buf.as_mut_ptr().add(c * 8);
        let v = _mm256_loadu_ps(ptr);
        let s = sigmoid_avx2_8(v);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(v, s));
    }
    for i in rem_start..n {
        let x = buf[i];
        let s = 1.0_f32 / (1.0 + (-x).exp());
        buf[i] = x * s;
    }
}

/// AVX2 in-place row-wise softmax. Per row: 8-wide max + horizontal
/// reduce, 8-wide `exp(v - m)` with vector sum, 8-wide multiply by
/// `1/sum`. Rows with a non-finite max become all zeros (matches the
/// scalar reference).
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_row_inplace_avx2(scores: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    let chunks8 = cols / 8;
    let rem_start = chunks8 * 8;

    for r in 0..rows {
        let row = &mut scores[r * cols..(r + 1) * cols];

        // 1. Row max.
        let mut m = f32::NEG_INFINITY;
        if chunks8 > 0 {
            let mut mv = _mm256_loadu_ps(row.as_ptr());
            for c in 1..chunks8 {
                let v = _mm256_loadu_ps(row.as_ptr().add(c * 8));
                mv = _mm256_max_ps(mv, v);
            }
            let lo = _mm256_castps256_ps128(mv);
            let hi = _mm256_extractf128_ps(mv, 1);
            let max128 = _mm_max_ps(lo, hi);
            let shuf = _mm_movehdup_ps(max128);
            let maxs = _mm_max_ps(max128, shuf);
            let shuf2 = _mm_movehl_ps(maxs, maxs);
            let final_ = _mm_max_ss(maxs, shuf2);
            let m_chunks = _mm_cvtss_f32(final_);
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
        let mv = _mm256_set1_ps(m);
        let mut sum_v = _mm256_setzero_ps();
        for c in 0..chunks8 {
            let ptr = row.as_mut_ptr().add(c * 8);
            let v = _mm256_loadu_ps(ptr);
            let e = expf_avx2_8(_mm256_sub_ps(v, mv));
            sum_v = _mm256_add_ps(sum_v, e);
            _mm256_storeu_ps(ptr, e);
        }
        let mut sum = hsum256(sum_v);
        for j in rem_start..cols {
            let e = (row[j] - m).exp();
            row[j] = e;
            sum += e;
        }

        // 3. Divide by sum.
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let inv_v = _mm256_set1_ps(inv);
        for c in 0..chunks8 {
            let ptr = row.as_mut_ptr().add(c * 8);
            let v = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, _mm256_mul_ps(v, inv_v));
        }
        for j in rem_start..cols {
            row[j] *= inv;
        }
    }
}

// ───────── fp16-resident matrix products (F16C + FMA) ─────────

/// Horizontal sum of two accumulators as used by the fp16 kernels. The
/// reduction order (`_mm_hadd_ps` twice) differs from [`hsum256`] and is
/// part of the numerical contract shared by [`matmul_mv_f16_avx2`] and
/// [`matmul_mm_f16_avx2`].
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f16_pair(acc0: __m256, acc1: __m256) -> f32 {
    let s = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(s, 1);
    let lo = _mm256_castps256_ps128(s);
    let q = _mm_add_ps(lo, hi);
    let q = _mm_hadd_ps(q, q);
    let q = _mm_hadd_ps(q, q);
    _mm_cvtss_f32(q)
}

/// fp16-resident GEMV: weights converted in-register with F16C
/// (`_mm256_cvtph_ps`), fp32 FMA accumulation in 2 accumulators over
/// 16-wide chunks, scalar tail.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn matmul_mv_f16_avx2(
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
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for c in 0..chunks16 {
            let b = c * 16;
            let h0 = _mm_loadu_si128(w_row.as_ptr().add(b) as *const __m128i);
            let h1 = _mm_loadu_si128(w_row.as_ptr().add(b + 8) as *const __m128i);
            let w0 = _mm256_cvtph_ps(h0);
            let w1 = _mm256_cvtph_ps(h1);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(input.as_ptr().add(b)), w0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(input.as_ptr().add(b + 8)), w1, acc1);
        }
        let mut acc = hsum_f16_pair(acc0, acc1);
        for i in rem_start..in_features {
            acc += crate::f16::f16_to_f32(w_row[i]) * input[i];
        }
        output[o] = acc;
    }
}

/// Batched fp16 GEMM `out[m × n] = a · Wᵀ`, `W` fp16 row-major `[n × k]`.
/// Weight-row outer loop with a 4-row block over `a`: each converted
/// weight chunk feeds four FMAs. The per-dot accumulator structure is
/// verbatim [`matmul_mv_f16_avx2`] (2 accumulators × 16-wide chunks,
/// same reduction, scalar tail after it) — bit-equal to `m` GEMV calls.
///
/// # Safety
/// The CPU must support AVX2, FMA and F16C.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn matmul_mm_f16_avx2(
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

    for j in 0..n {
        let w_row = &weight[j * k..(j + 1) * k];
        let mut i = 0;
        while i < m4 {
            let a0 = a.as_ptr().add(i * k);
            let a1 = a.as_ptr().add((i + 1) * k);
            let a2 = a.as_ptr().add((i + 2) * k);
            let a3 = a.as_ptr().add((i + 3) * k);
            let mut r0a = _mm256_setzero_ps();
            let mut r0b = _mm256_setzero_ps();
            let mut r1a = _mm256_setzero_ps();
            let mut r1b = _mm256_setzero_ps();
            let mut r2a = _mm256_setzero_ps();
            let mut r2b = _mm256_setzero_ps();
            let mut r3a = _mm256_setzero_ps();
            let mut r3b = _mm256_setzero_ps();
            for c in 0..chunks16 {
                let b = c * 16;
                let h0 = _mm_loadu_si128(w_row.as_ptr().add(b) as *const __m128i);
                let h1 = _mm_loadu_si128(w_row.as_ptr().add(b + 8) as *const __m128i);
                let w0 = _mm256_cvtph_ps(h0);
                let w1 = _mm256_cvtph_ps(h1);
                r0a = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(b)), w0, r0a);
                r0b = _mm256_fmadd_ps(_mm256_loadu_ps(a0.add(b + 8)), w1, r0b);
                r1a = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(b)), w0, r1a);
                r1b = _mm256_fmadd_ps(_mm256_loadu_ps(a1.add(b + 8)), w1, r1b);
                r2a = _mm256_fmadd_ps(_mm256_loadu_ps(a2.add(b)), w0, r2a);
                r2b = _mm256_fmadd_ps(_mm256_loadu_ps(a2.add(b + 8)), w1, r2b);
                r3a = _mm256_fmadd_ps(_mm256_loadu_ps(a3.add(b)), w0, r3a);
                r3b = _mm256_fmadd_ps(_mm256_loadu_ps(a3.add(b + 8)), w1, r3b);
            }
            let mut s0 = hsum_f16_pair(r0a, r0b);
            let mut s1 = hsum_f16_pair(r1a, r1b);
            let mut s2 = hsum_f16_pair(r2a, r2b);
            let mut s3 = hsum_f16_pair(r3a, r3b);
            for p in rem_start..k {
                let w = crate::f16::f16_to_f32(w_row[p]);
                s0 += w * *a0.add(p);
                s1 += w * *a1.add(p);
                s2 += w * *a2.add(p);
                s3 += w * *a3.add(p);
            }
            out[i * n + j] = s0;
            out[(i + 1) * n + j] = s1;
            out[(i + 2) * n + j] = s2;
            out[(i + 3) * n + j] = s3;
            i += 4;
        }
        // Remaining m % 4 rows, one at a time — same skeleton as the GEMV.
        while i < m {
            let a_row = &a[i * k..(i + 1) * k];
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            for c in 0..chunks16 {
                let b = c * 16;
                let h0 = _mm_loadu_si128(w_row.as_ptr().add(b) as *const __m128i);
                let h1 = _mm_loadu_si128(w_row.as_ptr().add(b + 8) as *const __m128i);
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(b)),
                    _mm256_cvtph_ps(h0),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(b + 8)),
                    _mm256_cvtph_ps(h1),
                    acc1,
                );
            }
            let mut acc = hsum_f16_pair(acc0, acc1);
            for p in rem_start..k {
                acc += crate::f16::f16_to_f32(w_row[p]) * a_row[p];
            }
            out[i * n + j] = acc;
            i += 1;
        }
    }
}
