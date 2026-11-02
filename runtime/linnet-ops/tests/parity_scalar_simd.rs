//! Parity of every dispatcher against its scalar reference, plus the
//! structural guarantees the model code relies on (batched GEMMs equal
//! row-wise GEMVs bit for bit; batched depthwise conv equals stepwise).
//!
//! Tolerances follow the numerical contract documented on each kernel:
//! bit-exact for single-rounding element-wise ops, ≤ 1 ULP for FMA-fused
//! element-wise ops, 1e-4 abs for reductions over inputs in [-1, 1],
//! and reference-point checks for the activation functions.
//!
//! With `--features force-scalar` every dispatcher is the scalar
//! function, so these tests degenerate to self-consistency checks —
//! that is intended: the same suite runs on every target.

use linnet_ops::f16::f32_to_f16;
use linnet_ops::ops::*;

/// Deterministic LCG, uniform in [-1, 1).
fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn assert_bitexact(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: i={i} {x} vs {y}");
    }
}

/// Distance in ULPs between two finite f32 of the same sign (or zero).
fn ulp_distance(a: f32, b: f32) -> u32 {
    let ia = a.to_bits() as i32;
    let ib = b.to_bits() as i32;
    // Map to a monotonic integer line (sign-magnitude → two's complement).
    let ma = if ia < 0 { i32::MIN - ia } else { ia };
    let mb = if ib < 0 { i32::MIN - ib } else { ib };
    (ma as i64 - mb as i64).unsigned_abs() as u32
}

fn assert_within_ulps(a: &[f32], b: &[f32], ulps: u32, what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            ulp_distance(*x, *y) <= ulps,
            "{what}: i={i} {x} vs {y} ({} ULP)",
            ulp_distance(*x, *y)
        );
    }
}

/// Sizes that exercise every chunking path in the kernels: multiples of
/// 32 / 16 / 8 / 4, each with and without a scalar remainder.
const SIZES: &[usize] = &[
    1, 3, 4, 7, 8, 15, 16, 18, 24, 32, 40, 45, 64, 96, 100, 127, 128, 256, 257, 512,
];

// ───────────────────────── element-wise ─────────────────────────

#[test]
fn add_inplace_is_bitexact() {
    let mut seed = 1;
    for &n in SIZES {
        let a0 = rand_vec(n, &mut seed);
        let b = rand_vec(n, &mut seed);
        let mut a = a0.clone();
        let mut r = a0.clone();
        add_inplace(&mut a, &b);
        add_inplace_scalar(&mut r, &b);
        assert_bitexact(&a, &r, &format!("add_inplace n={n}"));
    }
}

#[test]
fn add_scaled_inplace_within_one_ulp() {
    let mut seed = 2;
    for &n in SIZES {
        let a0 = rand_vec(n, &mut seed);
        let b = rand_vec(n, &mut seed);
        let mut a = a0.clone();
        let mut r = a0.clone();
        add_scaled_inplace(&mut a, &b, 0.5);
        add_scaled_inplace_scalar(&mut r, &b, 0.5);
        assert_within_ulps(&a, &r, 1, &format!("add_scaled_inplace n={n}"));
    }
    // Known values.
    let mut a = [4.0f32, 8.0];
    add_scaled_inplace(&mut a, &[4.0, 8.0], 0.5);
    assert_eq!(a, [6.0, 12.0]);
}

#[test]
fn add_mul_inplace_within_one_ulp() {
    let mut seed = 3;
    for &n in SIZES {
        let a0 = rand_vec(n, &mut seed);
        let b = rand_vec(n, &mut seed);
        let c = rand_vec(n, &mut seed);
        let mut a = a0.clone();
        let mut r = a0.clone();
        add_mul_inplace(&mut a, &b, &c);
        add_mul_inplace_scalar(&mut r, &b, &c);
        assert_within_ulps(&a, &r, 1, &format!("add_mul_inplace n={n}"));
    }
}

#[test]
fn modulate_inplace_matches_scalar() {
    let mut seed = 4;
    for &n in SIZES {
        let x0: Vec<f32> = rand_vec(n, &mut seed).iter().map(|v| v * 8.0).collect();
        let scale = rand_vec(n, &mut seed);
        let shift = rand_vec(n, &mut seed);
        let mut a = x0.clone();
        let mut b = x0.clone();
        modulate_inplace(&mut a, &scale, &shift);
        modulate_inplace_scalar(&mut b, &scale, &shift);
        assert!(max_abs_diff(&a, &b) < 1e-6, "modulate n={n}");
        for i in 0..n {
            assert!((b[i] - (x0[i] * (1.0 + scale[i]) + shift[i])).abs() < 1e-6);
        }
    }
}

// ───────────────────────── activations ─────────────────────────

#[test]
fn silu_matches_scalar_and_reference() {
    let mut seed = 5;
    for &n in SIZES {
        let x0: Vec<f32> = rand_vec(n, &mut seed).iter().map(|v| v * 10.0).collect();
        let mut a = x0.clone();
        let mut b = x0.clone();
        silu_inplace(&mut a);
        silu_inplace_scalar(&mut b);
        // Polynomial exp in the SIMD paths: ≤ 1e-6 abs on sigmoid, scaled by |x| ≤ 10.
        assert!(
            max_abs_diff(&a, &b) < 1e-5,
            "silu n={n} diff={}",
            max_abs_diff(&a, &b)
        );
    }
    let mut x = [0.0f32, 1.0, -1.0, 20.0, -20.0];
    silu_inplace(&mut x);
    assert!((x[0]).abs() < 1e-6);
    assert!((x[1] - 0.731_058_6).abs() < 1e-5);
    assert!((x[2] + 0.268_941_4).abs() < 1e-5);
    assert!((x[3] - 20.0).abs() < 1e-4);
    assert!(x[4].abs() < 1e-6);
}

#[test]
fn gelu_matches_scalar_and_reference() {
    let mut seed = 6;
    for &n in SIZES {
        let x0: Vec<f32> = rand_vec(n, &mut seed).iter().map(|v| v * 6.0).collect();
        let mut a = x0.clone();
        let mut b = x0.clone();
        gelu_inplace(&mut a);
        gelu_inplace_scalar(&mut b);
        assert!(
            max_abs_diff(&a, &b) < 2e-5,
            "gelu n={n} diff={}",
            max_abs_diff(&a, &b)
        );
    }
    // torch.nn.functional.gelu reference values.
    let mut v = vec![0.0f32, 1.0, -1.0, 2.5, -2.5, 0.1];
    gelu_inplace(&mut v);
    let expect = [0.0, 0.841345, -0.158655, 2.484475, -0.015525, 0.053983];
    for (a, e) in v.iter().zip(&expect) {
        assert!((a - e).abs() < 2e-5, "{a} vs {e}");
    }
}

#[test]
fn gelu_tanh_matches_scalar_and_reference() {
    let mut seed = 7;
    for &n in SIZES {
        let x0: Vec<f32> = rand_vec(n, &mut seed).iter().map(|v| v * 6.0).collect();
        let mut a = x0.clone();
        let mut b = x0.clone();
        gelu_tanh_inplace(&mut a);
        gelu_tanh_inplace_scalar(&mut b);
        assert!(
            max_abs_diff(&a, &b) < 2e-5,
            "gelu_tanh n={n} diff={}",
            max_abs_diff(&a, &b)
        );
    }
    // torch.nn.functional.gelu(x, approximate="tanh") reference values.
    let mut v = vec![0.0f32, 1.0, -1.0, 2.0, 2.5, -2.5, 0.1];
    gelu_tanh_inplace(&mut v);
    let expect = [
        0.0, 0.841192, -0.158808, 1.954598, 2.484916, -0.015084, 0.053983,
    ];
    for (a, e) in v.iter().zip(&expect) {
        assert!((a - e).abs() < 2e-5, "{a} vs {e}");
    }
    // Scalar reference is tighter than the dispatcher contract.
    let mut v = [0.0f32, 1.0, -1.0, 2.0];
    gelu_tanh_inplace_scalar(&mut v);
    let expect = [0.0f32, 0.841_192, -0.158_808, 1.954_598];
    for (a, e) in v.iter().zip(&expect) {
        assert!((a - e).abs() < 2e-6, "{a} vs {e}");
    }
}

#[test]
fn softmax_matches_scalar_and_handles_masks() {
    let mut seed = 8;
    for &cols in SIZES {
        let rows = 3;
        let s0: Vec<f32> = rand_vec(rows * cols, &mut seed)
            .iter()
            .map(|v| v * 12.0)
            .collect();
        let mut a = s0.clone();
        let mut b = s0.clone();
        softmax_row_inplace(&mut a, rows, cols);
        softmax_row_inplace_scalar(&mut b, rows, cols);
        assert!(
            max_abs_diff(&a, &b) < 2e-6,
            "softmax cols={cols} diff={}",
            max_abs_diff(&a, &b)
        );
        for r in 0..rows {
            let sum: f32 = a[r * cols..(r + 1) * cols].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}");
        }
    }
    // -inf masking and all-masked rows (two rows, 9 columns → SIMD chunk + tail).
    let ninf = f32::NEG_INFINITY;
    let mut s = vec![0.0f32, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf];
    s.extend_from_slice(&[ninf; 9]);
    s.extend_from_slice(&[1.0, 2.0, ninf, ninf, ninf, ninf, ninf, ninf, ninf]);
    let mut r = s.clone();
    softmax_row_inplace(&mut s, 3, 9);
    softmax_row_inplace_scalar(&mut r, 3, 9);
    // Scalar: masked entries are exactly 0. SIMD paths clamp the exp
    // argument at -87 before the polynomial, so a masked entry comes out
    // as ~1.6e-38 instead of 0 — negligible, but not bit-zero.
    const MASKED: f32 = 1e-30;
    assert!((s[0] - 1.0).abs() < 1e-6);
    assert!(s[1..9].iter().all(|&v| v.abs() < MASKED), "{:?}", &s[1..9]);
    assert!(r[1..9].iter().all(|&v| v == 0.0));
    assert!(
        s[9..18].iter().all(|&v| v == 0.0),
        "all-masked row must be zeros"
    );
    let e = 1.0f32.exp();
    let denom = 1.0 + e;
    assert!((s[18] - 1.0 / denom).abs() < 1e-6);
    assert!((s[19] - e / denom).abs() < 1e-6);
    assert!(s[20..27].iter().all(|&v| v.abs() < MASKED));
    assert!(r[20..27].iter().all(|&v| v == 0.0));
}

#[test]
fn exp_clamp_matches_reference() {
    let lo = -12.0f32;
    let hi = 100.0f32.ln();
    let mut v = [-20.0f32, -12.0, 0.0, 3.0, 4.7, 10.0];
    let mut r = v;
    exp_clamp_inplace(&mut v, lo, hi);
    exp_clamp_inplace_scalar(&mut r, lo, hi);
    assert_bitexact(&v, &r, "exp_clamp");
    let expect = [lo.exp(), lo.exp(), 1.0, 3.0f32.exp(), hi.exp(), hi.exp()];
    for (a, e) in v.iter().zip(&expect) {
        assert!((a - e).abs() <= e.abs() * 1e-6, "{a} vs {e}");
    }
    assert!((v[5] - 100.0).abs() < 1e-3, "{}", v[5]);
}

#[test]
fn sincos_unit_norm_and_reference() {
    let p: Vec<f32> = (0..64).map(|i| (i as f32) * 1.7 - 50.0).collect();
    let mut c = vec![0.0f32; 64];
    let mut s = vec![0.0f32; 64];
    sincos_into(&p, &mut c, &mut s);
    for i in 0..64 {
        assert!((c[i] * c[i] + s[i] * s[i] - 1.0).abs() < 1e-6);
        assert!((c[i] - p[i].cos()).abs() < 1e-6);
        assert!((s[i] - p[i].sin()).abs() < 1e-6);
    }
}

// ───────────────────────── normalisation ─────────────────────────

#[test]
fn layer_norm_frames1_matches_scalar() {
    let mut seed = 9;
    for &channels in SIZES {
        if channels < 2 {
            continue;
        }
        let input = rand_vec(channels, &mut seed);
        let gamma: Vec<f32> = rand_vec(channels, &mut seed)
            .iter()
            .map(|v| 0.5 + v)
            .collect();
        let beta = rand_vec(channels, &mut seed);
        let mut a = vec![0.0f32; channels];
        let mut b = vec![0.0f32; channels];
        layer_norm(&input, &gamma, &beta, channels, 1, 1e-5, &mut a);
        layer_norm_scalar(&input, &gamma, &beta, channels, 1, 1e-5, &mut b);
        let d = max_abs_diff(&a, &b);
        assert!(d < 1e-4, "layer_norm c={channels} diff={d}");
    }
}

#[test]
fn layer_norm_multi_frame_is_scalar_path() {
    let mut seed = 10;
    let (c, t) = (24, 5);
    let input = rand_vec(c * t, &mut seed);
    let gamma = rand_vec(c, &mut seed);
    let beta = rand_vec(c, &mut seed);
    let mut a = vec![0.0f32; c * t];
    let mut b = vec![0.0f32; c * t];
    layer_norm(&input, &gamma, &beta, c, t, 1e-5, &mut a);
    layer_norm_scalar(&input, &gamma, &beta, c, t, 1e-5, &mut b);
    assert_bitexact(&a, &b, "layer_norm frames>1");
    // Frames are independent: column t of the batched call equals a
    // frames=1 call on that column (up to the SIMD tolerance).
    for tt in 0..t {
        let col: Vec<f32> = (0..c).map(|cc| input[cc * t + tt]).collect();
        let mut one = vec![0.0f32; c];
        layer_norm(&col, &gamma, &beta, c, 1, 1e-5, &mut one);
        for cc in 0..c {
            assert!((one[cc] - a[cc * t + tt]).abs() < 1e-4);
        }
    }
}

#[test]
fn layer_norm_known_values() {
    // x = [1, 2, 3]: μ = 2, biased var = 2/3.
    let input = [1.0f32, 2.0, 3.0];
    let gamma = [1.0f32; 3];
    let beta = [0.0f32; 3];
    let mut out = [0.0f32; 3];
    layer_norm(&input, &gamma, &beta, 3, 1, 1e-5, &mut out);
    let sigma = ((2.0_f32 / 3.0) + 1e-5).sqrt();
    assert!((out[0] + 1.0 / sigma).abs() < 1e-4);
    assert!(out[1].abs() < 1e-4);
    assert!((out[2] - 1.0 / sigma).abs() < 1e-4);
}

#[test]
fn rms_norm_unbiased_known_values() {
    // x = [1,2,3,4]: mean 2.5, var(n-1) = 5/3.
    let x = [1.0, 2.0, 3.0, 4.0];
    let alpha = [1.0f32; 4];
    let mut y = [0.0f32; 4];
    rms_norm_unbiased(&x, &alpha, 0.0, &mut y);
    let inv = 1.0 / (5.0f32 / 3.0).sqrt();
    for i in 0..4 {
        assert!(
            (y[i] - x[i] * inv).abs() < 1e-6,
            "{} vs {}",
            y[i],
            x[i] * inv
        );
    }
}

// ───────────────────────── f32 matrix products ─────────────────────────

#[test]
fn matmul_mv_matches_scalar() {
    let mut seed = 11;
    for &in_f in SIZES {
        let out_f = 5;
        let x = rand_vec(in_f, &mut seed);
        let w = rand_vec(in_f * out_f, &mut seed);
        let mut a = vec![0.0f32; out_f];
        let mut b = vec![0.0f32; out_f];
        matmul_mv(&x, &w, in_f, out_f, &mut a);
        matmul_mv_scalar(&x, &w, in_f, out_f, &mut b);
        let d = max_abs_diff(&a, &b);
        assert!(d < 1e-4, "matmul_mv in={in_f} diff={d}");
    }
    let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut out = vec![0.0_f32; 2];
    matmul_mv(&input, &weight, 5, 2, &mut out);
    assert_eq!(out, [15.0, 6.0]);
}

#[test]
fn dot_product_matches_scalar() {
    let mut seed = 12;
    for &n in SIZES {
        let a = rand_vec(n, &mut seed);
        let b = rand_vec(n, &mut seed);
        let d = (dot_product_f32(&a, &b) - dot_product_f32_scalar(&a, &b)).abs();
        assert!(d < 1e-4, "dot n={n} diff={d}");
    }
    assert_eq!(dot_product_f32(&[], &[]), 0.0);
}

#[test]
fn matmul_mm_matches_scalar() {
    let mut seed = 13;
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (3, 5, 7),
        (4, 16, 8),
        (5, 24, 9),
        (2, 33, 17),
        (7, 64, 32),
    ] {
        let a = rand_vec(m * k, &mut seed);
        let b = rand_vec(k * n, &mut seed);
        let mut x = vec![0.0f32; m * n];
        let mut y = vec![0.0f32; m * n];
        matmul_mm(&a, &b, m, k, n, &mut x);
        matmul_mm_scalar(&a, &b, m, k, n, &mut y);
        let d = max_abs_diff(&x, &y);
        assert!(d < 1e-4, "matmul_mm {m}x{k}x{n} diff={d}");
    }
    // Identity.
    let a = [1.0f32, 0.0, 0.0, 1.0];
    let b = [1.0f32, 2.0, 3.0, 4.0];
    let mut out = [0.0f32; 4];
    matmul_mm(&a, &b, 2, 2, 2, &mut out);
    assert_eq!(out, b);
}

#[test]
fn matmul_mm_transposed_b_matches_scalar() {
    let mut seed = 14;
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (3, 5, 7),
        (4, 16, 8),
        (5, 24, 9),
        (2, 33, 17),
        (7, 64, 32),
        (3, 100, 4),
    ] {
        let a = rand_vec(m * k, &mut seed);
        let b = rand_vec(n * k, &mut seed);
        let mut x = vec![0.0f32; m * n];
        let mut y = vec![0.0f32; m * n];
        matmul_mm_transposed_b(&a, &b, m, k, n, &mut x);
        matmul_mm_transposed_b_scalar(&a, &b, m, k, n, &mut y);
        let d = max_abs_diff(&x, &y);
        assert!(d < 1e-4, "matmul_mm_transposed_b {m}x{k}x{n} diff={d}");
        // Equals matmul_mm against the explicitly transposed b.
        let mut bt = vec![0.0f32; k * n];
        for j in 0..n {
            for p in 0..k {
                bt[p * n + j] = b[j * k + p];
            }
        }
        let mut z = vec![0.0f32; m * n];
        matmul_mm_scalar(&a, &bt, m, k, n, &mut z);
        assert_bitexact(&y, &z, "transposed_b scalar vs mm scalar");
    }
}

// ───────────────────────── fp16 matrix products ─────────────────────────

#[test]
fn matmul_mv_f16_matches_scalar() {
    let mut seed = 15;
    for &in_f in SIZES {
        let out_f = 6;
        let x = rand_vec(in_f, &mut seed);
        let w16: Vec<u16> = rand_vec(in_f * out_f, &mut seed)
            .iter()
            .map(|v| f32_to_f16(v * 0.1))
            .collect();
        let mut a = vec![0.0f32; out_f];
        let mut b = vec![0.0f32; out_f];
        matmul_mv_f16(&x, &w16, in_f, out_f, &mut a);
        matmul_mv_f16_scalar(&x, &w16, in_f, out_f, &mut b);
        let d = max_abs_diff(&a, &b);
        assert!(d < 1e-5, "matmul_mv_f16 in={in_f} diff={d}");
    }
}

#[test]
fn matmul_mv_f16_handles_subnormal_halves() {
    // smallest subnormal, largest subnormal, -smallest subnormal, smallest normal
    let w16 = vec![0x0001u16, 0x03FF, 0x8001, 0x0400];
    let x = vec![1.0f32; 4];
    let mut y = vec![0.0; 1];
    matmul_mv_f16(&x, &w16, 4, 1, &mut y);
    let mut yr = vec![0.0; 1];
    matmul_mv_f16_scalar(&x, &w16, 4, 1, &mut yr);
    assert!((y[0] - yr[0]).abs() < 1e-12, "{} vs {}", y[0], yr[0]);
    // 16 + 4 lanes so the subnormals also go through the vector path.
    let w16: Vec<u16> = (0..20)
        .map(|i| [0x0001u16, 0x03FF, 0x8001, 0x0400, 0x0200][i % 5])
        .collect();
    let x = vec![1.0f32; 20];
    matmul_mv_f16(&x, &w16, 20, 1, &mut y);
    matmul_mv_f16_scalar(&x, &w16, 20, 1, &mut yr);
    assert!((y[0] - yr[0]).abs() < 1e-10, "{} vs {}", y[0], yr[0]);
}

#[test]
fn matmul_mm_f16_bitexact_vs_rowwise_gemv() {
    // The batched GEMM must be bit-equal to row-wise GEMV calls on the
    // same dispatch path. m covers the 4-row block + tail, k covers
    // 16-wide chunks + scalar remainder, n covers more than one panel.
    let mut seed = 16;
    for (m, k, n) in [
        (5usize, 48usize, 7usize),
        (4, 37, 3),
        (1, 16, 2),
        (7, 64, 33),
        (9, 80, 70),
    ] {
        let a = rand_vec(m * k, &mut seed);
        let w16: Vec<u16> = rand_vec(n * k, &mut seed)
            .iter()
            .map(|v| f32_to_f16(v * 0.1))
            .collect();
        let mut mm = vec![0.0f32; m * n];
        matmul_mm_f16(&a, &w16, m, k, n, &mut mm);
        let mut mms = vec![0.0f32; m * n];
        matmul_mm_f16_scalar(&a, &w16, m, k, n, &mut mms);
        let mut mv = vec![0.0f32; n];
        for i in 0..m {
            matmul_mv_f16(&a[i * k..(i + 1) * k], &w16, k, n, &mut mv);
            assert_bitexact(
                &mm[i * n..(i + 1) * n],
                &mv,
                &format!("dispatch m={m} k={k} n={n} i={i}"),
            );
            matmul_mv_f16_scalar(&a[i * k..(i + 1) * k], &w16, k, n, &mut mv);
            assert_bitexact(
                &mms[i * n..(i + 1) * n],
                &mv,
                &format!("scalar m={m} k={k} n={n} i={i}"),
            );
        }
        assert!(max_abs_diff(&mm, &mms) < 1e-5);
    }
}

#[test]
fn embed_lookup_reads_expected_row() {
    // 3×2 fp16 table: rows [1,2], [3,4], [5,6].
    let table: Vec<u16> = vec![0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
    let mut out = [0.0f32; 2];
    embed_lookup_f16(&table, 2, 1, &mut out);
    assert_eq!(out, [3.0, 4.0]);
    embed_lookup_f16(&table, 2, 2, &mut out);
    assert_eq!(out, [5.0, 6.0]);
}

// ───────────────────────── int8 matrix products ─────────────────────────

#[test]
fn matmul_mv_i8_matches_scalar() {
    let mut seed = 17;
    for &in_f in SIZES {
        let out_f = 6;
        let x = rand_vec(in_f, &mut seed);
        let w: Vec<i8> = rand_vec(in_f * out_f, &mut seed)
            .iter()
            .map(|v| (v * 127.0) as i8)
            .collect();
        let scale: Vec<f32> = rand_vec(out_f, &mut seed)
            .iter()
            .map(|v| v.abs() * 0.01)
            .collect();
        let mut a = vec![0.0f32; out_f];
        let mut b = vec![0.0f32; out_f];
        matmul_mv_i8(&x, &w, &scale, in_f, out_f, &mut a);
        matmul_mv_i8_scalar(&x, &w, &scale, in_f, out_f, &mut b);
        let d = max_abs_diff(&a, &b);
        assert!(d < 1e-4, "matmul_mv_i8 in={in_f} diff={d}");
    }
}

#[test]
fn matmul_mm_i8_bitexact_vs_rowwise_gemv() {
    let mut seed = 18;
    for (m, k, n) in [
        (5usize, 48usize, 7usize),
        (4, 37, 3),
        (1, 16, 2),
        (7, 64, 33),
        (9, 80, 70),
    ] {
        let a = rand_vec(m * k, &mut seed);
        let w: Vec<i8> = rand_vec(n * k, &mut seed)
            .iter()
            .map(|v| (v * 127.0) as i8)
            .collect();
        let scale: Vec<f32> = rand_vec(n, &mut seed)
            .iter()
            .map(|v| v.abs() * 0.01)
            .collect();
        let mut mm = vec![0.0f32; m * n];
        matmul_mm_i8(&a, &w, &scale, m, k, n, &mut mm);
        let mut mms = vec![0.0f32; m * n];
        matmul_mm_i8_scalar(&a, &w, &scale, m, k, n, &mut mms);
        let mut mv = vec![0.0f32; n];
        for i in 0..m {
            matmul_mv_i8(&a[i * k..(i + 1) * k], &w, &scale, k, n, &mut mv);
            assert_bitexact(
                &mm[i * n..(i + 1) * n],
                &mv,
                &format!("dispatch m={m} k={k} n={n} i={i}"),
            );
            matmul_mv_i8_scalar(&a[i * k..(i + 1) * k], &w, &scale, k, n, &mut mv);
            assert_bitexact(
                &mms[i * n..(i + 1) * n],
                &mv,
                &format!("scalar m={m} k={k} n={n} i={i}"),
            );
        }
        assert!(max_abs_diff(&mm, &mms) < 1e-4);
    }
}

// ───────────────────────── depthwise convolution ─────────────────────────

/// Channel-major depthwise reference (PyTorch `[C, 1, K]` weight layout).
fn depthwise_channel_major_ref(
    input_cm: &[f32],
    weight_cm: &[f32],
    bias: &[f32],
    ch: usize,
    k: usize,
    t_in: usize,
    out_cm: &mut [f32],
) {
    let t_out = t_in - (k - 1);
    for c in 0..ch {
        for t in 0..t_out {
            let mut acc = bias[c];
            for kk in 0..k {
                acc += weight_cm[c * k + kk] * input_cm[c * t_in + t + kk];
            }
            out_cm[c * t_out + t] = acc;
        }
    }
}

#[test]
fn depthwise_nlc_matches_channel_major_reference() {
    let mut seed = 19;
    for &(ch, k, t_in) in &[
        (6usize, 7usize, 11usize),
        (8, 3, 3),
        (13, 5, 9),
        (64, 7, 10),
    ] {
        let t_out = t_in - (k - 1);
        let input_cm = rand_vec(ch * t_in, &mut seed);
        let weight_cm = rand_vec(ch * k, &mut seed);
        let bias = rand_vec(ch, &mut seed);
        let mut out_cm = vec![0.0f32; ch * t_out];
        depthwise_channel_major_ref(&input_cm, &weight_cm, &bias, ch, k, t_in, &mut out_cm);

        let mut input_nlc = vec![0.0f32; t_in * ch];
        for c in 0..ch {
            for t in 0..t_in {
                input_nlc[t * ch + c] = input_cm[c * t_in + t];
            }
        }
        let mut wk = vec![0.0f32; k * ch];
        for c in 0..ch {
            for kk in 0..k {
                wk[kk * ch + c] = weight_cm[c * k + kk];
            }
        }
        let mut out_nlc = vec![0.0f32; t_out * ch];
        let mut out_nlc_s = vec![0.0f32; t_out * ch];
        conv1d_depthwise_nlc(&input_nlc, &wk, &bias, ch, k, t_in, &mut out_nlc);
        conv1d_depthwise_nlc_scalar(&input_nlc, &wk, &bias, ch, k, t_in, &mut out_nlc_s);
        for c in 0..ch {
            for t in 0..t_out {
                let (a, s, r) = (
                    out_nlc[t * ch + c],
                    out_nlc_s[t * ch + c],
                    out_cm[c * t_out + t],
                );
                assert!((a - r).abs() < 1e-5, "dispatch c={c} t={t}: {a} vs {r}");
                assert_eq!(s.to_bits(), r.to_bits(), "scalar c={c} t={t}: {s} vs {r}");
            }
        }
    }
}

#[test]
fn depthwise_nlc_batch_equals_stepwise_bitexact() {
    // One call over T frames must equal T single-frame calls over the
    // sliding window bit for bit — offline and streaming decode agree.
    let mut seed = 20;
    for &(ch, k, t_out) in &[(8usize, 7usize, 5usize), (6, 3, 4), (37, 5, 6)] {
        let t_in = t_out + k - 1;
        let input = rand_vec(t_in * ch, &mut seed);
        let wk = rand_vec(k * ch, &mut seed);
        let bias = rand_vec(ch, &mut seed);
        let mut batch = vec![0.0f32; t_out * ch];
        conv1d_depthwise_nlc(&input, &wk, &bias, ch, k, t_in, &mut batch);
        for t in 0..t_out {
            let mut one = vec![0.0f32; ch];
            conv1d_depthwise_nlc(&input[t * ch..(t + k) * ch], &wk, &bias, ch, k, k, &mut one);
            assert_bitexact(
                &one,
                &batch[t * ch..(t + 1) * ch],
                &format!("ch={ch} k={k} t={t}"),
            );
        }
    }
}
