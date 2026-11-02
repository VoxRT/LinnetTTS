//! The batched fp16 GEMM must be bit-equal to `m` row-wise GEMV calls on
//! the same dispatch path, for every row-blocking case the kernel has
//! (blocks of 4, 2 and 1 rows, and their combinations) and for the
//! model's reduction lengths plus a length with a scalar tail.

use linnet_ops::f16::f32_to_f16;
use linnet_ops::ops::{matmul_mm_f16, matmul_mm_f16_scalar, matmul_mv_f16};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

#[test]
fn fused_gemm_is_bitexact_to_row_gemvs() {
    let mut seed = 0x5eed_u64;
    for &k in &[384usize, 512, 1152, 2048, 40] {
        for &m in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9] {
            for &n in &[1usize, 5, 32] {
                let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
                let w: Vec<u16> = (0..n * k).map(|_| f32_to_f16(lcg(&mut seed) * 0.1)).collect();
                let mut batched = vec![0.0f32; m * n];
                matmul_mm_f16(&a, &w, m, k, n, &mut batched);
                let mut rowwise = vec![0.0f32; m * n];
                for i in 0..m {
                    matmul_mv_f16(&a[i * k..(i + 1) * k], &w, k, n, &mut rowwise[i * n..(i + 1) * n]);
                }
                for (idx, (x, y)) in batched.iter().zip(&rowwise).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "k={k} m={m} n={n} idx={idx}: batched {x} vs row-wise {y}"
                    );
                }
                // And close to the sequential scalar reference.
                let mut reference = vec![0.0f32; m * n];
                matmul_mm_f16_scalar(&a, &w, m, k, n, &mut reference);
                // Reduction order differs from the sequential reference, so
                // allow a few ULP relative to the magnitude of the sum.
                for (idx, (x, y)) in batched.iter().zip(&reference).enumerate() {
                    assert!(
                        (x - y).abs() <= 4e-5 * y.abs().max(1.0),
                        "k={k} m={m} n={n} idx={idx}: {x} vs scalar {y}"
                    );
                }
            }
        }
    }
}

#[test]
fn fused_gemm_handles_special_halves() {
    // Subnormal, zero, negative zero and large binary16 values must take
    // the same path as the scalar conversion.
    let k = 32;
    let n = 2;
    let specials: [u16; 8] = [0x0001, 0x03ff, 0x0400, 0x8000, 0x0000, 0x7bff, 0xfbff, 0x3c00];
    let w: Vec<u16> = (0..n * k).map(|i| specials[i % specials.len()]).collect();
    let a: Vec<f32> = (0..3 * k).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut batched = vec![0.0f32; 3 * n];
    matmul_mm_f16(&a, &w, 3, k, n, &mut batched);
    let mut rowwise = vec![0.0f32; 3 * n];
    for i in 0..3 {
        matmul_mv_f16(&a[i * k..(i + 1) * k], &w, k, n, &mut rowwise[i * n..(i + 1) * n]);
    }
    assert_eq!(
        batched.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        rowwise.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let mut reference = vec![0.0f32; 3 * n];
    matmul_mm_f16_scalar(&a, &w, 3, k, n, &mut reference);
    for (x, y) in batched.iter().zip(&reference) {
        assert!((x - y).abs() <= 1e-2 * y.abs().max(1.0), "{x} vs {y}");
    }
}
