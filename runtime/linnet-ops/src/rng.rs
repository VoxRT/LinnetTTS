//! Seedable PRNG and Gaussian sampler for the flow head of the TTS model.
//!
//! xoshiro256** (Vigna, public domain): 256 bits of state, byte-identical
//! output on every platform for a given seed, which makes A/B runs and
//! unit tests deterministic. The seed is expanded into the state with
//! SplitMix64, as recommended by the algorithm's author, to avoid
//! near-zero states.
//!
//! The Gaussian sampler is Box–Muller over pairs of uniforms built from
//! the top 24 bits of each output (`(u >> 40) · 2⁻²⁴`).

/// xoshiro256** generator — 32 bytes of state, no heap.
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Xoshiro256StarStar {
    /// Seed the generator. The 64-bit seed is expanded through SplitMix64.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Xoshiro256StarStar { s }
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` from the top 24 bits (evenly spaced on the
    /// f32 grid).
    #[inline]
    pub fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }

    /// Fill `out` with independent `N(0, std²)` samples via Box–Muller.
    /// Two samples per pair of uniforms; an odd tail keeps only the first
    /// sample of its pair, so PRNG consumption depends only on the length.
    pub fn fill_gaussian(&mut self, out: &mut [f32], std: f32) {
        let mut i = 0;
        while i < out.len() {
            let mut u1 = self.next_f32_unit();
            let u2 = self.next_f32_unit();
            if u1 == 0.0 {
                // ln(0) = -inf; lift to the smallest grid step (probability 2^-24).
                u1 = 1.0 / (1u32 << 24) as f32;
            }
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * core::f32::consts::PI * u2;
            let (sin_t, cos_t) = theta.sin_cos();
            out[i] = r * cos_t * std;
            if i + 1 < out.len() {
                out[i + 1] = r * sin_t * std;
            }
            i += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed() {
        let mut a = Xoshiro256StarStar::new(42);
        let mut b = Xoshiro256StarStar::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Xoshiro256StarStar::new(43);
        assert_ne!(Xoshiro256StarStar::new(42).next_u64(), c.next_u64());
    }

    #[test]
    fn reference_stream_is_stable() {
        // Pin the first outputs for seed 0 so any change in the seeding or
        // the update rule is caught; these values must never change.
        let mut r = Xoshiro256StarStar::new(0);
        let first: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        assert_eq!(
            first,
            [
                0x99ec_5f36_cb75_f2b4,
                0xbf6e_1f78_4956_452a,
                0x1a5f_849d_4933_e6e0,
                0x6aa5_94f1_262d_2d2c,
            ],
            "{first:#x?}"
        );
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut r = Xoshiro256StarStar::new(7);
        for _ in 0..10_000 {
            let u = r.next_f32_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn gaussian_moments_match() {
        let std = 0.3f32.sqrt();
        let mut r = Xoshiro256StarStar::new(1234);
        let mut buf = vec![0.0f32; 100_000];
        r.fill_gaussian(&mut buf, std);
        let n = buf.len() as f64;
        let mean = buf.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = buf.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 4.0 * std as f64 / n.sqrt(), "mean {mean}");
        assert!((var.sqrt() - std as f64).abs() < 0.01, "std {}", var.sqrt());
    }

    #[test]
    fn odd_length_prefix_matches_even() {
        let mut a = Xoshiro256StarStar::new(5);
        let mut b = Xoshiro256StarStar::new(5);
        let mut x = vec![0.0f32; 7];
        let mut y = vec![0.0f32; 8];
        a.fill_gaussian(&mut x, 1.0);
        b.fill_gaussian(&mut y, 1.0);
        assert_eq!(&x[..6], &y[..6]);
        assert_eq!(x[6], y[6]);
    }
}
