//! Rotary position embedding on interleaved pairs: element `2i` is the
//! real part and `2i + 1` the imaginary part of pair `i` inside each
//! head. The frequency table is `exp(-ln(max_period) · 2i / D)` and the
//! angle for position `p` is `freq_i · p`, both in f32 as in the
//! reference.

/// Per-model RoPE tables: frequencies for one head.
pub struct Rope {
    freqs: Vec<f32>,
    head_width: usize,
}

impl Rope {
    pub fn new(head_width: usize, max_period: f64) -> Self {
        let half = head_width / 2;
        // The reference multiplies the f32 index by the f32-rounded constant.
        let c = (-(max_period.ln()) * 2.0 / head_width as f64) as f32;
        let freqs = (0..half).map(|i| (i as f32 * c).exp()).collect();
        Rope { freqs, head_width }
    }

    /// Number of (sin, cos) entries per position.
    pub fn half(&self) -> usize {
        self.freqs.len()
    }

    /// Fill `sin`/`cos` (each `half` long) for absolute position `pos`.
    pub fn tables(&self, pos: usize, sin: &mut [f32], cos: &mut [f32]) {
        let p = pos as f32;
        for (i, f) in self.freqs.iter().enumerate() {
            let (s, c) = (f * p).sin_cos();
            sin[i] = s;
            cos[i] = c;
        }
    }

    /// Rotate every head of `v` (`n_heads · head_width` values) in place
    /// with the tables of one position.
    pub fn apply(&self, v: &mut [f32], sin: &[f32], cos: &[f32]) {
        let hw = self.head_width;
        debug_assert_eq!(v.len() % hw, 0);
        for head in v.chunks_exact_mut(hw) {
            rotate_head(head, sin, cos);
        }
    }
}

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline]
fn rotate_head(head: &mut [f32], sin: &[f32], cos: &[f32]) {
    use core::arch::aarch64::*;
    let half = head.len() / 2;
    let quads = half / 4;
    // SAFETY: `head` holds 2·half floats, `sin`/`cos` hold `half`; every
    // vector access below stays within `quads · 4 ≤ half` pairs.
    unsafe {
        for qb in 0..quads {
            let p = head.as_mut_ptr().add(qb * 8);
            let pair = vld2q_f32(p as *const f32);
            let sn = vld1q_f32(sin.as_ptr().add(qb * 4));
            let cs = vld1q_f32(cos.as_ptr().add(qb * 4));
            let re = vfmsq_f32(vmulq_f32(pair.0, cs), pair.1, sn);
            let im = vfmaq_f32(vmulq_f32(pair.1, cs), pair.0, sn);
            vst2q_f32(p, float32x4x2_t(re, im));
        }
    }
    rotate_scalar(head, sin, cos, quads * 4);
}

#[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
#[inline]
fn rotate_head(head: &mut [f32], sin: &[f32], cos: &[f32]) {
    rotate_scalar(head, sin, cos, 0);
}

/// Scalar rotation of pairs `from..half`.
#[inline]
fn rotate_scalar(head: &mut [f32], sin: &[f32], cos: &[f32], from: usize) {
    let half = head.len() / 2;
    for i in from..half {
        let re = head[2 * i];
        let im = head[2 * i + 1];
        head[2 * i] = re * cos[i] - im * sin[i];
        head[2 * i + 1] = re * sin[i] + im * cos[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_norm_and_matches_scalar() {
        let rope = Rope::new(64, 10_000.0);
        let mut sin = vec![0.0; 32];
        let mut cos = vec![0.0; 32];
        rope.tables(37, &mut sin, &mut cos);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut a = v.clone();
        rope.apply(&mut a, &sin, &cos);
        let mut b = v.clone();
        for head in b.chunks_exact_mut(64) {
            rotate_scalar(head, &sin, &cos, 0);
        }
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-6, "{x} vs {y}");
        }
        let n0: f32 = v.iter().map(|x| x * x).sum();
        let n1: f32 = a.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3 * n0);
    }

    #[test]
    fn position_zero_is_identity() {
        let rope = Rope::new(8, 10_000.0);
        let mut sin = vec![0.0; 4];
        let mut cos = vec![0.0; 4];
        rope.tables(0, &mut sin, &mut cos);
        let mut v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let orig = v.clone();
        rope.apply(&mut v, &sin, &cos);
        assert_eq!(v, orig);
    }
}
