//! Radix-2 Cooley–Tukey FFT (f64 internally), inverse real FFT and the
//! periodic Hann window used by the iSTFT synthesis head.
//!
//! The transform runs in f64 so that the output is comparable with
//! `numpy.fft` / FFTW double-precision references; the f32 spectrum the
//! decoder produces is widened on entry and the time signal is narrowed
//! on exit.

/// Pre-computed FFT plan for a fixed power-of-two size.
pub struct FftPlan {
    n: usize,
    twiddles: Vec<(f64, f64)>,
    bit_rev: Vec<usize>,
}

impl FftPlan {
    /// Build a plan for transform size `n` (must be a power of two).
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n > 0,
            "FFT size must be a power of two, got {n}"
        );
        let log2_n = n.trailing_zeros() as usize;
        let pi = std::f64::consts::PI;
        let twiddles: Vec<(f64, f64)> = (0..n / 2)
            .map(|k| {
                let theta = -2.0 * pi * (k as f64) / (n as f64);
                (theta.cos(), theta.sin())
            })
            .collect();
        let bit_rev: Vec<usize> = (0..n).map(|i| reverse_bits(i, log2_n)).collect();
        Self {
            n,
            twiddles,
            bit_rev,
        }
    }

    /// Transform size this plan was built for.
    pub fn size(&self) -> usize {
        self.n
    }
}

#[inline]
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut r = 0;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Forward complex FFT, in place, f64 throughout. `data` holds `(re, im)`
/// pairs and must have exactly `plan.size()` elements.
pub fn fft_forward(plan: &FftPlan, data: &mut [(f64, f64)]) {
    let n = plan.n;
    assert_eq!(data.len(), n, "FFT input length mismatch");

    for i in 0..n {
        let j = plan.bit_rev[i];
        if j > i {
            data.swap(i, j);
        }
    }

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let step = n / size;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let (tw_cos, tw_sin) = plan.twiddles[k * step];
                let (a_re, a_im) = data[start + k];
                let (b_re, b_im) = data[start + k + half];
                let t_re = tw_cos * b_re - tw_sin * b_im;
                let t_im = tw_cos * b_im + tw_sin * b_re;
                data[start + k] = (a_re + t_re, a_im + t_im);
                data[start + k + half] = (a_re - t_re, a_im - t_im);
            }
            start += size;
        }
        size *= 2;
    }
}

/// Inverse real FFT (`numpy.fft.irfft` twin) built on the forward plan
/// via the conjugation identity `irfft(S) = real(FFT(conj(S_full))) / n`,
/// where `S_full` is the Hermitian extension of the `n/2 + 1` bins.
///
/// - `spec`: `n/2 + 1` complex bins as `(re, im)` f32 pairs. The
///   imaginary parts of the DC and Nyquist bins are used as given; the
///   caller zeroes them if the analysis convention requires real values.
/// - `scratch`: caller-owned `[(f64, f64); n]` so the hot path does
///   not allocate.
/// - `out`: `n` real samples.
pub fn irfft_real(
    plan: &FftPlan,
    spec: &[(f32, f32)],
    scratch: &mut [(f64, f64)],
    out: &mut [f32],
) {
    let n = plan.n;
    assert_eq!(spec.len(), n / 2 + 1, "spec must hold n/2+1 bins");
    assert_eq!(scratch.len(), n, "scratch length mismatch");
    assert_eq!(out.len(), n, "output length mismatch");
    // conj(X[k]) for k <= n/2; the mirror half k > n/2 satisfies
    // X[k] = conj(X[n-k]), hence conj(X[k]) = X[n-k] = (re[n-k], im[n-k]).
    for (k, &(re, im)) in spec.iter().enumerate() {
        scratch[k] = (re as f64, -(im as f64));
    }
    for k in n / 2 + 1..n {
        let (re, im) = spec[n - k];
        scratch[k] = (re as f64, im as f64);
    }
    fft_forward(plan, scratch);
    let inv_n = 1.0 / n as f64;
    for (o, &(re, _)) in out.iter_mut().zip(scratch.iter()) {
        *o = (re * inv_n) as f32;
    }
}

/// Periodic Hann window `w[k] = 0.5 * (1 - cos(2π k / n))` for `k in 0..n`.
/// Matches `torch.hann_window(n, periodic=True)`,
/// `scipy.signal.windows.hann(n, sym=False)` and `np.hanning(n + 1)[:-1]`.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    let two_pi = 2.0 * std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let c = (two_pi * (k as f64) / (n as f64)).cos();
            (0.5 - 0.5 * c) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_impulse_is_flat() {
        let plan = FftPlan::new(8);
        let mut data = vec![(0.0_f64, 0.0_f64); 8];
        data[0] = (1.0, 0.0);
        fft_forward(&plan, &mut data);
        for (re, im) in &data {
            assert!((re - 1.0).abs() < 1e-14, "re = {re}");
            assert!(im.abs() < 1e-14, "im = {im}");
        }
    }

    #[test]
    fn fft_dc_is_n() {
        let plan = FftPlan::new(8);
        let mut data = vec![(1.0_f64, 0.0_f64); 8];
        fft_forward(&plan, &mut data);
        assert!((data[0].0 - 8.0).abs() < 1e-13);
        for d in data.iter().skip(1) {
            assert!(d.0.abs() < 1e-13);
        }
    }

    #[test]
    fn fft_512_parseval() {
        let plan = FftPlan::new(512);
        let mut data: Vec<(f64, f64)> = (0..512).map(|i| ((0.123 * i as f64).sin(), 0.0)).collect();
        let energy_in: f64 = data.iter().map(|(re, _)| re * re).sum();
        fft_forward(&plan, &mut data);
        let energy_out: f64 = data.iter().map(|(re, im)| re * re + im * im).sum();
        let ratio = energy_out / 512.0 / energy_in;
        assert!((ratio - 1.0).abs() < 1e-12, "Parseval ratio = {ratio}");
    }

    #[test]
    fn irfft_dc_only_is_constant() {
        let n = 16;
        let plan = FftPlan::new(n);
        let mut spec = vec![(0.0_f32, 0.0_f32); n / 2 + 1];
        spec[0] = (n as f32, 0.0);
        let mut scratch = vec![(0.0_f64, 0.0_f64); n];
        let mut out = vec![0.0_f32; n];
        irfft_real(&plan, &spec, &mut scratch, &mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-6, "v = {v}");
        }
    }

    #[test]
    fn irfft_roundtrips_forward_fft() {
        let n = 1024;
        let plan = FftPlan::new(n);
        let sig: Vec<f32> = (0..n).map(|t| (0.37 * t as f32).sin() * 0.8).collect();
        let mut fwd: Vec<(f64, f64)> = sig.iter().map(|&v| (v as f64, 0.0)).collect();
        fft_forward(&plan, &mut fwd);
        let spec: Vec<(f32, f32)> = fwd[..n / 2 + 1]
            .iter()
            .map(|&(re, im)| (re as f32, im as f32))
            .collect();
        let mut scratch = vec![(0.0_f64, 0.0_f64); n];
        let mut out = vec![0.0_f32; n];
        irfft_real(&plan, &spec, &mut scratch, &mut out);
        for (t, (&a, &b)) in sig.iter().zip(&out).enumerate() {
            // f32 quantisation of the spectrum dominates the error budget.
            assert!((a - b).abs() < 1e-4, "t={t}: {a} vs {b}");
        }
    }

    #[test]
    fn hann_periodic_shape() {
        let w = hann_periodic(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6, "w[0] = {}", w[0]);
        assert!((w[200] - 1.0).abs() < 1e-6, "w[200] = {}", w[200]);
        // Periodic window: w[k] == w[n - k].
        for k in 1..200 {
            assert!((w[k] - w[400 - k]).abs() < 1e-6, "k = {k}");
        }
        // Sum of a periodic Hann window is exactly n/2.
        let s: f64 = w.iter().map(|&v| v as f64).sum();
        assert!((s - 200.0).abs() < 1e-3, "sum = {s}");
    }
}
