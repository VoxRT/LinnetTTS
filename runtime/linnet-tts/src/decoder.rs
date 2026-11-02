//! Causal Vocos-style codec decoder: latent frames → 24 kHz waveform.
//!
//! ```text
//! latent [24] @ 25 Hz → in_proj → n_low causal ConvNeXt blocks
//!   → up (×up_factor sub-frames) → n_high causal ConvNeXt blocks @ 100 Hz
//!   → LayerNorm → head: 513 log-magnitudes + 513 phases per STFT column
//!   → exp / cos / sin → irfft(n_fft)[..win] · hann → overlap-add (hop)
//! ```
//!
//! Activations are channels-last (`[frames × channels]`). The decoder is
//! streamed one latent frame at a time; its state is the left context of
//! every depthwise convolution (`kernel − 1` frames, zeros at start — the
//! same as the offline left zero-padding) plus the `win − hop` sample
//! overlap-add tail and the absolute column index used to slice the
//! window-power envelope. Offline decoding is the same loop, so both
//! paths agree bit for bit.

use linnet_ops::fft::{hann_periodic, irfft_real, FftPlan};
use linnet_ops::ops;

use crate::config::DecoderConfig;
use crate::error::{Error, Result};
use crate::weights::{ConvNeXtWeights, DecoderWeights};

/// `nn.LayerNorm` default epsilon.
const LN_EPS: f32 = 1e-5;
/// Floor of the folded window-power envelope.
const ENV_FLOOR: f32 = 1e-8;

/// Streaming state (all zeros at the start of a stream).
pub struct DecoderState {
    low_tails: Vec<Vec<f32>>,
    high_tails: Vec<Vec<f32>>,
    ola_tail: Vec<f32>,
    col_idx: usize,
}

impl DecoderState {
    /// Back to the start-of-stream state.
    pub fn reset(&mut self) {
        for t in self.low_tails.iter_mut().chain(self.high_tails.iter_mut()) {
            t.iter_mut().for_each(|v| *v = 0.0);
        }
        self.ola_tail.iter_mut().for_each(|v| *v = 0.0);
        self.col_idx = 0;
    }
}

/// Reusable buffers for one latent frame.
pub struct DecoderScratch {
    x_low: Vec<f32>,
    x_high: Vec<f32>,
    xc: Vec<f32>,
    h: Vec<f32>,
    rows: Vec<f32>,
    ff: Vec<f32>,
    pw2: Vec<f32>,
    hn: Vec<f32>,
    ho: Vec<f32>,
    cosb: Vec<f32>,
    sinb: Vec<f32>,
    spec: Vec<(f32, f32)>,
    fft_buf: Vec<(f64, f64)>,
    frame: Vec<f32>,
    chunk: Vec<f32>,
}

/// The decoder: weights, window, envelope and FFT plan.
pub struct VocosDecoder {
    cfg: DecoderConfig,
    w: DecoderWeights,
    sample_rate: u32,
    spf: usize,
    uf: usize,
    hann: Vec<f32>,
    env_start: Vec<f32>,
    env_inner: Vec<f32>,
    fft: FftPlan,
}

impl VocosDecoder {
    pub fn new(cfg: DecoderConfig, w: DecoderWeights, sample_rate: u32) -> Self {
        let hann = hann_periodic(cfg.win);
        let (env_start, env_inner) = build_envelope(&hann, cfg.win, cfg.hop);
        let fft = FftPlan::new(cfg.n_fft);
        let spf = cfg.samples_per_frame(sample_rate);
        let uf = cfg.up_factor(sample_rate);
        VocosDecoder {
            cfg,
            w,
            sample_rate,
            spf,
            uf,
            hann,
            env_start,
            env_inner,
            fft,
        }
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Samples produced per latent frame.
    pub fn samples_per_frame(&self) -> usize {
        self.spf
    }

    pub fn new_state(&self) -> DecoderState {
        let ctx = self.cfg.kernel - 1;
        DecoderState {
            low_tails: (0..self.cfg.n_low).map(|_| vec![0.0; ctx * self.cfg.d_low]).collect(),
            high_tails: (0..self.cfg.n_high).map(|_| vec![0.0; ctx * self.cfg.d_high]).collect(),
            ola_tail: vec![0.0; self.cfg.ola_tail()],
            col_idx: 0,
        }
    }

    pub fn new_scratch(&self) -> DecoderScratch {
        let cfg = &self.cfg;
        let d = cfg.d_low.max(cfg.d_high);
        let uf = self.uf;
        let n_bins = cfg.n_bins();
        DecoderScratch {
            x_low: vec![0.0; cfg.d_low],
            x_high: vec![0.0; uf * cfg.d_high],
            xc: vec![0.0; (cfg.kernel - 1 + uf) * d],
            h: vec![0.0; uf * d],
            rows: vec![0.0; uf * d],
            ff: vec![0.0; uf * cfg.expand * d],
            pw2: vec![0.0; uf * d],
            hn: vec![0.0; cfg.d_high],
            ho: vec![0.0; 2 * n_bins],
            cosb: vec![0.0; n_bins],
            sinb: vec![0.0; n_bins],
            spec: vec![(0.0, 0.0); n_bins],
            fft_buf: vec![(0.0, 0.0); cfg.n_fft],
            frame: vec![0.0; cfg.n_fft],
            chunk: vec![0.0; self.spf + cfg.ola_tail()],
        }
    }

    /// Window-power envelope at absolute sample position `p`.
    #[inline]
    fn env_at(&self, p: usize) -> f32 {
        let tail = self.cfg.ola_tail();
        if p < tail {
            self.env_start[p]
        } else {
            self.env_inner[p % self.cfg.hop]
        }
    }

    /// One causal ConvNeXt block over `t` frames of width `d`, in place.
    fn convnext(&self, bw: &ConvNeXtWeights, x: &mut [f32], t: usize, d: usize, tail: &mut [f32], sc: &mut DecoderScratch) {
        let k = self.cfg.kernel;
        let ctx = (k - 1) * d;
        debug_assert_eq!(x.len(), t * d);
        let xc = &mut sc.xc[..ctx + t * d];
        xc[..ctx].copy_from_slice(tail);
        xc[ctx..].copy_from_slice(x);
        tail.copy_from_slice(&xc[t * d..t * d + ctx]);
        let h = &mut sc.h[..t * d];
        ops::conv1d_depthwise_nlc(xc, &bw.dw_w, &bw.dw_b, d, k, k - 1 + t, h);
        let e = self.cfg.expand * d;
        let rows = &mut sc.rows[..t * d];
        let ff = &mut sc.ff[..t * e];
        let pw2 = &mut sc.pw2[..t * d];
        for r in 0..t {
            ops::layer_norm(&h[r * d..(r + 1) * d], &bw.norm_w, &bw.norm_b, d, 1, LN_EPS, &mut rows[r * d..(r + 1) * d]);
        }
        // All t rows through the pointwise layers at once: the block
        // weights are streamed from memory once per call, not once per row.
        bw.pw1.gemm(rows, t, ff);
        for r in 0..t {
            ops::add_inplace(&mut ff[r * e..(r + 1) * e], &bw.pw1_b);
        }
        ops::gelu_inplace(ff);
        bw.pw2.gemm(ff, t, pw2);
        for r in 0..t {
            ops::add_inplace(&mut pw2[r * d..(r + 1) * d], &bw.pw2_b);
            ops::add_mul_inplace(&mut x[r * d..(r + 1) * d], &pw2[r * d..(r + 1) * d], &bw.gamma);
        }
    }

    /// Spectral head of one column → windowed time frame in `sc.frame[..win]`.
    fn head_column(&self, col: &[f32], sc: &mut DecoderScratch) {
        let cfg = &self.cfg;
        let n_bins = cfg.n_bins();
        ops::layer_norm(col, &self.w.head_norm_w, &self.w.head_norm_b, cfg.d_high, 1, LN_EPS, &mut sc.hn);
        self.w.head.gemv(&sc.hn, &mut sc.ho);
        ops::add_inplace(&mut sc.ho, &self.w.head_b);
        let (mag, phase) = sc.ho.split_at_mut(n_bins);
        ops::exp_clamp_inplace(mag, cfg.logmag_min as f32, cfg.logmag_max as f32);
        ops::sincos_into(phase, &mut sc.cosb, &mut sc.sinb);
        for b in 0..n_bins {
            sc.spec[b] = (mag[b] * sc.cosb[b], mag[b] * sc.sinb[b]);
        }
        // DC and Nyquist are real.
        sc.spec[0].1 = 0.0;
        sc.spec[n_bins - 1].1 = 0.0;
        irfft_real(&self.fft, &sc.spec, &mut sc.fft_buf, &mut sc.frame);
        for i in 0..cfg.win {
            sc.frame[i] *= self.hann[i];
        }
    }

    /// Decode one latent frame (`latent_dim` values) into
    /// `samples_per_frame()` samples, advancing the stream state.
    pub fn stream_step(&self, latent: &[f32], st: &mut DecoderState, sc: &mut DecoderScratch, out: &mut [f32]) -> Result<()> {
        let cfg = &self.cfg;
        if latent.len() != cfg.latent_dim {
            return Err(Error::Invalid(format!(
                "decoder input has {} values, expected {}",
                latent.len(),
                cfg.latent_dim
            )));
        }
        if out.len() != self.spf {
            return Err(Error::Invalid(format!(
                "decoder output buffer has {} samples, expected {}",
                out.len(),
                self.spf
            )));
        }
        let uf = self.uf;
        let (d_low, d_high) = (cfg.d_low, cfg.d_high);

        let mut x_low = std::mem::take(&mut sc.x_low);
        self.w.in_proj.gemv(latent, &mut x_low);
        ops::add_inplace(&mut x_low, &self.w.in_proj_b);
        for (bw, tail) in self.w.low.iter().zip(st.low_tails.iter_mut()) {
            self.convnext(bw, &mut x_low, 1, d_low, tail, sc);
        }
        let mut x_high = std::mem::take(&mut sc.x_high);
        self.w.up.gemv(&x_low, &mut x_high);
        ops::add_inplace(&mut x_high, &self.w.up_b);
        sc.x_low = x_low;
        for (bw, tail) in self.w.high.iter().zip(st.high_tails.iter_mut()) {
            self.convnext(bw, &mut x_high, uf, d_high, tail, sc);
        }

        // Overlap-add of the uf windowed columns into one chunk, then
        // merge with the carried tail and normalise by the envelope.
        let hop = cfg.hop;
        let win = cfg.win;
        let tail_len = cfg.ola_tail();
        let n_emit = self.spf;
        let mut chunk = std::mem::take(&mut sc.chunk);
        chunk.iter_mut().for_each(|v| *v = 0.0);
        for u in 0..uf {
            self.head_column(&x_high[u * d_high..(u + 1) * d_high], sc);
            ops::add_inplace(&mut chunk[u * hop..u * hop + win], &sc.frame[..win]);
        }
        sc.x_high = x_high;
        for i in 0..tail_len {
            chunk[i] += st.ola_tail[i];
        }
        st.ola_tail.copy_from_slice(&chunk[n_emit..n_emit + tail_len]);
        let start = st.col_idx * hop;
        for i in 0..n_emit {
            out[i] = chunk[i] / self.env_at(start + i);
        }
        sc.chunk = chunk;
        st.col_idx += uf;
        Ok(())
    }

    /// Offline decode of `[T × latent_dim]` frames → `T · samples_per_frame`
    /// samples (no output-delay handling; see `Linnet::decode`).
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let ld = self.cfg.latent_dim;
        if latents.len() % ld != 0 {
            return Err(Error::Invalid(format!(
                "latents length {} is not a multiple of {ld}",
                latents.len()
            )));
        }
        let t = latents.len() / ld;
        let mut st = self.new_state();
        let mut sc = self.new_scratch();
        let mut out = vec![0.0f32; t * self.spf];
        for j in 0..t {
            self.stream_step(&latents[j * ld..(j + 1) * ld], &mut st, &mut sc, &mut out[j * self.spf..(j + 1) * self.spf])?;
        }
        Ok(out)
    }
}

/// Folded `hann²` envelope split into the start ramp (`win − hop`
/// positions with fewer than `win / hop` overlaps) and one period of the
/// steady state (position `p ≥ win − hop` depends only on `p % hop`).
fn build_envelope(hann: &[f32], win: usize, hop: usize) -> (Vec<f32>, Vec<f32>) {
    let w2: Vec<f32> = hann.iter().map(|v| v * v).collect();
    let tail = win - hop;
    let mut env_start = vec![0.0f32; tail];
    for (p, e) in env_start.iter_mut().enumerate() {
        let mut c = 0;
        while c * hop <= p {
            *e += w2[p - c * hop];
            c += 1;
        }
        *e = e.max(ENV_FLOOR);
    }
    let mut env_inner = vec![0.0f32; hop];
    for (r, e) in env_inner.iter_mut().enumerate() {
        // Sum in increasing column order: largest window offset first.
        let mut off = r + (win / hop - 1) * hop;
        loop {
            *e += w2[off];
            if off < hop {
                break;
            }
            off -= hop;
        }
        *e = e.max(ENV_FLOOR);
    }
    (env_start, env_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_matches_direct_fold() {
        let (win, hop) = (960, 240);
        let hann = hann_periodic(win);
        let (start, inner) = build_envelope(&hann, win, hop);
        let cols = 12;
        let len = (cols - 1) * hop + win;
        let mut direct = vec![0.0f32; len];
        for c in 0..cols {
            for i in 0..win {
                direct[c * hop + i] += hann[i] * hann[i];
            }
        }
        for p in 0..(win - hop) {
            assert!((direct[p].max(ENV_FLOOR) - start[p]).abs() < 1e-6, "p={p}");
        }
        for p in (win - hop)..(cols - 1) * hop {
            assert!((direct[p] - inner[p % hop]).abs() < 1e-5, "p={p}");
        }
    }
}
