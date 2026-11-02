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
//! streamed a few latent frames at a time ([`VocosDecoder::stream_steps`];
//! the session passes the two frames of one LM step); its state is the
//! left context of every depthwise convolution (`kernel − 1` frames, zeros
//! at start — the same as the offline left zero-padding) plus the
//! `win − hop` sample overlap-add tail and the absolute column index used
//! to slice the window-power envelope.
//!
//! Batching frames changes nothing numerically: every kernel the neural
//! part uses is either per-row (LayerNorm, biases, activations, layer
//! scale) or contractually bit-equal between its batched and row-wise
//! forms (the GEMMs equal stacked GEMVs, the depthwise convolution over
//! `T` frames equals `T` sliding-window steps), and the spectral head and
//! overlap-add still run one frame of `up_factor` columns at a time in the
//! original order. Offline decoding is the same loop, so offline and
//! streamed output agree bit for bit whatever the batching.

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

/// Reusable buffers for up to `max_frames` latent frames per call. Nothing
/// is allocated after construction.
pub struct DecoderScratch {
    max_frames: usize,
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

impl DecoderScratch {
    /// Largest number of latent frames one `stream_steps` call may take.
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
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

    /// Scratch for calls of up to `max_frames` latent frames (at least 1).
    pub fn new_scratch(&self, max_frames: usize) -> DecoderScratch {
        let cfg = &self.cfg;
        let max_frames = max_frames.max(1);
        let d = cfg.d_low.max(cfg.d_high);
        let uf = self.uf;
        let n_bins = cfg.n_bins();
        // Most rows any ConvNeXt block sees in one call.
        let t_max = max_frames * uf;
        DecoderScratch {
            max_frames,
            x_low: vec![0.0; max_frames * cfg.d_low],
            x_high: vec![0.0; t_max * cfg.d_high],
            xc: vec![0.0; (cfg.kernel - 1 + t_max) * d],
            h: vec![0.0; t_max * d],
            rows: vec![0.0; t_max * d],
            ff: vec![0.0; t_max * cfg.expand * d],
            pw2: vec![0.0; t_max * d],
            hn: vec![0.0; t_max * cfg.d_high],
            ho: vec![0.0; t_max * 2 * n_bins],
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

    /// Spectral head of `n_cols` columns (`x` is `n_cols × d_high`):
    /// LayerNorm per column, one GEMM for all columns, bias. The result
    /// (`n_cols × 2·n_bins`: log-magnitudes then phases) is left in
    /// `sc.ho`.
    fn head_rows(&self, x: &[f32], n_cols: usize, sc: &mut DecoderScratch) {
        let d = self.cfg.d_high;
        let nb2 = 2 * self.cfg.n_bins();
        debug_assert_eq!(x.len(), n_cols * d);
        let hn = &mut sc.hn[..n_cols * d];
        for c in 0..n_cols {
            ops::layer_norm(&x[c * d..(c + 1) * d], &self.w.head_norm_w, &self.w.head_norm_b, d, 1, LN_EPS, &mut hn[c * d..(c + 1) * d]);
        }
        let ho = &mut sc.ho[..n_cols * nb2];
        self.w.head.gemm(hn, n_cols, ho);
        for c in 0..n_cols {
            ops::add_inplace(&mut ho[c * nb2..(c + 1) * nb2], &self.w.head_b);
        }
    }

    /// Column `col` of `sc.ho` → windowed time frame in `sc.frame[..win]`.
    fn column_frame(&self, col: usize, sc: &mut DecoderScratch) {
        let cfg = &self.cfg;
        let n_bins = cfg.n_bins();
        let (mag, phase) = sc.ho[col * 2 * n_bins..(col + 1) * 2 * n_bins].split_at_mut(n_bins);
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
        self.stream_steps(latent, st, sc, out)
    }

    /// Decode `n` consecutive latent frames (`n × latent_dim` values,
    /// `1 ≤ n ≤ sc.max_frames()`) into `n · samples_per_frame()` samples,
    /// advancing the stream state. Bit-identical to `n` calls of
    /// [`VocosDecoder::stream_step`]; the weights are read once.
    pub fn stream_steps(&self, latents: &[f32], st: &mut DecoderState, sc: &mut DecoderScratch, out: &mut [f32]) -> Result<()> {
        let cfg = &self.cfg;
        let ld = cfg.latent_dim;
        let nf = latents.len() / ld;
        if nf == 0 || latents.len() != nf * ld || nf > sc.max_frames {
            return Err(Error::Invalid(format!(
                "decoder input has {} values, expected 1..={} frames of {ld}",
                latents.len(),
                sc.max_frames
            )));
        }
        if out.len() != nf * self.spf {
            return Err(Error::Invalid(format!(
                "decoder output buffer has {} samples, expected {}",
                out.len(),
                nf * self.spf
            )));
        }
        let uf = self.uf;
        let (d_low, d_high) = (cfg.d_low, cfg.d_high);
        let n_cols = nf * uf;

        // Neural part over all nf frames at once.
        let mut x_low = std::mem::take(&mut sc.x_low);
        {
            let xl = &mut x_low[..nf * d_low];
            self.w.in_proj.gemm(latents, nf, xl);
            for r in 0..nf {
                ops::add_inplace(&mut xl[r * d_low..(r + 1) * d_low], &self.w.in_proj_b);
            }
            for (bw, tail) in self.w.low.iter().zip(st.low_tails.iter_mut()) {
                self.convnext(bw, xl, nf, d_low, tail, sc);
            }
        }
        let mut x_high = std::mem::take(&mut sc.x_high);
        {
            let xh = &mut x_high[..n_cols * d_high];
            let ud = uf * d_high;
            self.w.up.gemm(&x_low[..nf * d_low], nf, xh);
            for r in 0..nf {
                ops::add_inplace(&mut xh[r * ud..(r + 1) * ud], &self.w.up_b);
            }
            for (bw, tail) in self.w.high.iter().zip(st.high_tails.iter_mut()) {
                self.convnext(bw, xh, n_cols, d_high, tail, sc);
            }
            self.head_rows(xh, n_cols, sc);
        }
        sc.x_low = x_low;
        sc.x_high = x_high;

        // Overlap-add one frame (uf windowed columns) at a time, in the
        // original order: merge with the carried tail, normalise by the
        // envelope.
        let hop = cfg.hop;
        let win = cfg.win;
        let tail_len = cfg.ola_tail();
        let n_emit = self.spf;
        let mut chunk = std::mem::take(&mut sc.chunk);
        for f in 0..nf {
            chunk.iter_mut().for_each(|v| *v = 0.0);
            for u in 0..uf {
                self.column_frame(f * uf + u, sc);
                ops::add_inplace(&mut chunk[u * hop..u * hop + win], &sc.frame[..win]);
            }
            for i in 0..tail_len {
                chunk[i] += st.ola_tail[i];
            }
            st.ola_tail.copy_from_slice(&chunk[n_emit..n_emit + tail_len]);
            let start = st.col_idx * hop;
            let out_f = &mut out[f * n_emit..(f + 1) * n_emit];
            for i in 0..n_emit {
                out_f[i] = chunk[i] / self.env_at(start + i);
            }
            st.col_idx += uf;
        }
        sc.chunk = chunk;
        Ok(())
    }

    /// Offline decode of `[T × latent_dim]` frames → `T · samples_per_frame`
    /// samples (no output-delay handling; see `Linnet::decode`), taking up
    /// to `frames_per_call` frames through the network at a time (the
    /// result does not depend on this number).
    pub fn decode(&self, latents: &[f32], frames_per_call: usize) -> Result<Vec<f32>> {
        let ld = self.cfg.latent_dim;
        if latents.len() % ld != 0 {
            return Err(Error::Invalid(format!(
                "latents length {} is not a multiple of {ld}",
                latents.len()
            )));
        }
        let batch = frames_per_call.max(1);
        let t = latents.len() / ld;
        let mut st = self.new_state();
        let mut sc = self.new_scratch(batch);
        let mut out = vec![0.0f32; t * self.spf];
        let mut j = 0;
        while j < t {
            let nf = batch.min(t - j);
            self.stream_steps(&latents[j * ld..(j + nf) * ld], &mut st, &mut sc, &mut out[j * self.spf..(j + nf) * self.spf])?;
            j += nf;
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
    use crate::weights::Mat;
    use linnet_ops::f16::f32_to_f16;

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

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn vecf(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| lcg(seed) * scale).collect()
    }

    fn mat(rows: usize, cols: usize, scale: f32, half: bool, seed: &mut u64) -> Mat {
        let w = vecf(rows * cols, scale, seed);
        if half {
            Mat::F16 { w: w.iter().map(|&v| f32_to_f16(v)).collect(), rows, cols }
        } else {
            Mat::F32 { w, rows, cols }
        }
    }

    fn block(d: usize, expand: usize, k: usize, half: bool, seed: &mut u64) -> ConvNeXtWeights {
        ConvNeXtWeights {
            dw_w: vecf(k * d, 0.3, seed),
            dw_b: vecf(d, 0.1, seed),
            norm_w: vecf(d, 0.5, seed).iter().map(|v| 1.0 + v).collect(),
            norm_b: vecf(d, 0.1, seed),
            pw1: mat(expand * d, d, 0.2, half, seed),
            pw1_b: vecf(expand * d, 0.1, seed),
            pw2: mat(d, expand * d, 0.2, half, seed),
            pw2_b: vecf(d, 0.1, seed),
            gamma: vecf(d, 0.5, seed),
        }
    }

    /// A small random decoder with the real frame geometry.
    fn tiny_decoder(half: bool) -> VocosDecoder {
        let cfg = DecoderConfig {
            latent_dim: 6,
            fps_x2: 50,
            d_low: 16,
            n_low: 2,
            d_high: 24,
            n_high: 3,
            expand: 3,
            kernel: 7,
            hop: 240,
            win: 960,
            n_fft: 1024,
            layer_scale_init: 1e-6,
            mag_bias_init: -4.0,
            logmag_min: -12.0,
            logmag_max: 4.605,
        };
        let mut seed = if half { 0x1234 } else { 0x5678 };
        let uf = cfg.up_factor(24_000);
        let nb = cfg.n_bins();
        let w = DecoderWeights {
            in_proj: mat(cfg.d_low, cfg.latent_dim, 0.5, half, &mut seed),
            in_proj_b: vecf(cfg.d_low, 0.1, &mut seed),
            low: (0..cfg.n_low).map(|_| block(cfg.d_low, cfg.expand, cfg.kernel, half, &mut seed)).collect(),
            up: mat(uf * cfg.d_high, cfg.d_low, 0.3, half, &mut seed),
            up_b: vecf(uf * cfg.d_high, 0.1, &mut seed),
            high: (0..cfg.n_high).map(|_| block(cfg.d_high, cfg.expand, cfg.kernel, half, &mut seed)).collect(),
            head_norm_w: vecf(cfg.d_high, 0.5, &mut seed).iter().map(|v| 1.0 + v).collect(),
            head_norm_b: vecf(cfg.d_high, 0.1, &mut seed),
            head: mat(2 * nb, cfg.d_high, 0.3, half, &mut seed),
            head_b: vecf(2 * nb, 1.0, &mut seed),
        };
        VocosDecoder::new(cfg, w, 24_000)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Streaming two frames per call, one frame per call and the offline
    /// decode with any batch size must produce the same bytes.
    #[test]
    fn batched_frames_are_bit_identical_to_single_frames() {
        for half in [false, true] {
            let dec = tiny_decoder(half);
            let ld = dec.config().latent_dim;
            let spf = dec.samples_per_frame();
            let t = 7;
            let mut seed = 99;
            let latents = vecf(t * ld, 1.0, &mut seed);

            let mut st1 = dec.new_state();
            let mut sc1 = dec.new_scratch(1);
            let mut one = vec![0.0f32; t * spf];
            for j in 0..t {
                dec.stream_step(&latents[j * ld..(j + 1) * ld], &mut st1, &mut sc1, &mut one[j * spf..(j + 1) * spf]).unwrap();
            }

            let mut st2 = dec.new_state();
            let mut sc2 = dec.new_scratch(2);
            let mut two = vec![0.0f32; t * spf];
            let mut j = 0;
            while j < t {
                let nf = 2.min(t - j);
                dec.stream_steps(&latents[j * ld..(j + nf) * ld], &mut st2, &mut sc2, &mut two[j * spf..(j + nf) * spf]).unwrap();
                j += nf;
            }
            assert_eq!(bits(&one), bits(&two), "half={half}: two frames per call");

            let mut st3 = dec.new_state();
            let mut sc3 = dec.new_scratch(3);
            let mut three = vec![0.0f32; t * spf];
            let mut j = 0;
            while j < t {
                let nf = 3.min(t - j);
                dec.stream_steps(&latents[j * ld..(j + nf) * ld], &mut st3, &mut sc3, &mut three[j * spf..(j + nf) * spf]).unwrap();
                j += nf;
            }
            assert_eq!(bits(&one), bits(&three), "half={half}: three frames per call");

            for batch in [1usize, 2, 4, 7, 9] {
                let off = dec.decode(&latents, batch).unwrap();
                assert_eq!(bits(&one), bits(&off), "half={half}: offline batch {batch}");
            }
            assert!(one.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn stream_steps_rejects_bad_sizes() {
        let dec = tiny_decoder(false);
        let ld = dec.config().latent_dim;
        let spf = dec.samples_per_frame();
        let mut st = dec.new_state();
        let mut sc = dec.new_scratch(2);
        let lat = vec![0.0f32; 3 * ld];
        let mut out = vec![0.0f32; 3 * spf];
        assert!(dec.stream_steps(&lat, &mut st, &mut sc, &mut out).is_err(), "more frames than the scratch holds");
        assert!(dec.stream_steps(&lat[..ld + 1], &mut st, &mut sc, &mut out[..spf]).is_err(), "partial frame");
        assert!(dec.stream_steps(&lat[..2 * ld], &mut st, &mut sc, &mut out[..spf]).is_err(), "output too small");
        assert!(dec.stream_steps(&lat[..2 * ld], &mut st, &mut sc, &mut out[..2 * spf]).is_ok());
    }
}
