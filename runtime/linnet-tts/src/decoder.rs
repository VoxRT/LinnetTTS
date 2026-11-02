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
//!
//! On the model's thread pool the pointwise products are split over
//! output rows, the depthwise convolution and LayerNorm over frames and
//! the spectral columns (exp, sincos, iFFT, window) over columns; each
//! value is still produced by one thread in the same order, so the thread
//! count does not change the output either.

use linnet_ops::fft::{hann_periodic, irfft_real, FftPlan};
use linnet_ops::ops;

use crate::config::DecoderConfig;
use crate::error::{Error, Result};
use crate::pool::{SendPtr, ThreadPool};
use crate::weights::{Act, ConvNeXtWeights, DecoderWeights};

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

/// Per-column buffers of the spectral head (one set per STFT column so
/// the columns can run concurrently).
struct ColScratch {
    cosb: Vec<f32>,
    sinb: Vec<f32>,
    spec: Vec<(f32, f32)>,
    fft_buf: Vec<(f64, f64)>,
    frame: Vec<f32>,
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
    cols: Vec<ColScratch>,
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
            cols: (0..t_max)
                .map(|_| ColScratch {
                    cosb: vec![0.0; n_bins],
                    sinb: vec![0.0; n_bins],
                    spec: vec![(0.0, 0.0); n_bins],
                    fft_buf: vec![(0.0, 0.0); cfg.n_fft],
                    frame: vec![0.0; cfg.n_fft],
                })
                .collect(),
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
    fn convnext(&self, bw: &ConvNeXtWeights, x: &mut [f32], t: usize, d: usize, tail: &mut [f32], sc: &mut DecoderScratch, pool: &ThreadPool) {
        let k = self.cfg.kernel;
        let ctx = (k - 1) * d;
        debug_assert_eq!(x.len(), t * d);
        let xc = &mut sc.xc[..ctx + t * d];
        xc[..ctx].copy_from_slice(tail);
        xc[ctx..].copy_from_slice(x);
        tail.copy_from_slice(&xc[t * d..t * d + ctx]);
        let h = &mut sc.h[..t * d];
        let rows = &mut sc.rows[..t * d];
        // Depthwise convolution and LayerNorm, one frame per chunk: the
        // convolution over a window of `k` input frames is exactly one
        // frame of the batched call (sequential taps per channel).
        {
            let xc: &[f32] = xc;
            let hp = SendPtr::new(h);
            let rp = SendPtr::new(rows);
            pool.run(t, &|r| {
                // SAFETY: frame r owns row r of h and of rows.
                let h_r = unsafe { hp.slice(r * d, d) };
                let row = unsafe { rp.slice(r * d, d) };
                ops::conv1d_depthwise_nlc(&xc[r * d..(r + k) * d], &bw.dw_w, &bw.dw_b, d, k, k, h_r);
                ops::layer_norm(h_r, &bw.norm_w, &bw.norm_b, d, 1, LN_EPS, row);
            });
        }
        let e = self.cfg.expand * d;
        let ff = &mut sc.ff[..t * e];
        let pw2 = &mut sc.pw2[..t * d];
        // All t rows through the pointwise layers at once: the block
        // weights are streamed from memory once per call, not once per row.
        bw.pw1.gemm_par(pool, rows, t, ff, Some(&bw.pw1_b), Act::Gelu);
        bw.pw2.gemm_par(pool, ff, t, pw2, Some(&bw.pw2_b), Act::None);
        for r in 0..t {
            ops::add_mul_inplace(&mut x[r * d..(r + 1) * d], &pw2[r * d..(r + 1) * d], &bw.gamma);
        }
    }

    /// Spectral head of `n_cols` columns (`x` is `n_cols × d_high`):
    /// LayerNorm per column, one GEMM for all columns, bias. The result
    /// (`n_cols × 2·n_bins`: log-magnitudes then phases) is left in
    /// `sc.ho`.
    fn head_rows(&self, x: &[f32], n_cols: usize, sc: &mut DecoderScratch, pool: &ThreadPool) {
        let d = self.cfg.d_high;
        let nb2 = 2 * self.cfg.n_bins();
        debug_assert_eq!(x.len(), n_cols * d);
        let hn = &mut sc.hn[..n_cols * d];
        for c in 0..n_cols {
            ops::layer_norm(&x[c * d..(c + 1) * d], &self.w.head_norm_w, &self.w.head_norm_b, d, 1, LN_EPS, &mut hn[c * d..(c + 1) * d]);
        }
        let ho = &mut sc.ho[..n_cols * nb2];
        self.w.head.gemm_par(pool, hn, n_cols, ho, Some(&self.w.head_b), Act::None);
    }

    /// Columns `0..n_cols` of `sc.ho` → windowed time frames in
    /// `sc.cols[c].frame[..win]`, one column per chunk.
    fn column_frames(&self, n_cols: usize, sc: &mut DecoderScratch, pool: &ThreadPool) {
        let cfg = &self.cfg;
        let n_bins = cfg.n_bins();
        let hop = SendPtr::new(&mut sc.ho);
        let cp = SendPtr::new(&mut sc.cols);
        pool.run(n_cols, &|col| {
            // SAFETY: column col owns row col of ho and sc.cols[col].
            let (mag, phase) = unsafe { hop.slice(col * 2 * n_bins, 2 * n_bins) }.split_at_mut(n_bins);
            let c = &mut unsafe { cp.slice(col, 1) }[0];
            ops::exp_clamp_inplace(mag, cfg.logmag_min as f32, cfg.logmag_max as f32);
            ops::sincos_into(phase, &mut c.cosb, &mut c.sinb);
            for b in 0..n_bins {
                c.spec[b] = (mag[b] * c.cosb[b], mag[b] * c.sinb[b]);
            }
            // DC and Nyquist are real.
            c.spec[0].1 = 0.0;
            c.spec[n_bins - 1].1 = 0.0;
            irfft_real(&self.fft, &c.spec, &mut c.fft_buf, &mut c.frame);
            for i in 0..cfg.win {
                c.frame[i] *= self.hann[i];
            }
        });
    }

    /// Decode one latent frame (`latent_dim` values) into
    /// `samples_per_frame()` samples, advancing the stream state.
    pub fn stream_step(&self, latent: &[f32], st: &mut DecoderState, sc: &mut DecoderScratch, out: &mut [f32], pool: &ThreadPool) -> Result<()> {
        self.stream_steps(latent, st, sc, out, pool)
    }

    /// Decode `n` consecutive latent frames (`n × latent_dim` values,
    /// `1 ≤ n ≤ sc.max_frames()`) into `n · samples_per_frame()` samples,
    /// advancing the stream state. Bit-identical to `n` calls of
    /// [`VocosDecoder::stream_step`]; the weights are read once.
    pub fn stream_steps(&self, latents: &[f32], st: &mut DecoderState, sc: &mut DecoderScratch, out: &mut [f32], pool: &ThreadPool) -> Result<()> {
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
            self.w.in_proj.gemm_par(pool, latents, nf, xl, Some(&self.w.in_proj_b), Act::None);
            for (bw, tail) in self.w.low.iter().zip(st.low_tails.iter_mut()) {
                self.convnext(bw, xl, nf, d_low, tail, sc, pool);
            }
        }
        let mut x_high = std::mem::take(&mut sc.x_high);
        {
            let xh = &mut x_high[..n_cols * d_high];
            self.w.up.gemm_par(pool, &x_low[..nf * d_low], nf, xh, Some(&self.w.up_b), Act::None);
            for (bw, tail) in self.w.high.iter().zip(st.high_tails.iter_mut()) {
                self.convnext(bw, xh, n_cols, d_high, tail, sc, pool);
            }
            self.head_rows(xh, n_cols, sc, pool);
        }
        sc.x_low = x_low;
        sc.x_high = x_high;
        self.column_frames(n_cols, sc, pool);

        // Overlap-add one frame (uf windowed columns) at a time, in the
        // original order: merge with the carried tail, normalise by the
        // envelope.
        let hop = cfg.hop;
        let win = cfg.win;
        let tail_len = cfg.ola_tail();
        let n_emit = self.spf;
        let chunk = &mut sc.chunk;
        for f in 0..nf {
            chunk.iter_mut().for_each(|v| *v = 0.0);
            for u in 0..uf {
                ops::add_inplace(&mut chunk[u * hop..u * hop + win], &sc.cols[f * uf + u].frame[..win]);
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
        Ok(())
    }

    /// Offline decode of `[T × latent_dim]` frames → `T · samples_per_frame`
    /// samples (no output-delay handling; see `Linnet::decode`), taking up
    /// to `frames_per_call` frames through the network at a time (the
    /// result does not depend on this number).
    pub fn decode(&self, latents: &[f32], frames_per_call: usize, pool: &ThreadPool) -> Result<Vec<f32>> {
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
            self.stream_steps(&latents[j * ld..(j + nf) * ld], &mut st, &mut sc, &mut out[j * self.spf..(j + nf) * self.spf], pool)?;
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
    /// decode with any batch size — on one thread or several — must
    /// produce the same bytes.
    #[test]
    fn batched_frames_and_threads_are_bit_identical_to_single_frames() {
        let one_thread = ThreadPool::new(1);
        let pools = [ThreadPool::new(1), ThreadPool::new(2), ThreadPool::new(3), ThreadPool::new(5)];
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
                dec.stream_step(&latents[j * ld..(j + 1) * ld], &mut st1, &mut sc1, &mut one[j * spf..(j + 1) * spf], &one_thread).unwrap();
            }
            assert!(one.iter().all(|v| v.is_finite()));

            for pool in &pools {
                for per_call in [1usize, 2, 3] {
                    let mut st = dec.new_state();
                    let mut sc = dec.new_scratch(per_call);
                    let mut got = vec![0.0f32; t * spf];
                    let mut j = 0;
                    while j < t {
                        let nf = per_call.min(t - j);
                        dec.stream_steps(&latents[j * ld..(j + nf) * ld], &mut st, &mut sc, &mut got[j * spf..(j + nf) * spf], pool).unwrap();
                        j += nf;
                    }
                    assert_eq!(bits(&one), bits(&got), "half={half} threads={} frames per call {per_call}", pool.threads());
                }
                for batch in [1usize, 2, 4, 7, 9] {
                    let off = dec.decode(&latents, batch, pool).unwrap();
                    assert_eq!(bits(&one), bits(&off), "half={half} threads={} offline batch {batch}", pool.threads());
                }
            }
        }
    }

    #[test]
    fn stream_steps_rejects_bad_sizes() {
        let dec = tiny_decoder(false);
        let pool = ThreadPool::new(1);
        let ld = dec.config().latent_dim;
        let spf = dec.samples_per_frame();
        let mut st = dec.new_state();
        let mut sc = dec.new_scratch(2);
        let lat = vec![0.0f32; 3 * ld];
        let mut out = vec![0.0f32; 3 * spf];
        assert!(dec.stream_steps(&lat, &mut st, &mut sc, &mut out, &pool).is_err(), "more frames than the scratch holds");
        assert!(dec.stream_steps(&lat[..ld + 1], &mut st, &mut sc, &mut out[..spf], &pool).is_err(), "partial frame");
        assert!(dec.stream_steps(&lat[..2 * ld], &mut st, &mut sc, &mut out[..spf], &pool).is_err(), "output too small");
        assert!(dec.stream_steps(&lat[..2 * ld], &mut st, &mut sc, &mut out[..2 * spf], &pool).is_ok());
    }
}
