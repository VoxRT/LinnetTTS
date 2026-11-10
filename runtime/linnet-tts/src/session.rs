//! The assembled model and the streaming synthesis session.
//!
//! A [`Session`] borrows a [`Linnet`] and owns everything that changes
//! during one utterance: the KV caches of the conditional and the
//! unconditional stream, the decoder state, the noise generator and the
//! output-delay bookkeeping. One step = one 48-dim latent = two codec
//! frames = 1920 samples; the first chunk is shorter by `output_delay`
//! samples and the session ends with `output_delay` zeros, so that the
//! concatenated chunks equal the offline decode of the same latents.
//!
//! The prefix starts with the voice (`[bos_voice | voice_proj(voice)]`,
//! `1 + frames` positions) and the attention is causal, so the trunk
//! state of those positions does not depend on the text. A session keeps
//! it from its first `prime` on, and re-priming runs only the text rows;
//! a [`VoiceState`] holds the same rows outside any session, so that a
//! session created from it skips the voice part of its first prime too.
//! Both paths produce the same bits as a full prime: every value of the
//! trunk is one dot product in a fixed order, whatever the row count.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use linnet_ops::Xoshiro256StarStar;

use crate::config::ModelConfig;
use crate::decoder::{DecoderScratch, DecoderState, VocosDecoder};
use crate::error::{Error, Result};
use crate::lm::{FlowLm, LmScratch, PrefixKv, TrunkState};
use crate::pool::{auto_threads, ThreadPool};
use crate::safetensors::SafeTensors;
use crate::tokenizer::Tokenizer;
use crate::voice::VoiceLatents;
use crate::weights;

/// Sampling parameters of one utterance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthParams {
    /// Variance of the flow start noise: `x0 ~ N(0, temp)`. `0` is deterministic.
    pub temp: f32,
    /// Classifier-free guidance weight; `1` disables the unconditional stream.
    pub cfg: f32,
    /// Upper bound on LM steps (80 ms each).
    pub max_frames: usize,
    /// Stop when the end-of-speech logit exceeds this value.
    pub eos_threshold: f32,
    /// Seed of the noise generator.
    pub seed: u64,
    /// Threads used by the model's pool while this session runs: `1` =
    /// single-threaded, `0` = automatic (the fast cores, at most 4; see
    /// [`crate::auto_threads`]). Creating a session applies the value to
    /// the model ([`Linnet::set_threads`]). The output never depends on
    /// it.
    pub threads: usize,
}

impl Default for SynthParams {
    fn default() -> Self {
        SynthParams {
            temp: 0.3,
            cfg: 2.0,
            max_frames: 300,
            eos_threshold: -1.0,
            seed: 0,
            threads: 0,
        }
    }
}

/// The part of a session's trunk state that depends on the voice and the
/// model but not on the text: the KV rows of the voice prefix
/// (`1 + frames` positions of the conditional stream), the two-row
/// prefix of the unconditional stream and the latter's output. Computed
/// once per voice by [`Linnet::voice_state`]; a session created with
/// [`Session::with_voice_state`] starts from a copy and its first prime
/// runs only the text rows, like every re-prime of an existing session.
/// About 2 MB for a 76-frame voice (f32), in memory only.
pub struct VoiceState {
    model_id: u64,
    frames: usize,
    dim: usize,
    cond: PrefixKv,
    null: PrefixKv,
    z_null: Vec<f32>,
}

impl VoiceState {
    /// Frames of the voice prompt.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Width of the voice latents.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Positions of the prefix a session skips when it starts from this
    /// state (`1 + frames`).
    pub fn prefix_len(&self) -> usize {
        self.cond.len()
    }

    /// True when `model` is the instance that computed this state (the
    /// only one it can be used with).
    pub fn is_for(&self, model: &Linnet) -> bool {
        self.model_id == model.id
    }
}

/// Identity of a loaded model within the process, so that a
/// [`VoiceState`] is never used with a model other than the one that
/// computed it.
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

/// Weights, tokenizer and derived constants. Immutable apart from the
/// thread setting, shareable between threads; every other mutable piece
/// lives in a [`Session`].
pub struct Linnet {
    id: u64,
    cfg: ModelConfig,
    lm: FlowLm,
    decoder: VocosDecoder,
    tokenizer: Tokenizer,
    /// One persistent pool per model; replaced by [`Linnet::set_threads`].
    /// Sessions take an `Arc` per call, so a replaced pool finishes the
    /// step it is in before its workers are joined.
    pool: Mutex<Arc<ThreadPool>>,
}

impl Linnet {
    /// Load the safetensors weights and the SentencePiece model.
    pub fn load(weights: impl AsRef<std::path::Path>, tokenizer: impl AsRef<std::path::Path>) -> Result<Self> {
        let st = SafeTensors::open(weights)?;
        let tok = Tokenizer::load(tokenizer)?;
        Self::from_parts(st, tok)
    }

    /// Same as [`Linnet::load`] from in-memory images.
    pub fn from_bytes(weights: Vec<u8>, tokenizer: &[u8]) -> Result<Self> {
        let st = SafeTensors::from_bytes(weights)?;
        let tok = Tokenizer::from_model_bytes(tokenizer)?;
        Self::from_parts(st, tok)
    }

    fn from_parts(st: SafeTensors, tokenizer: Tokenizer) -> Result<Self> {
        let cfg = ModelConfig::from_safetensors(&st)?;
        if tokenizer.vocab_size() != cfg.lm.vocab_size {
            return Err(Error::Shape(format!(
                "tokenizer has {} pieces, the model expects {}",
                tokenizer.vocab_size(),
                cfg.lm.vocab_size
            )));
        }
        let lm = FlowLm::new(cfg.lm.clone(), weights::load_lm(&st, &cfg.lm)?);
        let dec = VocosDecoder::new(
            cfg.decoder.clone(),
            weights::load_decoder(&st, &cfg.decoder, cfg.sample_rate)?,
            cfg.sample_rate,
        );
        Ok(Linnet {
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            cfg,
            lm,
            decoder: dec,
            tokenizer,
            pool: Mutex::new(Arc::new(ThreadPool::new(auto_threads()))),
        })
    }

    /// The pool a step runs on.
    pub(crate) fn pool(&self) -> Arc<ThreadPool> {
        Arc::clone(&self.pool.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Threads the model currently uses (1 = single-threaded).
    pub fn threads(&self) -> usize {
        self.pool().threads()
    }

    /// Use `threads` threads from now on: `1` single-threaded (no pool, no
    /// synchronisation), `0` automatic ([`auto_threads`]), larger values
    /// capped at [`crate::pool::MAX_THREADS`]. Results are identical for
    /// every value. A no-op when the count does not change.
    pub fn set_threads(&self, threads: usize) {
        let want = if threads == 0 { auto_threads() } else { threads };
        let mut slot = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if slot.threads() != want.clamp(1, crate::pool::MAX_THREADS) {
            *slot = Arc::new(ThreadPool::new(want));
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    pub fn lm(&self) -> &FlowLm {
        &self.lm
    }

    pub fn decoder(&self) -> &VocosDecoder {
        &self.decoder
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    /// Samples per LM step (two codec frames).
    pub fn samples_per_step(&self) -> usize {
        self.decoder.samples_per_frame() * (self.cfg.lm.latent_dim / self.cfg.decoder.latent_dim)
    }

    /// Latent width of one LM step.
    pub fn latent_dim(&self) -> usize {
        self.cfg.lm.latent_dim
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Offline decode of `[T × latent_dim]` codec-space latents into
    /// `T · samples_per_step` samples with the output delay compensated
    /// (first `output_delay` samples dropped, as many zeros appended).
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        let _big = crate::pool::affinity::BigCoreGuard::enter();
        if latents.is_empty() {
            return Ok(Vec::new());
        }
        let frames_per_step = self.cfg.lm.latent_dim / self.cfg.decoder.latent_dim;
        let pool = self.pool();
        let mut wav = self.decoder.decode(latents, frames_per_step, &pool)?;
        let delay = self.cfg.output_delay.min(wav.len());
        wav.drain(..delay);
        wav.resize(wav.len() + delay, 0.0);
        Ok(wav)
    }

    /// Whole utterance in one call.
    pub fn synthesize(&self, text: &str, voice: &VoiceLatents, params: SynthParams) -> Result<Vec<f32>> {
        let mut s = Session::new(self, voice, params)?;
        s.prime(text)?;
        Ok(s.collect())
    }

    /// [`Linnet::synthesize`] from a prepared [`VoiceState`]: the same
    /// audio, without the voice part of the prime pass.
    pub fn synthesize_with_state(&self, text: &str, state: &VoiceState, params: SynthParams) -> Result<Vec<f32>> {
        let mut s = Session::with_voice_state(self, state, params)?;
        s.prime(text)?;
        Ok(s.collect())
    }

    /// Run the voice prefix and the unconditional prefix through the
    /// trunk once, for any number of sessions to start from
    /// ([`Session::with_voice_state`]). Costs about one prime pass over the
    /// voice; the result is tied to this model instance.
    pub fn voice_state(&self, voice: &VoiceLatents) -> Result<VoiceState> {
        let _big = crate::pool::affinity::BigCoreGuard::enter();
        let ld = self.cfg.lm.latent_dim;
        let d = self.cfg.lm.d_model;
        if voice.dim() != ld {
            return Err(Error::Invalid(format!(
                "voice latents have width {}, the model expects {ld}",
                voice.dim()
            )));
        }
        let lm = &self.lm;
        let pool = self.pool();
        let nv = 1 + voice.frames();
        let mut sc = lm.new_scratch(1, nv.max(2));
        let mut rows = vec![0.0; nv * d];
        lm.build_voice_rows(voice.latents(), &mut rows)?;
        let mut st = lm.new_state(nv);
        lm.prefill(&mut rows, nv, &mut st, &mut sc, &pool)?;
        let cond = st.export_prefix(nv);
        rows.clear();
        rows.resize(2 * d, 0.0);
        lm.build_null_prefix(&mut rows);
        let mut st_null = lm.new_state(2);
        lm.prefill(&mut rows, 2, &mut st_null, &mut sc, &pool)?;
        let null = st_null.export_prefix(2);
        let mut z_null = vec![0.0; d];
        lm.out_norm(&rows[d..], &mut z_null);
        Ok(VoiceState { model_id: self.id, frames: voice.frames(), dim: ld, cond, null, z_null })
    }
}

/// One utterance in progress.
pub struct Session<'m> {
    model: &'m Linnet,
    /// Voice latents, used by the first prime only; empty when the session
    /// started from a [`VoiceState`].
    voice: Vec<f32>,
    /// Positions of the voice prefix (`1 + frames`).
    nv: usize,
    /// The voice prefix (and, with guidance, the unconditional prefix and
    /// `z_null`) are in the trunk states; a prime runs only the text rows.
    voice_kept: bool,
    /// Output of the unconditional prefix, restored at each re-prime.
    z_null: Vec<f32>,
    params: SynthParams,
    n_streams: usize,
    states: Vec<TrunkState>,
    sc: LmScratch,
    dec_state: DecoderState,
    dec_sc: DecoderScratch,
    rng: Xoshiro256StarStar,
    z: Vec<f32>,
    z_rows: Vec<f32>,
    x0: Vec<f32>,
    latent_norm: Vec<f32>,
    latent_codec: Vec<f32>,
    xs: Vec<f32>,
    chunk: Vec<f32>,
    out: Vec<f32>,
    /// Prefix rows of `prime`; kept so re-priming does not allocate.
    prefix_rows: Vec<f32>,
    skip: usize,
    frames: usize,
    primed: bool,
    finished: bool,
    tail_pending: bool,
}

impl<'m> Session<'m> {
    /// Allocate a session for `voice`. Call [`Session::prime`] before
    /// stepping. The first prime runs the whole prefix; the voice part of
    /// its state is kept, so later primes of the same session run only
    /// the text rows.
    pub fn new(model: &'m Linnet, voice: &VoiceLatents, params: SynthParams) -> Result<Self> {
        let ld = model.cfg.lm.latent_dim;
        if voice.dim() != ld {
            return Err(Error::Invalid(format!(
                "voice latents have width {}, the model expects {ld}",
                voice.dim()
            )));
        }
        Self::new_inner(model, voice.latents().to_vec(), voice.frames(), params)
    }

    /// Allocate a session that starts from `state` ([`Linnet::voice_state`]
    /// of the same model): the first prime already runs only the text
    /// rows. The audio is identical to a session made with
    /// [`Session::new`] from the same voice.
    pub fn with_voice_state(model: &'m Linnet, state: &VoiceState, params: SynthParams) -> Result<Self> {
        if state.model_id != model.id {
            return Err(Error::Invalid("voice state was prepared for a different model".into()));
        }
        let mut s = Self::new_inner(model, Vec::new(), state.frames, params)?;
        s.states[0].import_prefix(&state.cond)?;
        if s.n_streams == 2 {
            s.states[1].import_prefix(&state.null)?;
            s.z_null.copy_from_slice(&state.z_null);
        }
        s.voice_kept = true;
        Ok(s)
    }

    fn new_inner(model: &'m Linnet, voice: Vec<f32>, frames: usize, params: SynthParams) -> Result<Self> {
        let cfg = &model.cfg;
        let ld = cfg.lm.latent_dim;
        let d = cfg.lm.d_model;
        if !(params.temp >= 0.0 && params.temp.is_finite()) {
            return Err(Error::Invalid(format!("temp {} must be finite and >= 0", params.temp)));
        }
        if !params.cfg.is_finite() {
            return Err(Error::Invalid("cfg must be finite".into()));
        }
        if params.max_frames == 0 {
            return Err(Error::Invalid("max_frames must be at least 1".into()));
        }
        model.set_threads(params.threads);
        let n_streams = if params.cfg == 1.0 { 1 } else { 2 };
        let cap = 2 + frames + params.max_frames + 2;
        let states = (0..n_streams).map(|_| model.lm.new_state(cap)).collect();
        let sps = model.samples_per_step();
        Ok(Session {
            model,
            voice,
            nv: 1 + frames,
            voice_kept: false,
            z_null: vec![0.0; d],
            params,
            n_streams,
            states,
            sc: model.lm.new_scratch(n_streams, cap),
            dec_state: model.decoder.new_state(),
            // The two codec frames of one step go through the decoder together.
            dec_sc: model.decoder.new_scratch(ld / cfg.decoder.latent_dim),
            rng: Xoshiro256StarStar::new(params.seed),
            z: vec![0.0; d],
            z_rows: vec![0.0; n_streams * d],
            x0: vec![0.0; ld],
            latent_norm: vec![0.0; ld],
            latent_codec: vec![0.0; ld],
            xs: vec![0.0; n_streams * d],
            chunk: vec![0.0; sps],
            out: vec![0.0; sps],
            prefix_rows: Vec::new(),
            skip: 0,
            frames: 0,
            primed: false,
            finished: false,
            tail_pending: false,
        })
    }

    /// Tokenize `text` and run the prefix through the trunk. Resets any
    /// previous utterance (the KV caches must fit the new prefix).
    pub fn prime(&mut self, text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Err(Error::Invalid("empty input: the text is blank".into()));
        }
        let tokens = self.model.tokenize(text);
        self.prime_tokens(&tokens)
    }

    /// Like [`Session::prime`] with ready-made token ids.
    pub fn prime_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::Invalid("empty input: the text produced no tokens".into()));
        }
        let _big = crate::pool::affinity::BigCoreGuard::enter();
        let lm = &self.model.lm;
        let d = self.model.cfg.lm.d_model;
        let tv = self.nv - 1;
        let prefix_len = lm.prefix_len(tokens.len(), tv);
        let cap = prefix_len + self.params.max_frames + 2;
        for st in &mut self.states {
            st.grow(cap); // keeps the voice prefix when it is there
        }
        if self.sc.m_capacity_scores() < cap {
            self.sc = lm.new_scratch(self.n_streams, cap);
        }
        self.dec_state.reset();
        self.rng = Xoshiro256StarStar::new(self.params.seed);
        self.skip = self.model.cfg.output_delay;
        self.frames = 0;
        self.finished = false;
        self.tail_pending = false;

        let pool = self.model.pool();
        let mut rows = std::mem::take(&mut self.prefix_rows);
        rows.clear();
        if self.voice_kept {
            // The voice positions (and the unconditional prefix) are in
            // the states: drop the previous text and run the new one. The
            // K/V of every row is one fixed-order dot product per value,
            // so the result equals a full prime bit for bit.
            let t = tokens.len() + 1;
            rows.resize(t * d, 0.0);
            lm.build_text_rows(tokens, &mut rows)?;
            self.states[0].truncate(self.nv);
            lm.prefill(&mut rows, t, &mut self.states[0], &mut self.sc, &pool)?;
            lm.out_norm(&rows[(t - 1) * d..], &mut self.z_rows[..d]);
            if self.n_streams == 2 {
                self.states[1].truncate(2);
                self.z_rows[d..].copy_from_slice(&self.z_null);
            }
        } else {
            for st in &mut self.states {
                st.reset();
            }
            rows.resize(prefix_len * d, 0.0);
            lm.build_prefix(tokens, &self.voice, &mut rows)?;
            lm.prefill(&mut rows, prefix_len, &mut self.states[0], &mut self.sc, &pool)?;
            lm.out_norm(&rows[(prefix_len - 1) * d..], &mut self.z_rows[..d]);
            if self.n_streams == 2 {
                rows.clear();
                rows.resize(2 * d, 0.0);
                lm.build_null_prefix(&mut rows);
                lm.prefill(&mut rows, 2, &mut self.states[1], &mut self.sc, &pool)?;
                lm.out_norm(&rows[d..], &mut self.z_rows[d..]);
                self.z_null.copy_from_slice(&self.z_rows[d..]);
            }
            self.voice_kept = true;
        }
        self.prefix_rows = rows;
        self.mix_z();
        self.primed = true;
        Ok(())
    }

    /// Every remaining chunk, concatenated.
    pub fn collect(&mut self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.params.max_frames * self.model.samples_per_step());
        while let Some(chunk) = self.step() {
            out.extend_from_slice(chunk);
        }
        out
    }

    /// `z = z_null + cfg · (z_cond − z_null)` (or `z_cond` without guidance).
    fn mix_z(&mut self) {
        let d = self.model.cfg.lm.d_model;
        if self.n_streams == 1 {
            self.z.copy_from_slice(&self.z_rows[..d]);
        } else {
            let (zc, zn) = self.z_rows.split_at(d);
            let w = self.params.cfg;
            for i in 0..d {
                self.z[i] = zn[i] + w * (zc[i] - zn[i]);
            }
        }
    }

    /// Produce the next chunk of audio, or `None` once the utterance is
    /// over. Chunk lengths: `samples_per_step − output_delay` for the
    /// first, `samples_per_step` afterwards, `output_delay` zeros last.
    pub fn step(&mut self) -> Option<&[f32]> {
        if !self.primed || self.finished {
            return None;
        }
        if !self.tail_pending {
            let std = self.params.temp.sqrt();
            if std > 0.0 {
                let mut x0 = std::mem::take(&mut self.x0);
                self.rng.fill_gaussian(&mut x0, std);
                self.x0 = x0;
            } else {
                self.x0.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        self.advance()
    }

    /// Like [`Session::step`] with caller-provided start noise (`latent_dim`
    /// values, already scaled by `sqrt(temp)`); the internal generator is
    /// not consulted. Used for cross-implementation parity checks.
    pub fn step_with_noise(&mut self, noise: &[f32]) -> Option<&[f32]> {
        if !self.primed || self.finished {
            return None;
        }
        if !self.tail_pending {
            if noise.len() != self.x0.len() {
                return None;
            }
            self.x0.copy_from_slice(noise);
        }
        self.advance()
    }

    fn advance(&mut self) -> Option<&[f32]> {
        let _big = crate::pool::affinity::BigCoreGuard::enter();
        let model = self.model;
        let lm = &model.lm;
        let d = model.cfg.lm.d_model;

        if self.tail_pending {
            self.tail_pending = false;
            self.finished = true;
            let delay = model.cfg.output_delay;
            self.out[..delay].iter_mut().for_each(|v| *v = 0.0);
            return Some(&self.out[..delay]);
        }

        let pool = model.pool();
        // Sample, then decide whether this step is the end.
        lm.flow_sample(&self.z, &self.x0, &mut self.sc, &mut self.latent_norm, &pool);
        if lm.eos_logit(&self.z) > self.params.eos_threshold {
            return self.end();
        }
        lm.denormalize(&self.latent_norm, &mut self.latent_codec);
        // Both codec frames of the step in one call: the decoder weights
        // are streamed once; the output is the same as frame by frame.
        model
            .decoder
            .stream_steps(&self.latent_codec, &mut self.dec_state, &mut self.dec_sc, &mut self.chunk, &pool)
            .expect("decoder buffers are sized at construction");
        self.frames += 1;

        // Next step's conditioning, unless the frame budget is spent.
        if self.frames < self.params.max_frames {
            for r in 0..self.n_streams {
                lm.latent_feedback(&self.latent_norm, &mut self.xs[r * d..(r + 1) * d]);
            }
            lm.step(&mut self.xs, &mut self.states, &mut self.sc, &pool).expect("KV cache sized for max_frames");
            for r in 0..self.n_streams {
                lm.out_norm(&self.xs[r * d..(r + 1) * d], &mut self.z_rows[r * d..(r + 1) * d]);
            }
            self.mix_z();
        } else {
            self.tail_pending = true;
        }

        // Output delay: drop the first `output_delay` samples of the stream.
        let cut = self.skip.min(self.chunk.len());
        self.skip -= cut;
        let n = self.chunk.len() - cut;
        self.out[..n].copy_from_slice(&self.chunk[cut..]);
        Some(&self.out[..n])
    }

    fn end(&mut self) -> Option<&[f32]> {
        if self.frames == 0 {
            self.finished = true;
            return None;
        }
        self.tail_pending = true;
        self.advance()
    }

    /// Codec-space latent of the most recent step (`latent_dim` values).
    pub fn latent(&self) -> &[f32] {
        &self.latent_codec
    }

    /// LM steps emitted so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// True once the last chunk (the `output_delay` zeros) has been
    /// returned; the next [`Session::step`] returns `None`. Also true when
    /// the utterance ended without any chunk (EOS on the first step).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Samples of one full chunk (`samples_per_step` of the model).
    pub fn samples_per_step(&self) -> usize {
        self.model.samples_per_step()
    }

    pub fn params(&self) -> &SynthParams {
        &self.params
    }

    /// True once the voice prefix is in the trunk state (after the first
    /// prime, or from construction with [`Session::with_voice_state`]):
    /// the next prime runs only the text rows.
    pub fn voice_kept(&self) -> bool {
        self.voice_kept
    }
}
