//! Python extension module for the LinnetTTS runtime.
//!
//! The module is deliberately thin: it exposes the model, a voice and a
//! streaming session as Python classes and moves audio across the
//! boundary as little-endian float32 `bytes`. The pure-Python package
//! `linnet_tts` wraps those bytes in NumPy arrays and adds the
//! conveniences (iteration, `Voice.from_array`, keyword defaults).
//!
//! Long-running calls (`load`, `prime`, `step`, `synthesize`) release the
//! GIL, so one model may serve several sessions from several threads.

// `is_multiple_of` needs Rust 1.87 and the crate's MSRV is 1.85.
#![allow(clippy::manual_is_multiple_of)]

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use linnet_tts::{Error, Linnet, Session, SynthParams, VoiceLatents, VoiceState};

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Io(_) => PyIOError::new_err(e.to_string()),
        Error::Format(_) | Error::Shape(_) => PyValueError::new_err(e.to_string()),
        Error::Invalid(_) => PyValueError::new_err(e.to_string()),
    }
}

fn pcm_bytes<'py>(py: Python<'py>, pcm: &[f32]) -> Bound<'py, PyBytes> {
    PyBytes::new_with(py, pcm.len() * 4, |buf| {
        for (dst, s) in buf.chunks_exact_mut(4).zip(pcm) {
            dst.copy_from_slice(&s.to_le_bytes());
        }
        Ok(())
    })
    .expect("byte buffer")
}

fn params_from(temp: f32, cfg: f32, max_frames: u32, eos_threshold: f32, seed: u64, threads: u32) -> PyResult<SynthParams> {
    if max_frames == 0 {
        return Err(PyValueError::new_err("max_frames must be >= 1"));
    }
    Ok(SynthParams {
        temp,
        cfg,
        max_frames: max_frames as usize,
        eos_threshold,
        seed,
        threads: threads as usize,
    })
}

/// Loaded weights and tokenizer. Immutable; shared by sessions.
#[pyclass(name = "Model", module = "linnet_tts._native", frozen)]
pub struct PyModel {
    inner: Arc<Linnet>,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(py: Python<'_>, weights_path: &str, tokenizer_path: &str) -> PyResult<Self> {
        let (w, t) = (weights_path.to_owned(), tokenizer_path.to_owned());
        let model = py.detach(move || Linnet::load(w, t)).map_err(to_py)?;
        Ok(PyModel { inner: Arc::new(model) })
    }

    /// Output sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Samples produced per LM step (two codec frames).
    #[getter]
    fn samples_per_step(&self) -> usize {
        self.inner.samples_per_step()
    }

    /// Width of one LM latent (and of a voice row).
    #[getter]
    fn latent_dim(&self) -> usize {
        self.inner.latent_dim()
    }

    /// Delay compensated by the streaming session, in samples.
    #[getter]
    fn output_delay(&self) -> usize {
        self.inner.config().output_delay
    }

    /// SentencePiece token ids of `text`.
    fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenize(text)
    }

    /// Threads the model uses (1 = single-threaded).
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    /// Use `threads` threads from now on: 1 = single-threaded, 0 = automatic
    /// (the fast cores, at most 4). The audio is identical for every value.
    fn set_threads(&self, threads: u32) {
        self.inner.set_threads(threads as usize)
    }

    /// Load `[T, latent_dim]` float32 latents from a `.npy` file and prepare
    /// the voice for this model (`prepare_voice`).
    fn load_voice(&self, py: Python<'_>, path: &str) -> PyResult<PyVoice> {
        let p = path.to_owned();
        let v = py.detach(move || VoiceLatents::load(p)).map_err(to_py)?;
        if v.dim() != self.inner.latent_dim() {
            return Err(PyValueError::new_err(format!(
                "voice latents have width {}, the model expects {}",
                v.dim(),
                self.inner.latent_dim()
            )));
        }
        let voice = PyVoice::wrap(v);
        self.prepare_voice(py, &voice)?;
        Ok(voice)
    }

    /// Run `voice` through the model once and keep the result in the voice,
    /// so that sessions created from it for this model skip the voice part
    /// of the prime pass (the first chunk then costs the text rows plus one
    /// step). The audio is identical. Tied to this model instance (a voice
    /// holds one state: preparing it for another model replaces it); about
    /// 2 MB per voice.
    fn prepare_voice(&self, py: Python<'_>, voice: &PyVoice) -> PyResult<()> {
        let model = Arc::clone(&self.inner);
        let latents = Arc::clone(&voice.inner);
        let state = py.detach(move || model.voice_state(&latents)).map_err(to_py)?;
        *voice.state.lock().unwrap_or_else(|e| e.into_inner()) = Some(state);
        Ok(())
    }

    /// True when `prepare_voice` (or `load_voice`) ran for this model.
    fn is_prepared(&self, voice: &PyVoice) -> bool {
        voice.state.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.is_for(&self.inner))
    }

    /// Whole utterance as little-endian float32 bytes.
    #[pyo3(signature = (text, voice, temp=0.3, cfg=2.0, max_frames=300, eos_threshold=-1.0, seed=0, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn synthesize<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        voice: &PyVoice,
        temp: f32,
        cfg: f32,
        max_frames: u32,
        eos_threshold: f32,
        seed: u64,
        threads: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let params = params_from(temp, cfg, max_frames, eos_threshold, seed, threads)?;
        let model = Arc::clone(&self.inner);
        let text = text.to_owned();
        let pcm = py
            .detach(move || {
                let mut s = voice.session_for(&model, params)?;
                s.prime(&text)?;
                Ok::<_, Error>(s.collect())
            })
            .map_err(to_py)?;
        Ok(pcm_bytes(py, &pcm))
    }

    /// Offline decode of `[T, latent_dim]` codec latents (float32 LE bytes in,
    /// float32 LE bytes out, output delay compensated).
    fn decode<'py>(&self, py: Python<'py>, latents: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        if latents.len() % 4 != 0 {
            return Err(PyValueError::new_err("latents must be float32 bytes"));
        }
        let lat: Vec<f32> = latents
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if lat.len() % self.inner.latent_dim() != 0 {
            return Err(PyValueError::new_err(format!(
                "latents length {} is not a multiple of latent_dim {}",
                lat.len(),
                self.inner.latent_dim()
            )));
        }
        let model = Arc::clone(&self.inner);
        let pcm = py.detach(move || model.decode(&lat)).map_err(to_py)?;
        Ok(pcm_bytes(py, &pcm))
    }

    /// Streaming session bound to `voice`.
    #[pyo3(signature = (voice, temp=0.3, cfg=2.0, max_frames=300, eos_threshold=-1.0, seed=0, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn session(
        &self,
        voice: &PyVoice,
        temp: f32,
        cfg: f32,
        max_frames: u32,
        eos_threshold: f32,
        seed: u64,
        threads: u32,
    ) -> PyResult<PySession> {
        let params = params_from(temp, cfg, max_frames, eos_threshold, seed, threads)?;
        PySession::create(Arc::clone(&self.inner), voice, params)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(sample_rate={}, samples_per_step={}, latent_dim={})",
            self.inner.sample_rate(),
            self.inner.samples_per_step(),
            self.inner.latent_dim()
        )
    }
}

/// Voice prompt latents, plus the trunk state of the voice prefix for one
/// model once `Model.prepare_voice` ran.
#[pyclass(name = "Voice", module = "linnet_tts._native", frozen)]
pub struct PyVoice {
    inner: Arc<VoiceLatents>,
    state: Mutex<Option<VoiceState>>,
}

impl PyVoice {
    fn wrap(v: VoiceLatents) -> Self {
        PyVoice { inner: Arc::new(v), state: Mutex::new(None) }
    }

    /// A session for `m`: from the prepared state when it is `m`'s, else
    /// from the latents (the audio is the same either way).
    fn session_for<'m>(&self, m: &'m Linnet, params: SynthParams) -> Result<Session<'m>, Error> {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match st.as_ref() {
            Some(s) if s.is_for(m) => Session::with_voice_state(m, s, params),
            _ => Session::new(m, &self.inner, params),
        }
    }
}

#[pymethods]
impl PyVoice {
    /// Build from float32 LE bytes laid out as `[frames, dim]` (C order).
    #[staticmethod]
    fn from_bytes(data: &[u8], frames: usize, dim: usize) -> PyResult<Self> {
        if data.len() != frames * dim * 4 {
            return Err(PyValueError::new_err(format!(
                "expected {} bytes for [{frames}, {dim}] float32, got {}",
                frames * dim * 4,
                data.len()
            )));
        }
        let v: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let inner = VoiceLatents::from_vec(v, dim).map_err(to_py)?;
        Ok(PyVoice::wrap(inner))
    }

    /// Load from a `.npy` file without checking the width against a model.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let p = path.to_owned();
        let v = py.detach(move || VoiceLatents::load(p)).map_err(to_py)?;
        Ok(PyVoice::wrap(v))
    }

    #[getter]
    fn frames(&self) -> usize {
        self.inner.frames()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The latents as float32 LE bytes (`[frames, dim]`, C order).
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        pcm_bytes(py, self.inner.latents())
    }

    fn __repr__(&self) -> String {
        format!("Voice(frames={}, dim={})", self.inner.frames(), self.inner.dim())
    }
}

/// One utterance in progress. Holds the model alive for as long as it lives.
///
/// Field order matters: `session` borrows the `Linnet` behind `model` with
/// an erased lifetime and must be dropped first; Rust drops fields in
/// declaration order. The `Arc` keeps the model at a fixed address.
#[pyclass(name = "Session", module = "linnet_tts._native", unsendable)]
pub struct PySession {
    session: Session<'static>,
    #[allow(dead_code)]
    model: Arc<Linnet>,
}

impl PySession {
    fn create(model: Arc<Linnet>, voice: &PyVoice, params: SynthParams) -> PyResult<Self> {
        // SAFETY: `model` is heap-allocated behind an `Arc` that this struct
        // owns and drops only after `session`; the reference therefore
        // stays valid for the session's whole life.
        let m: &'static Linnet = unsafe { &*Arc::as_ptr(&model) };
        let session = voice.session_for(m, params).map_err(to_py)?;
        Ok(PySession { session, model })
    }
}

#[pymethods]
impl PySession {
    /// Tokenize `text` and run the prefix. Resets any previous utterance;
    /// the voice part of the prefix is kept, so only the text rows run.
    fn prime(&mut self, py: Python<'_>, text: &str) -> PyResult<()> {
        let s = &mut self.session;
        py.detach(move || s.prime(text)).map_err(to_py)
    }

    /// Run the prefix from ready-made token ids.
    fn prime_tokens(&mut self, py: Python<'_>, tokens: Vec<u32>) -> PyResult<()> {
        let s = &mut self.session;
        py.detach(move || s.prime_tokens(&tokens)).map_err(to_py)
    }

    /// Next chunk as float32 LE bytes, or `None` once the utterance is over.
    fn step<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        let s = &mut self.session;
        let chunk: Option<Vec<f32>> = py.detach(move || s.step().map(|c| c.to_vec()));
        chunk.map(|c| pcm_bytes(py, &c))
    }

    /// Codec-space latent of the most recent step, float32 LE bytes.
    fn latent<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        pcm_bytes(py, self.session.latent())
    }

    /// LM steps emitted so far.
    #[getter]
    fn frames(&self) -> usize {
        self.session.frames()
    }

    /// True once the last chunk (the `output_delay` zeros) has been returned; the next `step` returns `None`.
    #[getter]
    fn finished(&self) -> bool {
        self.session.is_finished()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", linnet_tts::VERSION)?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PySession>()?;
    Ok(())
}
