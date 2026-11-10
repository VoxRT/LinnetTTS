//! LinnetTTS runtime: a small autoregressive flow LM over continuous codec
//! latents plus a causal Vocos-style decoder, with streaming synthesis.
//!
//! ```no_run
//! use linnet_tts::{Linnet, Session, SynthParams, VoiceLatents};
//!
//! let model = Linnet::load("linnet-en-35m.fp16.safetensors", "tokenizer.model")?;
//! let voice = VoiceLatents::load("voice.npy")?;
//! let mut session = Session::new(&model, &voice, SynthParams::default())?;
//! session.prime("Hello there.")?;
//! let mut audio = Vec::new();
//! while let Some(chunk) = session.step() {
//!     audio.extend_from_slice(chunk); // 24 kHz f32
//! }
//! # Ok::<(), linnet_tts::Error>(())
//! ```
//!
//! Weights are a single safetensors file (the fp16 runtime file; the fp32
//! reference file loads too but is slower and samples a different
//! trajectory, see `docs/weights-format.md`); the tokenizer is a SentencePiece
//! `.model` file; voices are `[T, 48]` `.npy` latents. As a dependency:
//! `linnet-tts = { git = "https://github.com/VoxRT/LinnetTTS", tag = "v0.1.1" }`. All arithmetic accumulates
//! in f32 through the kernels of `linnet-ops`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod c_api;
pub mod config;
pub mod decoder;
pub mod error;
pub mod lm;
pub mod pool;
pub mod rope;
pub mod safetensors;
pub mod session;
pub mod tokenizer;
pub mod voice;
pub mod weights;

pub use config::{DecoderConfig, LmConfig, ModelConfig, WEIGHTS_FORMAT};
pub use decoder::VocosDecoder;
pub use error::{Error, Result};
pub use lm::FlowLm;
pub use pool::{auto_threads, ThreadPool};
pub use session::{Linnet, Session, SynthParams, VoiceState};
pub use tokenizer::Tokenizer;
pub use voice::VoiceLatents;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
