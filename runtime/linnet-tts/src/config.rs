//! Model configuration, read from the `__metadata__` of the weight file
//! (`lm_config` and `decoder_config` are JSON strings; the remaining keys
//! are plain strings). Every shape the loader validates is derived from
//! these values.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::safetensors::SafeTensors;

/// Container version this runtime accepts.
pub const WEIGHTS_FORMAT: &str = "linnet-tts/1";

/// Flow LM hyper-parameters (`lm_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct LmConfig {
    #[serde(default = "d48")]
    pub latent_dim: usize,
    #[serde(default = "d512")]
    pub d_model: usize,
    #[serde(default = "d6")]
    pub n_layers: usize,
    #[serde(default = "d8")]
    pub n_heads: usize,
    #[serde(default = "d4")]
    pub ff_mult: usize,
    #[serde(default = "d1e4")]
    pub rope_max_period: f64,
    #[serde(default = "d6000")]
    pub vocab_size: usize,
    #[serde(default = "d512")]
    pub head_dim: usize,
    #[serde(default = "d6")]
    pub head_depth: usize,
    #[serde(default = "d256")]
    pub head_time_dim: usize,
    #[serde(default = "d1e4")]
    pub head_time_max_period: f64,
}

impl LmConfig {
    /// Feed-forward width.
    pub fn d_ff(&self) -> usize {
        self.d_model * self.ff_mult
    }
    /// Per-head width of the attention.
    pub fn head_width(&self) -> usize {
        self.d_model / self.n_heads
    }
}

/// Codec decoder hyper-parameters (`decoder_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    #[serde(default = "d24")]
    pub latent_dim: usize,
    /// Latent frame rate × 2 (stored doubled so it stays an integer).
    #[serde(default = "d50")]
    pub fps_x2: usize,
    #[serde(default = "d384")]
    pub d_low: usize,
    #[serde(default = "d2")]
    pub n_low: usize,
    #[serde(default = "d384")]
    pub d_high: usize,
    #[serde(default = "d8")]
    pub n_high: usize,
    #[serde(default = "d3")]
    pub expand: usize,
    #[serde(default = "d7")]
    pub kernel: usize,
    #[serde(default = "d240")]
    pub hop: usize,
    #[serde(default = "d960")]
    pub win: usize,
    #[serde(default = "d1024")]
    pub n_fft: usize,
    #[serde(default = "d1e_6")]
    pub layer_scale_init: f64,
    #[serde(default = "dm4")]
    pub mag_bias_init: f64,
    #[serde(default = "dm12")]
    pub logmag_min: f64,
    #[serde(default = "dln100")]
    pub logmag_max: f64,
}

impl DecoderConfig {
    /// Samples emitted per latent frame at `sample_rate`.
    pub fn samples_per_frame(&self, sample_rate: u32) -> usize {
        2 * sample_rate as usize / self.fps_x2
    }
    /// STFT columns per latent frame.
    pub fn up_factor(&self, sample_rate: u32) -> usize {
        self.samples_per_frame(sample_rate) / self.hop
    }
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
    /// Overlap-add tail carried between frames.
    pub fn ola_tail(&self) -> usize {
        self.win - self.hop
    }
}

/// Everything the metadata describes.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub lm: LmConfig,
    pub decoder: DecoderConfig,
    pub sample_rate: u32,
    /// The decoder reproduces its input delayed by this many samples;
    /// synthesis drops them at the start and pads the end.
    pub output_delay: usize,
}

fn meta_str<'a>(st: &'a SafeTensors, key: &str) -> Result<&'a str> {
    st.metadata()
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| Error::Format(format!("metadata key {key} missing")))
}

impl ModelConfig {
    /// Parse and validate the metadata of a weight file.
    pub fn from_safetensors(st: &SafeTensors) -> Result<Self> {
        let format = meta_str(st, "format")?;
        if format != WEIGHTS_FORMAT {
            return Err(Error::Format(format!(
                "weights format {format:?}, this runtime reads {WEIGHTS_FORMAT:?}"
            )));
        }
        let lm: LmConfig = serde_json::from_str(meta_str(st, "lm_config")?)
            .map_err(|e| Error::Format(format!("lm_config: {e}")))?;
        let decoder: DecoderConfig = serde_json::from_str(meta_str(st, "decoder_config")?)
            .map_err(|e| Error::Format(format!("decoder_config: {e}")))?;
        let sample_rate = match st.metadata().get("sample_rate") {
            Some(s) => s
                .parse::<u32>()
                .map_err(|e| Error::Format(format!("sample_rate {s:?}: {e}")))?,
            None => 24_000,
        };
        let output_delay = match st.metadata().get("output_delay") {
            Some(s) => s
                .parse::<usize>()
                .map_err(|e| Error::Format(format!("output_delay {s:?}: {e}")))?,
            None => 720,
        };
        let cfg = ModelConfig {
            lm,
            decoder,
            sample_rate,
            output_delay,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configurations the kernels cannot run (zero sizes, odd
    /// RoPE pairs, non power-of-two FFT, inconsistent frame geometry).
    pub fn validate(&self) -> Result<()> {
        let lm = &self.lm;
        let dc = &self.decoder;
        let nonzero = [
            (lm.latent_dim, "lm.latent_dim"),
            (lm.d_model, "lm.d_model"),
            (lm.n_layers, "lm.n_layers"),
            (lm.n_heads, "lm.n_heads"),
            (lm.ff_mult, "lm.ff_mult"),
            (lm.vocab_size, "lm.vocab_size"),
            (lm.head_dim, "lm.head_dim"),
            (lm.head_depth, "lm.head_depth"),
            (lm.head_time_dim, "lm.head_time_dim"),
            (dc.latent_dim, "decoder.latent_dim"),
            (dc.fps_x2, "decoder.fps_x2"),
            (dc.d_low, "decoder.d_low"),
            (dc.d_high, "decoder.d_high"),
            (dc.expand, "decoder.expand"),
            (dc.kernel, "decoder.kernel"),
            (dc.hop, "decoder.hop"),
            (dc.win, "decoder.win"),
            (dc.n_fft, "decoder.n_fft"),
            (self.sample_rate as usize, "sample_rate"),
        ];
        for (v, name) in nonzero {
            if v == 0 {
                return Err(Error::Shape(format!("{name} == 0")));
            }
        }
        if lm.d_model % lm.n_heads != 0 {
            return Err(Error::Shape("lm.d_model not divisible by lm.n_heads".into()));
        }
        if lm.head_width() % 2 != 0 {
            return Err(Error::Shape("attention head width must be even (RoPE pairs)".into()));
        }
        if lm.head_time_dim % 2 != 0 {
            return Err(Error::Shape("lm.head_time_dim must be even".into()));
        }
        if !(lm.rope_max_period > 0.0 && lm.head_time_max_period > 0.0) {
            return Err(Error::Shape("RoPE / time-embedding periods must be positive".into()));
        }
        if lm.latent_dim != 2 * dc.latent_dim {
            return Err(Error::Shape(format!(
                "lm.latent_dim {} must be twice decoder.latent_dim {}",
                lm.latent_dim, dc.latent_dim
            )));
        }
        if !dc.n_fft.is_power_of_two() {
            return Err(Error::Shape(format!("decoder.n_fft {} is not a power of two", dc.n_fft)));
        }
        if (2 * self.sample_rate as usize) % dc.fps_x2 != 0 {
            return Err(Error::Shape("sample_rate * 2 not divisible by fps_x2".into()));
        }
        let spf = dc.samples_per_frame(self.sample_rate);
        if spf % dc.hop != 0 || dc.win % dc.hop != 0 || dc.win <= dc.hop || dc.win > dc.n_fft {
            return Err(Error::Shape(format!(
                "decoder geometry: samples/frame {spf}, hop {}, win {}, n_fft {}",
                dc.hop, dc.win, dc.n_fft
            )));
        }
        if spf < dc.ola_tail() {
            return Err(Error::Shape("overlap-add tail longer than one frame".into()));
        }
        if !(dc.logmag_min.is_finite() && dc.logmag_max.is_finite() && dc.logmag_min <= dc.logmag_max) {
            return Err(Error::Shape("decoder log-magnitude clamp is not ordered".into()));
        }
        if self.output_delay > spf * 2 {
            return Err(Error::Shape("output_delay longer than one LM step".into()));
        }
        Ok(())
    }
}

fn d2() -> usize { 2 }
fn d3() -> usize { 3 }
fn d4() -> usize { 4 }
fn d6() -> usize { 6 }
fn d7() -> usize { 7 }
fn d8() -> usize { 8 }
fn d24() -> usize { 24 }
fn d48() -> usize { 48 }
fn d50() -> usize { 50 }
fn d240() -> usize { 240 }
fn d256() -> usize { 256 }
fn d384() -> usize { 384 }
fn d512() -> usize { 512 }
fn d960() -> usize { 960 }
fn d1024() -> usize { 1024 }
fn d6000() -> usize { 6000 }
fn d1e4() -> f64 { 10_000.0 }
fn d1e_6() -> f64 { 1e-6 }
fn dm4() -> f64 { -4.0 }
fn dm12() -> f64 { -12.0 }
fn dln100() -> f64 { 4.605_170_185_988_092 }
