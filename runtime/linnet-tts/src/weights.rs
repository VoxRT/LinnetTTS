//! Weight structures and their loader. Matrices keep the storage dtype of
//! the file (fp32 or binary16) and are consumed through [`Mat`], which
//! picks the matching kernel; every vector parameter (norms, biases,
//! layer scales, latent statistics) is widened to f32 at load time.
//!
//! The two time embeddings of the flow head only ever see the constants
//! `s = 0`, `t = 1` (one-step sampling), so their contribution to the
//! conditioning vector is evaluated once here and stored as a constant.

use linnet_ops::ops;

use crate::config::{DecoderConfig, LmConfig};
use crate::error::Result;
use crate::pool::{SendPtr, ThreadPool};
use crate::safetensors::{DType, SafeTensors};

/// Element-wise activation applied right after a matrix product (and its
/// bias) inside [`Mat::gemm_par`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    /// erf-GELU (decoder ConvNeXt blocks).
    Gelu,
    /// tanh-GELU (trunk feed-forward).
    GeluTanh,
    /// SiLU (flow head).
    Silu,
}

/// Bias and activation of one output row (or a 32-aligned piece of it).
/// Applying this piecewise is bit-equal to applying it to the whole row:
/// the SIMD activation kernels only depend on an element's position
/// through the vector/scalar-tail split, and chunk boundaries that are
/// multiples of 32 keep that split unchanged.
#[inline]
fn epilogue(row: &mut [f32], bias: Option<&[f32]>, act: Act) {
    if let Some(b) = bias {
        ops::add_inplace(row, b);
    }
    match act {
        Act::None => {}
        Act::Gelu => ops::gelu_inplace(row),
        Act::GeluTanh => ops::gelu_tanh_inplace(row),
        Act::Silu => ops::silu_inplace(row),
    }
}

/// Products smaller than this run on the calling thread.
const PAR_MIN_MACS: usize = 32 * 1024;
/// Output rows per chunk when a product is split over the weight rows.
const OUT_CHUNK: usize = 32;
/// Activation rows per chunk when a product is split over the inputs.
const ROW_BLOCK: usize = 4;
/// Up to this many activation rows the split is over weight rows.
const SMALL_M: usize = 8;

/// A dense `[rows × cols]` weight matrix in its storage dtype.
pub enum Mat {
    F32 { w: Vec<f32>, rows: usize, cols: usize },
    F16 { w: Vec<u16>, rows: usize, cols: usize },
}

impl Mat {
    pub fn rows(&self) -> usize {
        match self {
            Mat::F32 { rows, .. } | Mat::F16 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Mat::F32 { cols, .. } | Mat::F16 { cols, .. } => *cols,
        }
    }

    /// `out[rows] = W · x[cols]` (no bias), f32 accumulation.
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        match self {
            Mat::F32 { w, rows, cols } => ops::matmul_mv(x, w, *cols, *rows, out),
            Mat::F16 { w, rows, cols } => ops::matmul_mv_f16(x, w, *cols, *rows, out),
        }
    }

    /// `out[m × rows] = a[m × cols] · Wᵀ`: `m` stacked GEMVs with each
    /// weight row streamed once.
    pub fn gemm(&self, a: &[f32], m: usize, out: &mut [f32]) {
        debug_assert_eq!(a.len(), m * self.cols());
        debug_assert_eq!(out.len(), m * self.rows());
        if m == 1 {
            self.gemv(a, out);
            return;
        }
        match self {
            Mat::F32 { w, rows, cols } => ops::matmul_mm_transposed_b(a, w, m, *cols, *rows, out),
            Mat::F16 { w, rows, cols } => ops::matmul_mm_f16(a, w, m, *cols, *rows, out),
        }
    }

    /// `out[m × (j1 − j0)] = a[m × cols] · W[j0..j1]ᵀ`: the product
    /// restricted to weight rows `j0..j1`, same kernels as [`Mat::gemm`].
    fn gemm_rows(&self, j0: usize, j1: usize, a: &[f32], m: usize, out: &mut [f32]) {
        let k = self.cols();
        let n = j1 - j0;
        match self {
            Mat::F32 { w, .. } => {
                let w = &w[j0 * k..j1 * k];
                if m == 1 {
                    ops::matmul_mv(a, w, k, n, out)
                } else {
                    ops::matmul_mm_transposed_b(a, w, m, k, n, out)
                }
            }
            Mat::F16 { w, .. } => {
                let w = &w[j0 * k..j1 * k];
                if m == 1 {
                    ops::matmul_mv_f16(a, w, k, n, out)
                } else {
                    ops::matmul_mm_f16(a, w, m, k, n, out)
                }
            }
        }
    }

    /// [`Mat::gemm`] followed by an optional per-row `bias` and `act`,
    /// spread over the pool. Bit-equal to the single-threaded sequence
    /// (`gemm`, then `add_inplace(row, bias)` and the activation per row)
    /// for any thread count: each output value is one dot product done by
    /// one thread, and the epilogue is per row (see [`epilogue`]).
    ///
    /// Split: up to [`SMALL_M`] activation rows (the per-step shapes) over
    /// the weight rows in chunks of [`OUT_CHUNK`], so the weights are
    /// streamed once in total and each chunk writes its own columns of
    /// every output row; more rows (prefill) over the activation rows in
    /// blocks of [`ROW_BLOCK`], which keeps the kernels' four-row blocking.
    pub fn gemm_par(&self, pool: &ThreadPool, a: &[f32], m: usize, out: &mut [f32], bias: Option<&[f32]>, act: Act) {
        let (k, n) = (self.cols(), self.rows());
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(out.len(), m * n);
        debug_assert!(bias.is_none_or(|b| b.len() == n));
        if pool.threads() == 1 || m * k * n < PAR_MIN_MACS {
            self.gemm(a, m, out);
            for r in 0..m {
                epilogue(&mut out[r * n..(r + 1) * n], bias, act);
            }
            return;
        }
        let op = SendPtr::new(out);
        if m > SMALL_M {
            let blocks = m.div_ceil(ROW_BLOCK);
            pool.run(blocks, &|b| {
                let i0 = b * ROW_BLOCK;
                let i1 = (i0 + ROW_BLOCK).min(m);
                // SAFETY: block b owns output rows i0..i1.
                let o = unsafe { op.slice(i0 * n, (i1 - i0) * n) };
                self.gemm(&a[i0 * k..i1 * k], i1 - i0, o);
                for r in 0..i1 - i0 {
                    epilogue(&mut o[r * n..(r + 1) * n], bias, act);
                }
            });
            return;
        }
        let chunks = n.div_ceil(OUT_CHUNK);
        pool.run(chunks, &|c| {
            let j0 = c * OUT_CHUNK;
            let j1 = (j0 + OUT_CHUNK).min(n);
            let w = j1 - j0;
            let b = bias.map(|b| &b[j0..j1]);
            if m == 1 {
                // SAFETY: chunk c owns out[j0..j1].
                let o = unsafe { op.slice(j0, w) };
                self.gemm_rows(j0, j1, a, 1, o);
                epilogue(o, b, act);
            } else {
                let mut tmp = [0.0f32; SMALL_M * OUT_CHUNK];
                let t = &mut tmp[..m * w];
                self.gemm_rows(j0, j1, a, m, t);
                for r in 0..m {
                    // SAFETY: chunk c owns columns j0..j1 of every output row.
                    let o = unsafe { op.slice(r * n + j0, w) };
                    o.copy_from_slice(&t[r * w..(r + 1) * w]);
                    epilogue(o, b, act);
                }
            }
        });
    }

    /// Copy row `r` into `out` as f32 (embedding lookup).
    pub fn row(&self, r: usize, out: &mut [f32]) {
        debug_assert!(r < self.rows());
        match self {
            Mat::F32 { w, cols, .. } => out.copy_from_slice(&w[r * cols..(r + 1) * cols]),
            Mat::F16 { w, cols, .. } => ops::embed_lookup_f16(w, *cols, r, out),
        }
    }
}

fn mat(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Mat> {
    let info = st.expect_shape(name, &[rows, cols])?;
    Ok(match info.dtype {
        DType::F32 => Mat::F32 {
            w: st.f32_vec(name, &[rows, cols])?,
            rows,
            cols,
        },
        DType::F16 => Mat::F16 {
            w: st.f16_vec(name, &[rows, cols])?,
            rows,
            cols,
        },
    })
}

fn vec1(st: &SafeTensors, name: &str, n: usize) -> Result<Vec<f32>> {
    st.f32_vec(name, &[n])
}

// ───────────────────────── flow LM ─────────────────────────

/// One pre-LN transformer layer of the trunk (no biases in the linears).
pub struct TrunkLayer {
    pub norm_attn_w: Vec<f32>,
    pub norm_attn_b: Vec<f32>,
    /// `[3·d × d]`: fused Q, K, V (head-major inside each third).
    pub qkv: Mat,
    pub out: Mat,
    pub norm_ff_w: Vec<f32>,
    pub norm_ff_b: Vec<f32>,
    pub ff_in: Mat,
    pub ff_out: Mat,
}

/// One adaptive-LayerNorm block of the flow head.
pub struct AdaBlock {
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    /// `[3·h × h]`, chunks: shift | scale | gate.
    pub ada: Mat,
    pub ada_b: Vec<f32>,
    pub fc1: Mat,
    pub fc1_b: Vec<f32>,
    pub fc2: Mat,
    pub fc2_b: Vec<f32>,
}

/// The flow head: velocity field `v(x | cond)` at `(s, t) = (0, 1)`.
pub struct HeadWeights {
    pub cond_proj: Mat,
    pub cond_proj_b: Vec<f32>,
    pub in_proj: Mat,
    pub in_proj_b: Vec<f32>,
    /// `(time_embed_0(0) + time_embed_1(1)) / 2`, added to the conditioning.
    pub time_const: Vec<f32>,
    pub blocks: Vec<AdaBlock>,
    /// `[2·h × h]`, chunks: shift | scale.
    pub out_ada: Mat,
    pub out_ada_b: Vec<f32>,
    pub out_lin: Mat,
    pub out_lin_b: Vec<f32>,
    /// Unit gain / zero bias for the affine-free final LayerNorm.
    pub ones: Vec<f32>,
    pub zeros: Vec<f32>,
}

pub struct LmWeights {
    /// `[vocab + 1 × d]` token table (last row is the padding id).
    pub text_embed: Mat,
    pub bos_voice: Vec<f32>,
    pub voice_proj: Mat,
    pub bos_latent: Vec<f32>,
    pub latent_in: Mat,
    pub latent_mean: Vec<f32>,
    pub latent_std: Vec<f32>,
    pub layers: Vec<TrunkLayer>,
    pub out_norm_w: Vec<f32>,
    pub out_norm_b: Vec<f32>,
    pub eos_w: Vec<f32>,
    pub eos_b: f32,
    pub head: HeadWeights,
}

/// Sinusoidal time embedding → MLP → RMSNorm, evaluated at a constant `t`.
fn time_embed_const(st: &SafeTensors, prefix: &str, t: f32, cfg: &LmConfig) -> Result<Vec<f32>> {
    let h = cfg.head_dim;
    let half = cfg.head_time_dim / 2;
    let fc1 = mat(st, &format!("{prefix}.fc1.weight"), h, cfg.head_time_dim)?;
    let fc1_b = vec1(st, &format!("{prefix}.fc1.bias"), h)?;
    let fc2 = mat(st, &format!("{prefix}.fc2.weight"), h, h)?;
    let fc2_b = vec1(st, &format!("{prefix}.fc2.bias"), h)?;
    let scale = vec1(st, &format!("{prefix}.norm.scale"), h)?;

    // freqs[i] = exp(-ln(max_period) · i / half), evaluated in f32 like the reference.
    let neg_log = -(cfg.head_time_max_period.ln()) as f32;
    let mut emb = vec![0.0f32; cfg.head_time_dim];
    for i in 0..half {
        let f = (neg_log * i as f32 / half as f32).exp();
        let (s, c) = (t * f).sin_cos();
        emb[i] = c;
        emb[half + i] = s;
    }
    let mut h1 = vec![0.0f32; h];
    fc1.gemv(&emb, &mut h1);
    ops::add_inplace(&mut h1, &fc1_b);
    ops::silu_inplace(&mut h1);
    let mut h2 = vec![0.0f32; h];
    fc2.gemv(&h1, &mut h2);
    ops::add_inplace(&mut h2, &fc2_b);
    let mut out = vec![0.0f32; h];
    ops::rms_norm_unbiased(&h2, &scale, 1e-5, &mut out);
    Ok(out)
}

pub fn load_lm(st: &SafeTensors, cfg: &LmConfig) -> Result<LmWeights> {
    let d = cfg.d_model;
    let ld = cfg.latent_dim;
    let h = cfg.head_dim;
    let ff = cfg.d_ff();

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = format!("lm.trunk.layers.{i}");
        layers.push(TrunkLayer {
            norm_attn_w: vec1(st, &format!("{p}.norm_attn.weight"), d)?,
            norm_attn_b: vec1(st, &format!("{p}.norm_attn.bias"), d)?,
            qkv: mat(st, &format!("{p}.attn.qkv.weight"), 3 * d, d)?,
            out: mat(st, &format!("{p}.attn.out.weight"), d, d)?,
            norm_ff_w: vec1(st, &format!("{p}.norm_ff.weight"), d)?,
            norm_ff_b: vec1(st, &format!("{p}.norm_ff.bias"), d)?,
            ff_in: mat(st, &format!("{p}.ff_in.weight"), ff, d)?,
            ff_out: mat(st, &format!("{p}.ff_out.weight"), d, ff)?,
        });
    }

    let mut blocks = Vec::with_capacity(cfg.head_depth);
    for i in 0..cfg.head_depth {
        let p = format!("lm.head.blocks.{i}");
        blocks.push(AdaBlock {
            norm_w: vec1(st, &format!("{p}.norm.weight"), h)?,
            norm_b: vec1(st, &format!("{p}.norm.bias"), h)?,
            ada: mat(st, &format!("{p}.ada.weight"), 3 * h, h)?,
            ada_b: vec1(st, &format!("{p}.ada.bias"), 3 * h)?,
            fc1: mat(st, &format!("{p}.fc1.weight"), h, h)?,
            fc1_b: vec1(st, &format!("{p}.fc1.bias"), h)?,
            fc2: mat(st, &format!("{p}.fc2.weight"), h, h)?,
            fc2_b: vec1(st, &format!("{p}.fc2.bias"), h)?,
        });
    }

    let e_s = time_embed_const(st, "lm.head.time_embed.0", 0.0, cfg)?;
    let e_t = time_embed_const(st, "lm.head.time_embed.1", 1.0, cfg)?;
    let time_const: Vec<f32> = e_s.iter().zip(&e_t).map(|(a, b)| (a + b) / 2.0).collect();

    let head = HeadWeights {
        cond_proj: mat(st, "lm.head.cond_proj.weight", h, d)?,
        cond_proj_b: vec1(st, "lm.head.cond_proj.bias", h)?,
        in_proj: mat(st, "lm.head.in_proj.weight", h, ld)?,
        in_proj_b: vec1(st, "lm.head.in_proj.bias", h)?,
        time_const,
        blocks,
        out_ada: mat(st, "lm.head.out.ada.weight", 2 * h, h)?,
        out_ada_b: vec1(st, "lm.head.out.ada.bias", 2 * h)?,
        out_lin: mat(st, "lm.head.out.linear.weight", ld, h)?,
        out_lin_b: vec1(st, "lm.head.out.linear.bias", ld)?,
        ones: vec![1.0; h],
        zeros: vec![0.0; h],
    };

    Ok(LmWeights {
        text_embed: mat(st, "lm.text_embed.weight", cfg.vocab_size + 1, d)?,
        bos_voice: vec1(st, "lm.bos_voice", d)?,
        voice_proj: mat(st, "lm.voice_proj.weight", d, ld)?,
        bos_latent: vec1(st, "lm.bos_latent", ld)?,
        latent_in: mat(st, "lm.latent_in.weight", d, ld)?,
        latent_mean: vec1(st, "lm.latent_mean", ld)?,
        latent_std: vec1(st, "lm.latent_std", ld)?,
        layers,
        out_norm_w: vec1(st, "lm.out_norm.weight", d)?,
        out_norm_b: vec1(st, "lm.out_norm.bias", d)?,
        eos_w: st.f32_vec("lm.eos.weight", &[1, d])?,
        eos_b: vec1(st, "lm.eos.bias", 1)?[0],
        head,
    })
}

// ───────────────────────── codec decoder ─────────────────────────

/// Causal ConvNeXt block: depthwise conv (tap-major weights), LayerNorm,
/// pointwise expand / contract, layer scale.
pub struct ConvNeXtWeights {
    /// `[kernel × C]`: `dw_w[k·C + c] = torch_w[c, 0, k]`.
    pub dw_w: Vec<f32>,
    pub dw_b: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub norm_b: Vec<f32>,
    pub pw1: Mat,
    pub pw1_b: Vec<f32>,
    pub pw2: Mat,
    pub pw2_b: Vec<f32>,
    pub gamma: Vec<f32>,
}

pub struct DecoderWeights {
    pub in_proj: Mat,
    pub in_proj_b: Vec<f32>,
    pub low: Vec<ConvNeXtWeights>,
    /// `[up_factor · d_high × d_low]`.
    pub up: Mat,
    pub up_b: Vec<f32>,
    pub high: Vec<ConvNeXtWeights>,
    pub head_norm_w: Vec<f32>,
    pub head_norm_b: Vec<f32>,
    /// `[2 · n_bins × d_high]`: log-magnitudes then phases.
    pub head: Mat,
    pub head_b: Vec<f32>,
}

fn convnext(st: &SafeTensors, p: &str, d: usize, expand: usize, k: usize) -> Result<ConvNeXtWeights> {
    let dw = st.f32_vec(&format!("{p}.dw.weight"), &[d, 1, k])?;
    let mut dw_w = vec![0.0f32; k * d];
    for c in 0..d {
        for t in 0..k {
            dw_w[t * d + c] = dw[c * k + t];
        }
    }
    Ok(ConvNeXtWeights {
        dw_w,
        dw_b: vec1(st, &format!("{p}.dw.bias"), d)?,
        norm_w: vec1(st, &format!("{p}.norm.weight"), d)?,
        norm_b: vec1(st, &format!("{p}.norm.bias"), d)?,
        pw1: mat(st, &format!("{p}.pw1.weight"), expand * d, d)?,
        pw1_b: vec1(st, &format!("{p}.pw1.bias"), expand * d)?,
        pw2: mat(st, &format!("{p}.pw2.weight"), d, expand * d)?,
        pw2_b: vec1(st, &format!("{p}.pw2.bias"), d)?,
        gamma: vec1(st, &format!("{p}.gamma"), d)?,
    })
}

pub fn load_decoder(st: &SafeTensors, cfg: &DecoderConfig, sample_rate: u32) -> Result<DecoderWeights> {
    let uf = cfg.up_factor(sample_rate);
    let n_bins = cfg.n_bins();
    let mut low = Vec::with_capacity(cfg.n_low);
    for i in 0..cfg.n_low {
        low.push(convnext(st, &format!("decoder.low.{i}"), cfg.d_low, cfg.expand, cfg.kernel)?);
    }
    let mut high = Vec::with_capacity(cfg.n_high);
    for i in 0..cfg.n_high {
        high.push(convnext(st, &format!("decoder.high.{i}"), cfg.d_high, cfg.expand, cfg.kernel)?);
    }
    Ok(DecoderWeights {
        in_proj: mat(st, "decoder.in_proj.weight", cfg.d_low, cfg.latent_dim)?,
        in_proj_b: vec1(st, "decoder.in_proj.bias", cfg.d_low)?,
        low,
        up: mat(st, "decoder.up.weight", uf * cfg.d_high, cfg.d_low)?,
        up_b: vec1(st, "decoder.up.bias", uf * cfg.d_high)?,
        high,
        head_norm_w: vec1(st, "decoder.head_norm.weight", cfg.d_high)?,
        head_norm_b: vec1(st, "decoder.head_norm.bias", cfg.d_high)?,
        head: mat(st, "decoder.head.weight", 2 * n_bins, cfg.d_high)?,
        head_b: vec1(st, "decoder.head.bias", 2 * n_bins)?,
    })
}
