//! The flow LM: a causal pre-LN transformer trunk over
//! `[voice BOS, voice latents, text tokens, latent BOS, latent_1, …]`
//! with a KV cache, an end-of-speech logit and a one-step flow head.
//!
//! The trunk runs in two modes: `prefill` processes a whole prefix at
//! once (matrix–matrix products, one pass over the weights), `step`
//! advances one position for `m` independent streams at a time (the
//! conditional and, with classifier-free guidance, the unconditional
//! stream) so that the weights are read once per step for both.

use linnet_ops::ops;

use crate::config::LmConfig;
use crate::error::{Error, Result};
use crate::rope::Rope;
use crate::weights::LmWeights;

/// LayerNorm epsilon of the trunk (`nn.LayerNorm` default).
const TRUNK_LN_EPS: f32 = 1e-5;
/// LayerNorm epsilon of the flow head blocks.
const HEAD_LN_EPS: f32 = 1e-6;

/// Key/value cache of one layer: `[n_heads × capacity × head_width]`
/// so that the keys of one head form a contiguous `[n × head_width]`
/// matrix.
pub struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
}

/// KV cache of one stream for all layers plus its absolute position.
pub struct TrunkState {
    layers: Vec<LayerKv>,
    pos: usize,
    cap: usize,
}

impl TrunkState {
    pub fn new(cfg: &LmConfig, cap: usize) -> Self {
        let d = cfg.d_model;
        TrunkState {
            layers: (0..cfg.n_layers)
                .map(|_| LayerKv {
                    k: vec![0.0; cap * d],
                    v: vec![0.0; cap * d],
                    n: 0,
                })
                .collect(),
            pos: 0,
            cap,
        }
    }

    /// Forget everything; the buffers are kept.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.n = 0;
        }
        self.pos = 0;
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Number of positions written so far.
    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }
}

/// Reusable buffers for `step` (up to `m_max` streams) and the head.
pub struct LmScratch {
    m_max: usize,
    h: Vec<f32>,
    qkv: Vec<f32>,
    tmp: Vec<f32>,
    ff: Vec<f32>,
    scores: Vec<f32>,
    sin: Vec<f32>,
    cos: Vec<f32>,
    // flow head
    y: Vec<f32>,
    sy: Vec<f32>,
    fx: Vec<f32>,
    fh: Vec<f32>,
    fh2: Vec<f32>,
    mod3: Vec<f32>,
    mod2: Vec<f32>,
}

impl LmScratch {
    pub fn new(cfg: &LmConfig, m_max: usize, kv_cap: usize) -> Self {
        let d = cfg.d_model;
        let h = cfg.head_dim;
        let half = cfg.head_width() / 2;
        LmScratch {
            m_max,
            h: vec![0.0; m_max * d],
            qkv: vec![0.0; m_max * 3 * d],
            tmp: vec![0.0; m_max * d],
            ff: vec![0.0; m_max * cfg.d_ff()],
            scores: vec![0.0; kv_cap],
            sin: vec![0.0; m_max * half],
            cos: vec![0.0; m_max * half],
            y: vec![0.0; h],
            sy: vec![0.0; h],
            fx: vec![0.0; h],
            fh: vec![0.0; h],
            fh2: vec![0.0; h],
            mod3: vec![0.0; 3 * h],
            mod2: vec![0.0; 2 * h],
        }
    }

    /// Largest KV length the score buffer can hold.
    pub fn m_capacity_scores(&self) -> usize {
        self.scores.len()
    }
}

/// The autoregressive flow LM.
pub struct FlowLm {
    cfg: LmConfig,
    w: LmWeights,
    rope: Rope,
}

impl FlowLm {
    pub fn new(cfg: LmConfig, w: LmWeights) -> Self {
        let rope = Rope::new(cfg.head_width(), cfg.rope_max_period);
        FlowLm { cfg, w, rope }
    }

    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    pub fn new_state(&self, cap: usize) -> TrunkState {
        TrunkState::new(&self.cfg, cap)
    }

    pub fn new_scratch(&self, m_max: usize, kv_cap: usize) -> LmScratch {
        LmScratch::new(&self.cfg, m_max, kv_cap)
    }

    /// Row of the token table as f32.
    pub fn embed_token(&self, id: u32, out: &mut [f32]) -> Result<()> {
        if id as usize > self.cfg.vocab_size {
            return Err(Error::Invalid(format!(
                "token id {id} outside the table (vocab {})",
                self.cfg.vocab_size
            )));
        }
        self.w.text_embed.row(id as usize, out);
        Ok(())
    }

    /// Length of the conditional prefix including the latent BOS row.
    pub fn prefix_len(&self, n_tokens: usize, n_voice_frames: usize) -> usize {
        1 + n_voice_frames + n_tokens + 1
    }

    /// Fill `rows` (`prefix_len × d`) with
    /// `[bos_voice | voice_proj(voice) | embed(tokens) | latent_in(bos_latent)]`.
    pub fn build_prefix(&self, tokens: &[u32], voice: &[f32], rows: &mut [f32]) -> Result<()> {
        let d = self.cfg.d_model;
        let ld = self.cfg.latent_dim;
        if voice.len() % ld != 0 {
            return Err(Error::Invalid(format!(
                "voice latents length {} not a multiple of {ld}",
                voice.len()
            )));
        }
        let tv = voice.len() / ld;
        let n = self.prefix_len(tokens.len(), tv);
        if rows.len() != n * d {
            return Err(Error::Invalid("prefix buffer has the wrong size".into()));
        }
        rows[..d].copy_from_slice(&self.w.bos_voice);
        for f in 0..tv {
            self.w
                .voice_proj
                .gemv(&voice[f * ld..(f + 1) * ld], &mut rows[(1 + f) * d..(2 + f) * d]);
        }
        for (i, tok) in tokens.iter().enumerate() {
            let r = 1 + tv + i;
            self.embed_token(*tok, &mut rows[r * d..(r + 1) * d])?;
        }
        let r = n - 1;
        self.w.latent_in.gemv(&self.w.bos_latent, &mut rows[r * d..(r + 1) * d]);
        Ok(())
    }

    /// The unconditional prefix: `[bos_voice | latent_in(bos_latent)]`.
    pub fn build_null_prefix(&self, rows: &mut [f32]) {
        let d = self.cfg.d_model;
        debug_assert_eq!(rows.len(), 2 * d);
        rows[..d].copy_from_slice(&self.w.bos_voice);
        self.w.latent_in.gemv(&self.w.bos_latent, &mut rows[d..]);
    }

    /// `x = latent_in(latent)` — the trunk input for the next step
    /// (`latent` is in normalised space).
    pub fn latent_feedback(&self, latent: &[f32], x: &mut [f32]) {
        self.w.latent_in.gemv(latent, x);
    }

    /// Run the trunk over `t` rows (`rows` is `t × d`, overwritten with
    /// the hidden states before the final norm) and append to `st`.
    pub fn prefill(&self, rows: &mut [f32], t: usize, st: &mut TrunkState) -> Result<()> {
        let d = self.cfg.d_model;
        let nh = self.cfg.n_heads;
        let hw = self.cfg.head_width();
        let half = hw / 2;
        if t == 0 || rows.len() != t * d {
            return Err(Error::Invalid("prefill: empty or mis-sized input".into()));
        }
        let n0 = st.pos;
        let n = n0 + t;
        if n > st.cap {
            return Err(Error::Invalid(format!("KV cache overflow: {n} > {}", st.cap)));
        }
        let scale = 1.0 / (hw as f32).sqrt();

        let mut sines = vec![0.0f32; t * half];
        let mut coses = vec![0.0f32; t * half];
        for r in 0..t {
            self.rope.tables(n0 + r, &mut sines[r * half..(r + 1) * half], &mut coses[r * half..(r + 1) * half]);
        }
        let mut h_rows = vec![0.0f32; t * d];
        let mut q_rows = vec![0.0f32; t * d];
        let mut qkv_rows = vec![0.0f32; t * 3 * d];
        let mut scores = vec![0.0f32; t * n];
        let mut q_head = vec![0.0f32; t * hw];
        let mut out_head = vec![0.0f32; t * hw];
        let mut tmp_rows = vec![0.0f32; t * d];
        let mut ff_rows = vec![0.0f32; t * self.cfg.d_ff()];

        for (lw, kv) in self.w.layers.iter().zip(st.layers.iter_mut()) {
            for r in 0..t {
                ops::layer_norm(
                    &rows[r * d..(r + 1) * d],
                    &lw.norm_attn_w,
                    &lw.norm_attn_b,
                    d,
                    1,
                    TRUNK_LN_EPS,
                    &mut h_rows[r * d..(r + 1) * d],
                );
            }
            lw.qkv.gemm(&h_rows, t, &mut qkv_rows);
            for r in 0..t {
                let (q, rest) = qkv_rows[r * 3 * d..(r + 1) * 3 * d].split_at_mut(d);
                let (k_new, v_new) = rest.split_at_mut(d);
                let (sn, cs) = (&sines[r * half..(r + 1) * half], &coses[r * half..(r + 1) * half]);
                self.rope.apply(q, sn, cs);
                self.rope.apply(k_new, sn, cs);
                q_rows[r * d..(r + 1) * d].copy_from_slice(q);
                let slot = n0 + r;
                for hh in 0..nh {
                    let dst = hh * st.cap * hw + slot * hw;
                    kv.k[dst..dst + hw].copy_from_slice(&k_new[hh * hw..(hh + 1) * hw]);
                    kv.v[dst..dst + hw].copy_from_slice(&v_new[hh * hw..(hh + 1) * hw]);
                }
            }
            kv.n = n;
            for hh in 0..nh {
                for r in 0..t {
                    q_head[r * hw..(r + 1) * hw].copy_from_slice(&q_rows[r * d + hh * hw..r * d + (hh + 1) * hw]);
                }
                let k_mat = &kv.k[hh * st.cap * hw..hh * st.cap * hw + n * hw];
                ops::matmul_mm_transposed_b(&q_head, k_mat, t, hw, n, &mut scores);
                for r in 0..t {
                    let row = &mut scores[r * n..(r + 1) * n];
                    for s in row.iter_mut() {
                        *s *= scale;
                    }
                    // Causal: row r attends to keys 0..=n0+r; the rest is zeroed
                    // so the value product sees no mask.
                    let valid = n0 + r + 1;
                    ops::softmax_row_inplace(&mut row[..valid], 1, valid);
                    row[valid..].fill(0.0);
                }
                let v_mat = &kv.v[hh * st.cap * hw..hh * st.cap * hw + n * hw];
                ops::matmul_mm(&scores, v_mat, t, n, hw, &mut out_head);
                for r in 0..t {
                    h_rows[r * d + hh * hw..r * d + (hh + 1) * hw].copy_from_slice(&out_head[r * hw..(r + 1) * hw]);
                }
            }
            lw.out.gemm(&h_rows, t, &mut tmp_rows);
            ops::add_inplace(rows, &tmp_rows);
            for r in 0..t {
                ops::layer_norm(
                    &rows[r * d..(r + 1) * d],
                    &lw.norm_ff_w,
                    &lw.norm_ff_b,
                    d,
                    1,
                    TRUNK_LN_EPS,
                    &mut h_rows[r * d..(r + 1) * d],
                );
            }
            lw.ff_in.gemm(&h_rows, t, &mut ff_rows);
            ops::gelu_tanh_inplace(&mut ff_rows);
            lw.ff_out.gemm(&ff_rows, t, &mut tmp_rows);
            ops::add_inplace(rows, &tmp_rows);
        }
        st.pos = n;
        Ok(())
    }

    /// Advance `m = states.len()` streams by one position. `xs` is
    /// `m × d` (one input row per stream) and is overwritten with the
    /// hidden states before the final norm.
    pub fn step(&self, xs: &mut [f32], states: &mut [TrunkState], sc: &mut LmScratch) -> Result<()> {
        let m = states.len();
        let d = self.cfg.d_model;
        let nh = self.cfg.n_heads;
        let hw = self.cfg.head_width();
        let half = hw / 2;
        if m == 0 || m > sc.m_max || xs.len() != m * d {
            return Err(Error::Invalid("step: stream count or input size mismatch".into()));
        }
        for st in states.iter() {
            if st.pos + 1 > st.cap {
                return Err(Error::Invalid(format!("KV cache overflow: capacity {}", st.cap)));
            }
            if sc.scores.len() < st.pos + 1 {
                return Err(Error::Invalid("step: scratch too small for the KV cache".into()));
            }
        }
        let scale = 1.0 / (hw as f32).sqrt();
        for (r, st) in states.iter().enumerate() {
            self.rope.tables(st.pos, &mut sc.sin[r * half..(r + 1) * half], &mut sc.cos[r * half..(r + 1) * half]);
        }
        let h = &mut sc.h[..m * d];
        let qkv = &mut sc.qkv[..m * 3 * d];
        let tmp = &mut sc.tmp[..m * d];
        let ff = &mut sc.ff[..m * self.cfg.d_ff()];

        for (li, lw) in self.w.layers.iter().enumerate() {
            for r in 0..m {
                ops::layer_norm(
                    &xs[r * d..(r + 1) * d],
                    &lw.norm_attn_w,
                    &lw.norm_attn_b,
                    d,
                    1,
                    TRUNK_LN_EPS,
                    &mut h[r * d..(r + 1) * d],
                );
            }
            lw.qkv.gemm(h, m, qkv);
            for (r, st) in states.iter_mut().enumerate() {
                let cap = st.cap;
                let kv = &mut st.layers[li];
                let slot = kv.n;
                let (q, rest) = qkv[r * 3 * d..(r + 1) * 3 * d].split_at_mut(d);
                let (k_new, v_new) = rest.split_at_mut(d);
                let (sn, cs) = (&sc.sin[r * half..(r + 1) * half], &sc.cos[r * half..(r + 1) * half]);
                self.rope.apply(q, sn, cs);
                self.rope.apply(k_new, sn, cs);
                for hh in 0..nh {
                    let dst = hh * cap * hw + slot * hw;
                    kv.k[dst..dst + hw].copy_from_slice(&k_new[hh * hw..(hh + 1) * hw]);
                    kv.v[dst..dst + hw].copy_from_slice(&v_new[hh * hw..(hh + 1) * hw]);
                }
                kv.n += 1;
                let n = kv.n;
                let scores = &mut sc.scores[..n];
                for hh in 0..nh {
                    let qh = &q[hh * hw..(hh + 1) * hw];
                    let base = hh * cap * hw;
                    let k_mat = &kv.k[base..base + n * hw];
                    ops::matmul_mv(qh, k_mat, hw, n, scores);
                    for s in scores.iter_mut() {
                        *s *= scale;
                    }
                    ops::softmax_row_inplace(scores, 1, n);
                    let v_mat = &kv.v[base..base + n * hw];
                    ops::matmul_mm(scores, v_mat, 1, n, hw, &mut h[r * d + hh * hw..r * d + (hh + 1) * hw]);
                }
            }
            lw.out.gemm(h, m, tmp);
            ops::add_inplace(xs, tmp);
            for r in 0..m {
                ops::layer_norm(
                    &xs[r * d..(r + 1) * d],
                    &lw.norm_ff_w,
                    &lw.norm_ff_b,
                    d,
                    1,
                    TRUNK_LN_EPS,
                    &mut h[r * d..(r + 1) * d],
                );
            }
            lw.ff_in.gemm(h, m, ff);
            ops::gelu_tanh_inplace(ff);
            lw.ff_out.gemm(ff, m, tmp);
            ops::add_inplace(xs, tmp);
        }
        for st in states.iter_mut() {
            st.pos += 1;
        }
        Ok(())
    }

    /// `z = out_norm(h)`: the trunk output that feeds the EOS logit and
    /// conditions the head.
    pub fn out_norm(&self, h: &[f32], z: &mut [f32]) {
        ops::layer_norm(h, &self.w.out_norm_w, &self.w.out_norm_b, self.cfg.d_model, 1, TRUNK_LN_EPS, z);
    }

    /// End-of-speech logit for a (possibly guidance-mixed) `z`.
    pub fn eos_logit(&self, z: &[f32]) -> f32 {
        ops::dot_product_f32(&self.w.eos_w, z) + self.w.eos_b
    }

    /// One-step flow sample: `latent = x0 + v(x0 | z, s = 0, t = 1)`, in
    /// normalised latent space.
    pub fn flow_sample(&self, z: &[f32], x0: &[f32], sc: &mut LmScratch, latent: &mut [f32]) {
        let hd = self.cfg.head_dim;
        let hw = &self.w.head;
        debug_assert_eq!(z.len(), self.cfg.d_model);
        debug_assert_eq!(x0.len(), self.cfg.latent_dim);
        debug_assert_eq!(latent.len(), self.cfg.latent_dim);

        hw.cond_proj.gemv(z, &mut sc.y);
        ops::add_inplace(&mut sc.y, &hw.cond_proj_b);
        ops::add_inplace(&mut sc.y, &hw.time_const);
        sc.sy.copy_from_slice(&sc.y);
        ops::silu_inplace(&mut sc.sy);

        hw.in_proj.gemv(x0, &mut sc.fx);
        ops::add_inplace(&mut sc.fx, &hw.in_proj_b);
        for blk in &hw.blocks {
            blk.ada.gemv(&sc.sy, &mut sc.mod3);
            ops::add_inplace(&mut sc.mod3, &blk.ada_b);
            let (shift, rest) = sc.mod3.split_at(hd);
            let (scale, gate) = rest.split_at(hd);
            ops::layer_norm(&sc.fx, &blk.norm_w, &blk.norm_b, hd, 1, HEAD_LN_EPS, &mut sc.fh);
            ops::modulate_inplace(&mut sc.fh, scale, shift);
            blk.fc1.gemv(&sc.fh, &mut sc.fh2);
            ops::add_inplace(&mut sc.fh2, &blk.fc1_b);
            ops::silu_inplace(&mut sc.fh2);
            blk.fc2.gemv(&sc.fh2, &mut sc.fh);
            ops::add_inplace(&mut sc.fh, &blk.fc2_b);
            ops::add_mul_inplace(&mut sc.fx, &sc.fh, gate);
        }
        hw.out_ada.gemv(&sc.sy, &mut sc.mod2);
        ops::add_inplace(&mut sc.mod2, &hw.out_ada_b);
        let (shift, scale) = sc.mod2.split_at(hd);
        ops::layer_norm(&sc.fx, &hw.ones, &hw.zeros, hd, 1, HEAD_LN_EPS, &mut sc.fh);
        ops::modulate_inplace(&mut sc.fh, scale, shift);
        hw.out_lin.gemv(&sc.fh, latent);
        ops::add_inplace(latent, &hw.out_lin_b);
        ops::add_inplace(latent, x0);
    }

    /// Normalised latent → codec space: `latent · std + mean`.
    pub fn denormalize(&self, latent: &[f32], out: &mut [f32]) {
        for i in 0..self.cfg.latent_dim {
            out[i] = latent[i] * self.w.latent_std[i] + self.w.latent_mean[i];
        }
    }
}
