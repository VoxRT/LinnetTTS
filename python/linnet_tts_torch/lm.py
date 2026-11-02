"""Autoregressive flow language model over packed codec latents.

One LM step predicts a 48-dim vector = two consecutive 24-dim codec frames (25 Hz), so the LM
runs at 12.5 steps/s. The model is:

    prefix  = [voice BOS, voice latents @ voice_proj, text token embeddings]
    stream  = [latent BOS, latent_1, latent_2, ...] @ latent_in
    trunk   = causal pre-LN transformer with rotary positions (interleaved pairs) over prefix + stream
    z_t     = out_norm(trunk)[last]
    eos     = eos(z_t) > eos_threshold
    latent  = x0 + head(z_t, s=0, t=1, x0),  x0 ~ N(0, temp * I)      (one-step flow)

Classifier-free guidance runs a second, unconditional stream whose prefix is just the voice BOS and
mixes trunk outputs: z = z_null + cfg * (z_cond - z_null).

Latents are modelled in normalised space; `latent_mean` / `latent_std` map them back to codec space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LMConfig:
    latent_dim: int = 48
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    rope_max_period: float = 10000.0
    vocab_size: int = 6000  # embedding table has one extra (padding) row
    head_dim: int = 512
    head_depth: int = 6
    head_time_dim: int = 256
    head_time_max_period: float = 10000.0

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ff_mult

    @property
    def head_dim_per_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


# ----------------------------------------------------------------------------- trunk


def apply_rope(q: torch.Tensor, k: torch.Tensor, offset: int, max_period: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding on interleaved (even=real, odd=imag) pairs. q, k: [B, T, H, D]."""
    B, T, H, D = q.shape
    assert D % 2 == 0
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts + offset
    ts = ts.view(-1, 1, 1)

    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, k.shape[2], D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(B, T, H, D), ko.view(B, T, k.shape[2], D)


class KVCache:
    """Per-layer key/value cache laid out [B, S, H, D]; `pos` is the absolute index of the next write."""

    def __init__(self, batch: int, capacity: int, n_heads: int, dim: int, device, dtype):
        self.k = torch.empty(batch, capacity, n_heads, dim, device=device, dtype=dtype)
        self.v = torch.empty(batch, capacity, n_heads, dim, device=device, dtype=dtype)
        self.pos = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[1]
        assert self.pos + t <= self.k.shape[1], "KV cache capacity exceeded"
        self.k[:, self.pos : self.pos + t] = k
        self.v[:, self.pos : self.pos + t] = v
        n = self.pos + t
        return self.k[:, :n], self.v[:, :n]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.dim = cfg.head_dim_per_head
        self.max_period = cfg.rope_max_period
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        b, t, _ = x.shape
        packed = self.qkv(x).view(b, t, 3, self.n_heads, self.dim)
        q, k, v = torch.unbind(packed, dim=2)  # [B, T, H, D]
        offset = 0 if cache is None else cache.pos
        q, k = apply_rope(q, k, offset, self.max_period)
        q = q.transpose(1, 2)  # [B, H, T, D]
        if cache is None:
            k_all, v_all = k, v
        else:
            k_all, v_all = cache.append(k, v)
            cache.pos += t
        k_attn = k_all.permute(0, 2, 1, 3)  # [B, H, S, D]
        v_attn = v_all.permute(0, 2, 1, 3)
        s = k_attn.shape[2]
        pos_q = offset + torch.arange(t, device=x.device, dtype=torch.long).view(1, -1)
        pos_k = torch.arange(s, device=x.device, dtype=torch.long).view(1, -1)
        mask = (pos_q[:, :, None] - pos_k[:, None, :]) >= 0  # [1, T, S] causal
        mask = mask[:, None].expand(b, 1, t, s)
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, mask, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.dim)
        return self.out(y)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.norm_ff = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.ff_in = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.ff_out = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cache)
        x = x + self.ff_out(F.gelu(self.ff_in(self.norm_ff(x)), approximate="tanh"))
        return x


class Trunk(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])

    def new_cache(self, batch: int, capacity: int) -> list[KVCache]:
        p = self.layers[0].attn.qkv.weight
        return [
            KVCache(batch, capacity, self.cfg.n_heads, self.cfg.head_dim_per_head, p.device, p.dtype)
            for _ in self.layers
        ]

    def forward(self, x: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, None if caches is None else caches[i])
        return x


# ----------------------------------------------------------------------------- flow head


class RMSNorm(nn.Module):
    """x * scale / sqrt(eps + var(x)) with the (unbiased) sample variance along the last axis."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = self.eps + x.var(dim=-1, keepdim=True)
        return (x * (self.scale.to(var) * torch.rsqrt(var))).to(x.dtype)


class PlainLayerNorm(nn.Module):
    """LayerNorm written out explicitly (biased variance, eps inside the sqrt), optional affine."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of a scalar time followed by a small MLP and RMSNorm."""

    def __init__(self, dim: int, time_dim: int, max_period: float):
        super().__init__()
        assert time_dim % 2 == 0
        self.fc1 = nn.Linear(time_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        half = time_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs.to(t.dtype)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.norm(self.fc2(F.silu(self.fc1(emb))))


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = PlainLayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ada = nn.Linear(dim, 3 * dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(F.silu(y)).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        h = self.fc2(F.silu(self.fc1(h)))
        return x + gate * h


class AdaLNOut(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = PlainLayerNorm(dim, eps=1e-6, affine=False)
        self.linear = nn.Linear(dim, out_dim)
        self.ada = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(F.silu(y)).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class FlowHead(nn.Module):
    """Adaptive-LayerNorm MLP: velocity field v(x_t | cond, s, t) with two scalar time conditions."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        d = cfg.head_dim
        self.time_embed = nn.ModuleList(
            [TimeEmbed(d, cfg.head_time_dim, cfg.head_time_max_period) for _ in range(2)]
        )
        self.cond_proj = nn.Linear(cfg.d_model, d)
        self.in_proj = nn.Linear(cfg.latent_dim, d)
        self.blocks = nn.ModuleList([AdaLNBlock(d) for _ in range(cfg.head_depth)])
        self.out = AdaLNOut(d, cfg.latent_dim)

    def forward(self, cond: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        y = self.cond_proj(cond)
        y = y + (self.time_embed[0](s) + self.time_embed[1](t)) / 2
        for block in self.blocks:
            x = block(x, y)
        return self.out(x, y)


# ----------------------------------------------------------------------------- model


class FlowLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.text_embed = nn.Embedding(cfg.vocab_size + 1, cfg.d_model)
        self.bos_voice = nn.Parameter(torch.zeros(cfg.d_model))
        self.voice_proj = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.bos_latent = nn.Parameter(torch.zeros(cfg.latent_dim))
        self.latent_in = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.trunk = Trunk(cfg)
        self.out_norm = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.eos = nn.Linear(cfg.d_model, 1)
        self.head = FlowHead(cfg)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("latent_std", torch.ones(cfg.latent_dim))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        return self.bos_latent.device

    def build_prefix(self, tokens: torch.Tensor, voice: torch.Tensor) -> torch.Tensor:
        """tokens [Tt] long, voice [Tv, latent_dim] (codec space) -> conditioning prefix [1, 1+Tv+Tt, d_model]."""
        t_emb = self.text_embed(tokens.to(self.device)[None])
        v_emb = self.voice_proj(voice.to(self.device, self.voice_proj.weight.dtype)[None])
        return torch.cat([self.bos_voice.view(1, 1, -1), v_emb, t_emb], dim=1)

    def null_prefix(self) -> torch.Tensor:
        return self.bos_voice.view(1, 1, -1)

    def sample_step(
        self, z: torch.Tensor, temp: float, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """One-step flow sample from trunk output z [B, d_model] -> normalised latent [B, latent_dim]."""
        noise = temp**0.5 * torch.randn(
            z.shape[0], self.cfg.latent_dim, device=z.device, dtype=z.dtype, generator=generator
        )
        s = torch.zeros_like(noise[..., :1])
        t = torch.ones_like(noise[..., :1])
        return noise + self.head(z, s, t, noise)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        voice: torch.Tensor,
        *,
        temp: float = 0.3,
        cfg: float = 2.0,
        max_frames: int = 300,
        eos_threshold: float = -1.0,
        generator: torch.Generator | None = None,
    ):
        """Yield codec-space latents [latent_dim] one LM step at a time until EOS or max_frames.

        Random draws: exactly one `torch.randn(1, latent_dim)` per emitted step (the EOS step included),
        taken from `generator` or the global RNG.
        """
        prefixes = [self.build_prefix(tokens, voice)]
        if cfg != 1.0:
            prefixes.append(self.null_prefix())
        caches = [self.trunk.new_cache(1, p.shape[1] + max_frames + 2) for p in prefixes]

        x = self.bos_latent.view(1, 1, -1)
        first = True
        for _ in range(max_frames):
            zs = []
            for prefix, cache in zip(prefixes, caches):
                inp = self.latent_in(x)
                if first:
                    inp = torch.cat([prefix, inp], dim=1)
                out = self.out_norm(self.trunk(inp, cache))
                zs.append(out[:, -1].float())
            first = False
            z = zs[0] if len(zs) == 1 else zs[1] + cfg * (zs[0] - zs[1])
            latent = self.sample_step(z, temp, generator)
            if bool(self.eos(z).squeeze(-1) > eos_threshold):
                return
            yield (latent[0] * self.latent_std + self.latent_mean)
            x = latent[:, None, :].to(self.bos_latent.dtype)
