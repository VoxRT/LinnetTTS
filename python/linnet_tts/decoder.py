"""Causal Vocos-style codec decoder (latent frames -> 24 kHz waveform).

Architecture:
    latent [B, T, 24] @ 25 Hz
      -> Linear in_proj -> n_low causal ConvNeXt blocks (d=384) at frame rate
      -> Linear up (x4 sub-frame expansion) -> n_high causal ConvNeXt blocks (d=384) at 100 Hz
      -> LayerNorm + Linear head -> (log-magnitude, scalar phase) per STFT bin
      -> complex spectrum -> irfft -> windowed overlap-add (win 960 / hop 240 / n_fft 1024)
    output [B, 1, T * 960]

Streaming contract:
    * Every convolution pads on the left only, so latent frame j influences samples >= j * 960.
    * `init_stream` / `stream_step` reproduce `forward` bit-for-bit from the first frame: the
      conv states start as zeros (== offline left zero-padding) and the overlap-add tail
      (win - hop = 720 samples) is carried in the state, normalised by the same deterministic
      window envelope that `forward` uses.
    * The output length is strictly T * 960 samples; the overlap-add tail beyond that is state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 24000


@dataclass(frozen=True)
class DecoderConfig:
    latent_dim: int = 24
    # Latent frame rate, stored doubled so that it stays an integer (50 -> 25 Hz).
    fps_x2: int = 50
    d_low: int = 384
    n_low: int = 2
    d_high: int = 384
    n_high: int = 8
    expand: int = 3
    kernel: int = 7
    hop: int = 240
    win: int = 960
    n_fft: int = 1024
    layer_scale_init: float = 1e-6
    mag_bias_init: float = -4.0
    logmag_min: float = -12.0
    logmag_max: float = math.log(100.0)

    @property
    def samples_per_frame(self) -> int:
        assert (2 * SAMPLE_RATE) % self.fps_x2 == 0
        spf = 2 * SAMPLE_RATE // self.fps_x2
        # The overlap-add tail must fit inside one frame's worth of samples (stream_step relies on it).
        assert spf >= self.win - self.hop, (spf, self.win, self.hop)
        return spf

    @property
    def up_factor(self) -> int:
        assert self.samples_per_frame % self.hop == 0
        return self.samples_per_frame // self.hop

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def ola_tail(self) -> int:
        return self.win - self.hop


class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt block with strictly left (kernel-1) temporal padding; channels-last [B, T, C]."""

    def __init__(self, d: int, expand: int, kernel: int, ls_init: float):
        super().__init__()
        self.kernel = kernel
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, expand * d)
        self.pw2 = nn.Linear(expand * d, d)
        self.gamma = nn.Parameter(ls_init * torch.ones(d))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`state` [B, kernel-1, C] is the tail of the previous chunk (zeros at stream start)."""
        b, t, c = x.shape
        if state is None:
            state = x.new_zeros(b, self.kernel - 1, c)
        xc = torch.cat([state, x], dim=1)
        new_state = xc[:, -(self.kernel - 1):].detach() if self.kernel > 1 else state
        h = self.dw(xc.transpose(1, 2)).transpose(1, 2)  # valid conv -> length t
        h = self.norm(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + self.gamma * h, new_state


class FoldOLA(nn.Module):
    """Inverse STFT by windowed overlap-add (periodic Hann synthesis window, center=False semantics).

    The raw overlap-add numerator and the folded win^2 envelope (denominator) are exposed separately
    so that streaming can carry the numerator tail as state and slice the same envelope by absolute position.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        w = torch.hann_window(cfg.win, periodic=True, dtype=torch.float32)
        self.register_buffer("window", w, persistent=False)

    def frames_to_wave(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, T_col, win] -> raw overlap-add sum [B, (T_col-1)*hop + win] (unnormalised)."""
        b, t, win = frames.shape
        assert win == self.cfg.win
        fw = (frames * self.window).transpose(1, 2)  # [B, win, T]
        out_len = (t - 1) * self.cfg.hop + win
        return F.fold(
            fw, output_size=(1, out_len), kernel_size=(1, win), stride=(1, self.cfg.hop)
        ).squeeze(2).squeeze(1)

    def envelope(self, t_cols: int, device, dtype) -> torch.Tensor:
        """Folded win^2 envelope on the same grid [1, (T_col-1)*hop + win], floored at 1e-8."""
        w2 = (self.window * self.window)[None, :, None].expand(1, self.cfg.win, t_cols)
        out_len = (t_cols - 1) * self.cfg.hop + self.cfg.win
        env = F.fold(
            w2, output_size=(1, out_len), kernel_size=(1, self.cfg.win), stride=(1, self.cfg.hop)
        ).squeeze(2).squeeze(1)
        return env.to(device=device, dtype=dtype).clamp_min(1e-8)


class VocosDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.d_low)
        self.low = nn.ModuleList(
            [CausalConvNeXtBlock(cfg.d_low, cfg.expand, cfg.kernel, cfg.layer_scale_init)
             for _ in range(cfg.n_low)]
        )
        self.up = nn.Linear(cfg.d_low, cfg.up_factor * cfg.d_high)
        self.high = nn.ModuleList(
            [CausalConvNeXtBlock(cfg.d_high, cfg.expand, cfg.kernel, cfg.layer_scale_init)
             for _ in range(cfg.n_high)]
        )
        self.head_norm = nn.LayerNorm(cfg.d_high)
        self.head = nn.Linear(cfg.d_high, 2 * cfg.n_bins)
        with torch.no_grad():
            self.head.bias[: cfg.n_bins].fill_(cfg.mag_bias_init)
        self.ola = FoldOLA(cfg)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- spectral head (kept in fp32: exp / cos / sin of the phase must not run in half precision) --
    def _spectrum(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, T_col, d_high] -> complex spectrum [B, T_col, n_bins] with real DC and Nyquist bins."""
        cfg = self.cfg
        with torch.autocast(device_type=h.device.type, enabled=False):
            out = self.head(self.head_norm(h.float()))
            logmag, phase = out[..., : cfg.n_bins], out[..., cfg.n_bins:]
            mag = torch.exp(logmag.clamp(cfg.logmag_min, cfg.logmag_max))
            ux, uy = torch.cos(phase), torch.sin(phase)
            s = torch.complex(mag * ux, mag * uy)
        s = torch.cat(
            [s[..., :1].real.to(s.dtype) + 0j, s[..., 1:-1], s[..., -1:].real.to(s.dtype) + 0j],
            dim=-1,
        )
        return s

    def _frames(self, s: torch.Tensor) -> torch.Tensor:
        """Spectrum -> time frames [B, T_col, win] (irfft at n_fft, truncated to win)."""
        fr = torch.fft.irfft(s, n=self.cfg.n_fft, dim=-1)
        return fr[..., : self.cfg.win]

    def _trunk(self, latent: torch.Tensor, st: dict | None = None) -> torch.Tensor:
        cfg = self.cfg
        b, t, _ = latent.shape
        h = self.in_proj(latent)
        for i, blk in enumerate(self.low):
            if st is None:
                h, _ = blk(h)
            else:
                h, st["low"][i] = blk(h, st["low"][i])
        h = self.up(h).reshape(b, t * cfg.up_factor, cfg.d_high)
        for i, blk in enumerate(self.high):
            if st is None:
                h, _ = blk(h)
            else:
                h, st["high"][i] = blk(h, st["high"][i])
        return h

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B, T, latent_dim] -> waveform [B, 1, T * samples_per_frame] (offline)."""
        cfg = self.cfg
        t = latent.shape[1]
        h = self._trunk(latent)
        frames = self._frames(self._spectrum(h))
        wave = self.ola.frames_to_wave(frames)
        env = self.ola.envelope(frames.shape[1], wave.device, wave.dtype)
        wave = wave / env
        return wave[:, : t * cfg.samples_per_frame].unsqueeze(1)

    # -- streaming (bit-identical to forward from the first frame) --
    @torch.no_grad()
    def init_stream(self, batch: int, device=None) -> dict:
        cfg = self.cfg
        if device is None:
            device = self.in_proj.weight.device
        return {
            "low": [torch.zeros(batch, cfg.kernel - 1, cfg.d_low, device=device) for _ in self.low],
            "high": [torch.zeros(batch, cfg.kernel - 1, cfg.d_high, device=device) for _ in self.high],
            "ola_num": torch.zeros(batch, cfg.ola_tail, device=device),
            "col_idx": 0,
        }

    @torch.no_grad()
    def stream_step(self, latent: torch.Tensor, st: dict) -> torch.Tensor:
        """One latent frame [B, 1, latent_dim] -> [B, samples_per_frame] samples; updates `st` in place."""
        cfg = self.cfg
        b = latent.shape[0]
        assert latent.shape[1] == 1, latent.shape
        h = self._trunk(latent, st)
        frames = self._frames(self._spectrum(h))  # [B, up_factor, win]
        chunk = self.ola.frames_to_wave(frames)  # [B, (up_factor-1)*hop + win]
        tail = cfg.ola_tail
        n_emit = cfg.samples_per_frame
        num = torch.cat([st["ola_num"], torch.zeros(b, n_emit, device=chunk.device, dtype=chunk.dtype)], 1)
        num = num + chunk[:, : tail + n_emit]
        st["ola_num"] = chunk[:, n_emit:].clone()
        emit = num[:, :n_emit]
        start = st["col_idx"] * cfg.hop
        env = self._env_slice(start, n_emit, emit.device, emit.dtype)
        st["col_idx"] += cfg.up_factor
        return emit / env

    def _env_slice(self, start: int, n: int, device, dtype) -> torch.Tensor:
        cfg = self.cfg
        need_cols = (start + n - 1) // cfg.hop + cfg.win // cfg.hop + 2
        env = self.ola.envelope(need_cols, device, dtype)
        return env[0, start : start + n]
