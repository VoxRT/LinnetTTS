"""`Linnet`: text + voice latents -> 24 kHz speech (offline and streaming)."""

from __future__ import annotations

import math

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .decoder import SAMPLE_RATE, DecoderConfig, VocosDecoder
from .lm import FlowLM, LMConfig
from .tokenizer import Tokenizer

WEIGHTS_FORMAT = "linnet-tts/1"
# The codec decoder was trained to reproduce its input delayed by this many samples; the synthesis
# output is advanced by the same amount (first `OUTPUT_DELAY` samples dropped, zeros appended) so
# that it lines up with the latent grid.
OUTPUT_DELAY = 720


def _lm_frames_to_codec(latents: torch.Tensor) -> torch.Tensor:
    """[B, S, 48] LM steps -> [B, 2S, 24] codec frames (first half = even frame, second half = odd)."""
    b, s, d = latents.shape
    return latents.reshape(b, s * 2, d // 2)


class Linnet:
    def __init__(self, lm: FlowLM, decoder: VocosDecoder, tokenizer: Tokenizer):
        self.lm = lm.eval()
        self.decoder = decoder.eval()
        self.tokenizer = tokenizer
        self.sample_rate = SAMPLE_RATE

    # ------------------------------------------------------------------ weights
    @staticmethod
    def build_from_config(lm_cfg: LMConfig, dec_cfg: DecoderConfig) -> tuple[FlowLM, VocosDecoder]:
        return FlowLM(lm_cfg), VocosDecoder(dec_cfg)

    @classmethod
    def from_files(cls, weights: str | Path, tokenizer: str | Path, device: str | torch.device = "cpu") -> "Linnet":
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        if meta.get("format") != WEIGHTS_FORMAT:
            raise ValueError(f"unexpected weights format {meta.get('format')!r}, want {WEIGHTS_FORMAT!r}")
        lm_cfg = LMConfig(**json.loads(meta["lm_config"]))
        dec_cfg = DecoderConfig(**json.loads(meta["decoder_config"]))
        lm, dec = cls.build_from_config(lm_cfg, dec_cfg)
        tensors = load_file(str(weights), device="cpu")
        lm_sd = {k[len("lm."):]: v for k, v in tensors.items() if k.startswith("lm.")}
        dec_sd = {k[len("decoder."):]: v for k, v in tensors.items() if k.startswith("decoder.")}
        lm.load_state_dict(lm_sd, strict=True)
        dec.load_state_dict(dec_sd, strict=True)
        lm.to(device)
        dec.to(device)
        return cls(lm, dec, Tokenizer(tokenizer, expected_vocab=lm_cfg.vocab_size))

    @staticmethod
    def save_weights(path: str | Path, lm: FlowLM, decoder: VocosDecoder, extra_meta: dict | None = None):
        tensors = {f"lm.{k}": v.detach().cpu().contiguous() for k, v in lm.state_dict().items()}
        tensors.update({f"decoder.{k}": v.detach().cpu().contiguous() for k, v in decoder.state_dict().items()})
        meta = {
            "format": WEIGHTS_FORMAT,
            "lm_config": json.dumps(dataclasses.asdict(lm.cfg)),
            "decoder_config": json.dumps(dataclasses.asdict(decoder.cfg)),
            "sample_rate": str(SAMPLE_RATE),
            "output_delay": str(OUTPUT_DELAY),
        }
        if extra_meta:
            meta.update({k: str(v) for k, v in extra_meta.items()})
        save_file(tensors, str(path), metadata=meta)

    @staticmethod
    def load_voice(path: str | Path) -> torch.Tensor:
        """Voice-prompt latents [Tv, 48] from a .npy file; the voice files are distributed with the weights release
        (this package does not compute them)."""
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] != 48:
            raise ValueError(f"voice latents must be [T, 48], got {arr.shape}")
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    # ------------------------------------------------------------------ inference
    @property
    def device(self) -> torch.device:
        return self.lm.device

    def _voice(self, voice_latents) -> torch.Tensor:
        v = torch.as_tensor(np.asarray(voice_latents) if not torch.is_tensor(voice_latents) else voice_latents)
        ld = self.lm.cfg.latent_dim
        if v.ndim != 2 or v.shape[1] != ld or v.shape[0] == 0:
            raise ValueError(f"voice latents must be [T, {ld}] with T >= 1, got {tuple(v.shape)}")
        return v.to(self.device, torch.float32)

    @staticmethod
    def _check(text: str, temp: float, cfg: float, max_frames: int) -> None:
        """The same argument rules as the Rust runtime and the `linnet-tts` wheel."""
        if not text or not text.strip():
            raise ValueError("empty input: the text is blank")
        if not (temp >= 0.0 and math.isfinite(temp)):
            raise ValueError(f"temp {temp} must be finite and >= 0")
        if not math.isfinite(cfg):
            raise ValueError("cfg must be finite")
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")

    @torch.no_grad()
    def generate(
        self,
        text: str,
        voice_latents,
        temp: float = 0.3,
        cfg: float = 2.0,
        max_frames: int = 300,
        eos_threshold: float = -1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample LM latents for `text` -> [T, 48] in codec space (T <= max_frames)."""
        self._check(text, temp, cfg, max_frames)
        tokens = self.tokenizer.encode_tensor(text)
        frames = list(
            self.lm.generate(
                tokens, self._voice(voice_latents), temp=temp, cfg=cfg, max_frames=max_frames,
                eos_threshold=eos_threshold, generator=generator,
            )
        )
        if not frames:
            return torch.zeros(0, self.lm.cfg.latent_dim, device=self.device)
        return torch.stack(frames, dim=0)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[T, 48] LM latents -> waveform [T * 1920] float32 at 24 kHz (output delay compensated)."""
        if latents.shape[0] == 0:
            return torch.zeros(0, device=self.device)
        z = _lm_frames_to_codec(latents.to(self.device, torch.float32)[None])
        wav = self.decoder(z)[:, 0]  # [1, 2T * 960]
        wav = F.pad(wav[:, OUTPUT_DELAY:], (0, OUTPUT_DELAY))
        return wav[0]

    @torch.no_grad()
    def synthesize(self, text: str, voice_latents, **kw) -> torch.Tensor:
        return self.decode(self.generate(text, voice_latents, **kw))

    @torch.no_grad()
    def stream(
        self,
        text: str,
        voice_latents,
        temp: float = 0.3,
        cfg: float = 2.0,
        max_frames: int = 300,
        eos_threshold: float = -1.0,
        generator: torch.Generator | None = None,
        pad_tail: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield audio chunks (float32, 24 kHz) as the LM produces each step.

        Concatenating the chunks gives exactly `decode(generate(...))` when `pad_tail` is set:
        the first chunk is shorter by `OUTPUT_DELAY` samples and the final chunk is the delay's
        worth of zeros.
        """
        self._check(text, temp, cfg, max_frames)
        tokens = self.tokenizer.encode_tensor(text)
        st = self.decoder.init_stream(1, self.device)
        skip = OUTPUT_DELAY
        n_steps = 0
        for lat in self.lm.generate(
            tokens, self._voice(voice_latents), temp=temp, cfg=cfg, max_frames=max_frames,
            eos_threshold=eos_threshold, generator=generator,
        ):
            n_steps += 1
            z = _lm_frames_to_codec(lat.view(1, 1, -1))  # [1, 2, 24]
            pieces = [self.decoder.stream_step(z[:, i : i + 1], st)[0] for i in range(z.shape[1])]
            chunk = torch.cat(pieces)
            if skip:
                cut = min(skip, chunk.shape[0])
                chunk = chunk[cut:]
                skip -= cut
            if chunk.shape[0]:
                yield chunk
        if pad_tail and n_steps:
            yield torch.zeros(OUTPUT_DELAY, device=self.device)
