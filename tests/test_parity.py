"""Parity of the LinnetTTS package against a reference implementation.

The reference stack is private and not part of this repository, so these tests are skipped unless
the environment points at it:

    LINNET_REF_LOADER   python file defining `load_reference() -> dict` with four callables:
                          tokenize(text) -> list[int]
                          generate(tokens: list[int], voice: Tensor[Tv,48], temp, cfg, max_frames) -> Tensor[T,48]
                          decoder(z: Tensor[B,T,24]) -> Tensor[B,1,T*960]          (raw codec decoder)
                          decode(latents: Tensor[T,48]) -> Tensor[T*1920]          (with output-delay handling)
                        `generate` must draw exactly one torch.randn(1, 48) from the global RNG per step.
    LINNET_WEIGHTS      the released fp32 safetensors (or a conversion by the maintainers' tools/convert_from_checkpoint.py)
    LINNET_TOKENIZER    SentencePiece model
    LINNET_VOICE        voice-prompt latents .npy [Tv, 48]

Always-on tests (no reference needed): decoder streaming == offline, deterministic LM sampling.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from linnet_tts_torch import DecoderConfig, FlowLM, Linnet, LMConfig, VocosDecoder  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 1))

TEXT = "The quick brown fox jumps over the lazy dog."
TEMP, CFG, MAXF, SEED = 0.3, 2.0, 60, 1234


def _mad(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


# ----------------------------------------------------------------------------- self-contained


def test_decoder_stream_matches_offline():
    torch.manual_seed(0)
    dec = VocosDecoder(DecoderConfig()).eval()
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(0.05 * torch.randn_like(p))
        z = torch.randn(2, 9, 24)
        off = dec(z)
        st = dec.init_stream(2)
        stream = torch.cat([dec.stream_step(z[:, i : i + 1], st) for i in range(z.shape[1])], 1)[:, None]
    assert off.shape == (2, 1, 9 * 960)
    # overlap-add summation order differs between fold (offline) and the carried tail (stream): fp32 noise only
    assert _mad(off, stream) <= 1e-5 * max(1.0, float(off.abs().max()))


def test_lm_generate_is_deterministic_and_shaped():
    torch.manual_seed(0)
    lm = FlowLM(LMConfig(n_layers=1, head_depth=1)).eval()
    with torch.no_grad():
        for p in lm.parameters():
            p.normal_(0, 0.02)
        lm.eos.bias.fill_(-100.0)  # never stop
    toks = torch.tensor([5, 6, 7])
    voice = torch.randn(4, 48)
    a = torch.stack(list(lm.generate(toks, voice, max_frames=5, generator=torch.Generator().manual_seed(1))))
    b = torch.stack(list(lm.generate(toks, voice, max_frames=5, generator=torch.Generator().manual_seed(1))))
    assert a.shape == (5, 48)
    assert _mad(a, b) == 0.0


# ----------------------------------------------------------------------------- against the reference


@pytest.fixture(scope="module")
def ref():
    needed = {k: os.environ.get(k) for k in ("LINNET_REF_LOADER", "LINNET_WEIGHTS", "LINNET_TOKENIZER", "LINNET_VOICE")}
    missing = [k for k, v in needed.items() if not v or not Path(v).exists()]
    if missing:
        pytest.skip(f"reference inputs not available: {missing}")
    spec = importlib.util.spec_from_file_location("linnet_ref_loader", needed["LINNET_REF_LOADER"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    r = mod.load_reference()
    r["lin"] = Linnet.from_files(needed["LINNET_WEIGHTS"], needed["LINNET_TOKENIZER"])
    r["voice"] = Linnet.load_voice(needed["LINNET_VOICE"])
    return r


def test_tokens_match(ref):
    assert ref["tokenize"](TEXT) == ref["lin"].tokenizer.encode(TEXT)


def test_decoder_parity(ref):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(1, 40, 24, generator=g) * 1.5
    with torch.no_grad():
        assert _mad(ref["decoder"](z), ref["lin"].decoder(z)) <= 1e-5
        assert _mad(ref["decode"](z.reshape(40 // 2, 48)), ref["lin"].decode(z.reshape(40 // 2, 48))) <= 1e-5


def _gen_both(ref, max_frames=MAXF):
    toks = ref["lin"].tokenizer.encode(TEXT)
    torch.manual_seed(SEED)
    with torch.no_grad():
        lat_ref = ref["generate"](toks, ref["voice"], TEMP, CFG, max_frames)
    torch.manual_seed(SEED)
    lat_new = ref["lin"].generate(TEXT, ref["voice"], temp=TEMP, cfg=CFG, max_frames=max_frames)
    return lat_ref, lat_new


def test_lm_and_e2e_parity(ref):
    lat_ref, lat_new = _gen_both(ref)
    assert lat_ref.shape == lat_new.shape and lat_ref.shape[0] > 0
    assert _mad(lat_ref, lat_new) <= 1e-4
    with torch.no_grad():
        wav_ref = ref["decode"](lat_ref)
    wav_new = ref["lin"].decode(lat_new)
    assert _mad(wav_ref, wav_new) <= 1e-4
    torch.manual_seed(SEED)
    wav_stream = torch.cat(list(ref["lin"].stream(TEXT, ref["voice"], temp=TEMP, cfg=CFG, max_frames=MAXF)))
    assert _mad(wav_stream, wav_new) <= 1e-4  # fp32 overlap-add ordering, see test_decoder_stream_matches_offline


def test_parity_detects_perturbation(ref):
    """'Seen red': a 1e-3 bump on one trunk tensor must break latent parity."""
    w = ref["lin"].lm.trunk.layers[3].ff_in.weight
    orig = w.detach().clone()
    with torch.no_grad():
        w.add_(1e-3)
    try:
        lat_ref, lat_new = _gen_both(ref, max_frames=20)
        m = min(lat_ref.shape[0], lat_new.shape[0])
        assert m > 0
        assert _mad(lat_ref[:m], lat_new[:m]) > 1e-3
    finally:
        with torch.no_grad():
            w.copy_(orig)
