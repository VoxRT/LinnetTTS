"""Bit-for-bit check of the PyTorch package against the reference input/output vectors in tests/goldens/ref.

Requires the released weights and tokenizer:
    LINNET_WEIGHTS=/path/linnet-en-35m.safetensors LINNET_TOKENIZER=/path/lm_sp.model pytest tests/test_ref_goldens.py -q
Without them the tests are skipped. Set LINNET_GOLDEN_TOL (default 0.0) to allow summation-order noise on a different CPU / torch build.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

GOLD = Path(__file__).parent / "goldens" / "ref"
TOL = float(os.environ.get("LINNET_GOLDEN_TOL", "0.0"))

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LINNET_WEIGHTS") and os.environ.get("LINNET_TOKENIZER")),
    reason="LINNET_WEIGHTS / LINNET_TOKENIZER not set",
)


def _load_bin(entry: dict) -> np.ndarray:
    data = (GOLD / entry["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], f"golden file corrupted: {entry['file']}"
    return np.frombuffer(data, dtype="<f4").reshape(entry["shape"]).copy()


def _compare(name: str, got: np.ndarray, want: np.ndarray) -> None:
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    if TOL == 0.0:
        same = hashlib.sha256(got.astype("<f4").tobytes()).hexdigest() == hashlib.sha256(want.astype("<f4").tobytes()).hexdigest()
        assert same, f"{name}: not bit-identical, max|d| = {np.abs(got - want).max():.3e}"
    else:
        d = float(np.abs(got - want).max())
        assert d <= TOL, f"{name}: max|d| = {d:.3e} > {TOL}"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    from linnet_tts_torch.synth import Linnet

    torch.set_num_threads(4)
    return Linnet.from_files(os.environ["LINNET_WEIGHTS"], os.environ["LINNET_TOKENIZER"])


class _RecordedNoise:
    """Replaces torch.randn so that each call returns the next recorded unit-normal draw."""

    def __init__(self, rows: np.ndarray):
        self.rows = torch.from_numpy(rows)
        self.i = 0
        self._orig = torch.randn

    def __enter__(self):
        def fake(*shape, **kw):
            assert self.i < len(self.rows), "more noise draws than recorded"
            n = self.rows[self.i].reshape(1, -1).to(dtype=kw.get("dtype", torch.float32))
            self.i += 1
            assert tuple(n.shape) == tuple(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
            return n

        torch.randn = fake
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig


def test_decoder_only(model, manifest):
    g = manifest["decoder_only"]
    z24 = torch.from_numpy(_load_bin(g["latents_24"]))  # [60, 24]
    with torch.no_grad():
        raw = model.decoder(z24[None])
    raw = raw.reshape(-1).numpy()
    _compare("decoder raw", raw, _load_bin(g["wav_raw"]).reshape(-1))
    with torch.no_grad():
        shifted = model.decode(z24.reshape(-1, 48)).reshape(-1).numpy()
    _compare("decoder shifted", shifted, _load_bin(g["wav_shifted"]).reshape(-1))


def test_voice_roundtrip(model, manifest):
    for name, v in manifest["voices"].items():
        lat = torch.from_numpy(_load_bin(v["latent"]))  # [Tv, 48]
        with torch.no_grad():
            rt = model.decode(lat).reshape(-1).numpy()
        _compare(f"roundtrip {name}", rt, _load_bin(v["roundtrip_wav"]).reshape(-1))


@pytest.mark.parametrize("case", sorted(json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))["cases"]))
def test_lm_case(model, manifest, case):
    c = manifest["cases"][case]
    p = c.get("params", manifest["params"])
    assert model.tokenizer.encode(c["text"]) == c["tokens"], f"{case}: tokens differ"
    voice = torch.from_numpy(_load_bin(manifest["voices"][c["voice"]]["latent"]))
    noise = _load_bin(c["noise_unit"])  # [steps, 48]
    with _RecordedNoise(noise) as rec, torch.no_grad():
        lat = model.generate(c["text"], voice, temp=p["temp"], cfg=p["cfg"], max_frames=p["max_frames"], eos_threshold=p["eos_threshold"])
    assert rec.i == c["steps_drawn"], f"{case}: {rec.i} draws, reference made {c['steps_drawn']}"
    lat = lat.reshape(-1, 48).numpy()
    _compare(f"{case} latents", lat, _load_bin(c["latents_48"]))
    with torch.no_grad():
        wav = model.decode(torch.from_numpy(lat)).reshape(-1).numpy()
    _compare(f"{case} wav", wav, _load_bin(c["wav"]).reshape(-1))
