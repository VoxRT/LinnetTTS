"""Tests of the installed `linnet_tts` wheel against the local model assets."""

import os
import pathlib
import time

import numpy as np
import pytest

import linnet_tts

ASSETS = pathlib.Path(os.environ.get("LINNET_ASSETS", pathlib.Path(__file__).resolve().parents[3] / "local"))
WEIGHTS_FP16 = ASSETS / "linnet-en-35m.fp16.safetensors"
WEIGHTS_FP32 = ASSETS / "linnet-en-35m.safetensors"
TOKENIZER = ASSETS / "tokenizer.model"
VOICE = ASSETS / "voice_hifitts2_val0.npy"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS_FP16.exists() and TOKENIZER.exists() and VOICE.exists()),
    reason=f"model assets not found under {ASSETS} (set LINNET_ASSETS)",
)


@pytest.fixture(scope="module")
def model():
    return linnet_tts.Model(WEIGHTS_FP16, TOKENIZER)


@pytest.fixture(scope="module")
def voice(model):
    return model.load_voice(VOICE)


def test_version_and_constants(model):
    assert linnet_tts.__version__
    assert model.sample_rate == 24000
    assert model.samples_per_step == 1920
    assert model.output_delay == 720
    assert model.latent_dim == 48
    assert model.tokenize("Hello there.")


def test_voice_roundtrip(model, voice):
    arr = voice.to_array()
    assert arr.dtype == np.float32 and arr.ndim == 2 and arr.shape[1] == 48
    v2 = linnet_tts.Voice.from_array(arr.astype(np.float64))  # dtype conversion on the way in
    assert v2.frames == voice.frames and v2.dim == 48
    np.testing.assert_array_equal(v2.to_array(), arr)
    a = model.synthesize("Hello there.", voice, seed=3)
    b = model.synthesize("Hello there.", v2, seed=3)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        linnet_tts.Voice.from_array(np.zeros(48, dtype=np.float32))


def test_synthesize_length_is_whole_steps(model, voice):
    session = model.session(voice, seed=0)
    session.prime("Hello there.")
    chunks = list(session)
    frames = session.frames
    audio = model.synthesize("Hello there.", voice, seed=0)
    assert audio.dtype == np.float32
    assert frames >= 1
    assert len(audio) == frames * model.samples_per_step
    # Chunk layout per the streaming contract.
    assert len(chunks) == frames + 1
    assert len(chunks[0]) == model.samples_per_step - model.output_delay
    assert all(len(c) == model.samples_per_step for c in chunks[1:-1])
    assert len(chunks[-1]) == model.output_delay and not chunks[-1].any()


def test_stream_concat_equals_synthesize_bit_exact(model, voice):
    text = "The quick brown fox jumps over the lazy dog."
    offline = model.synthesize(text, voice, temp=0.3, cfg=2.0, seed=7)
    session = model.session(voice, temp=0.3, cfg=2.0, seed=7)
    session.prime(text)
    streamed = np.concatenate([c for c in session])
    assert streamed.shape == offline.shape
    np.testing.assert_array_equal(streamed, offline)
    assert session.finished and session.step() is None


def test_seed_is_deterministic_and_matters(model, voice):
    a = model.synthesize("Hello there.", voice, seed=11)
    b = model.synthesize("Hello there.", voice, seed=11)
    c = model.synthesize("Hello there.", voice, seed=12)
    np.testing.assert_array_equal(a, b)
    assert len(a) != len(c) or not np.array_equal(a, c)
    # temp = 0 is noise free: identical across seeds.
    d0 = model.synthesize("Hello there.", voice, temp=0.0, seed=1)
    d1 = model.synthesize("Hello there.", voice, temp=0.0, seed=2)
    np.testing.assert_array_equal(d0, d1)


def test_fp16_hello_has_sane_level(model, voice):
    audio = model.synthesize("Hello there.", voice)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.5
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 0.02, f"rms {rms:.4f}"
    # Roughly 0.3 .. 3 s for two words.
    assert 0.3 <= len(audio) / model.sample_rate <= 3.0


def test_session_reprime_resets(model, voice):
    session = model.session(voice, seed=5)
    session.prime("First sentence here.")
    first = np.concatenate(list(session))
    session.prime("First sentence here.")
    again = np.concatenate(list(session))
    np.testing.assert_array_equal(first, again)


def test_decode_matches_streamed_latents(model, voice):
    session = model.session(voice, seed=2)
    session.prime("Hello there.")
    latents, chunks = [], []
    while True:
        chunk = session.step()
        if chunk is None:
            break
        chunks.append(chunk)
        if not session.finished:
            latents.append(session.latent())
    lat = np.stack(latents)
    assert lat.shape == (session.frames, model.latent_dim)
    offline = model.decode(lat)
    streamed = np.concatenate(chunks)
    assert offline.shape == streamed.shape
    np.testing.assert_allclose(offline, streamed, atol=1e-5)


def test_errors_are_python_exceptions(model):
    with pytest.raises(OSError):
        linnet_tts.Model(ASSETS / "missing.safetensors", TOKENIZER)
    narrow = linnet_tts.Voice.from_array(np.zeros((4, 47), dtype=np.float32))  # wrong width: accepted here...
    with pytest.raises(ValueError):
        model.session(narrow)  # ...rejected by the model
    with pytest.raises(ValueError):
        model.synthesize("x", narrow)
    with pytest.raises(ValueError):
        model.session(model.load_voice(VOICE), max_frames=0)


@pytest.mark.skipif(not WEIGHTS_FP32.exists(), reason="fp32 weights not present")
def test_fp32_and_fp16_agree_closely(voice):
    m16 = linnet_tts.Model(WEIGHTS_FP16, TOKENIZER)
    m32 = linnet_tts.Model(WEIGHTS_FP32, TOKENIZER)
    a = m16.synthesize("Hello there.", voice, temp=0.0)
    b = m32.synthesize("Hello there.", voice, temp=0.0)
    # Same frame count is expected but not guaranteed near the EOS threshold.
    n = min(len(a), len(b))
    assert n > 0
    assert abs(len(a) - len(b)) <= 2 * 1920
    assert np.corrcoef(a[:n].astype(np.float64), b[:n].astype(np.float64))[0, 1] > 0.9


def test_perf_report(model, voice):
    """Not an assertion: prints the synthesis speed of a ~6 s utterance."""
    text = (
        "Lightweight on-device speech synthesis runs comfortably on a single core, "
        "streaming eighty milliseconds of audio per step without any lookahead."
    )
    model.synthesize(text, voice, seed=0)  # warm-up
    t0 = time.perf_counter()
    audio = model.synthesize(text, voice, seed=0)
    dt = time.perf_counter() - t0
    secs = len(audio) / model.sample_rate
    print(f"\nperf: {secs:.2f} s of audio in {dt*1e3:.0f} ms, RTF {100*dt/secs:.1f} %")
    assert secs > 0
