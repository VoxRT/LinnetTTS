# linnet-tts (Python)

Python package for the LinnetTTS native runtime: lightweight on-device text-to-speech at 24 kHz, streaming,
fp16 weights, no PyTorch. The extension module is built from the Rust crate `runtime/linnet-tts` with
[maturin](https://www.maturin.rs) (abi3, one wheel per platform for every CPython >= 3.9).

```python
import linnet_tts, soundfile as sf

model = linnet_tts.Model("linnet-en-35m.fp16.safetensors", "tokenizer.model")
voice = model.load_voice("voice.npy")                     # [T, 48] float32 .npy

audio = model.synthesize("Hello there.", voice, temp=0.3, cfg=2.0, seed=0)
sf.write("hello.wav", audio, model.sample_rate)           # float32, 24 kHz

session = model.session(voice, seed=0)
session.prime("Streaming gives the first 50 ms of audio after one step.")
for chunk in session:                                     # 80 ms per LM step
    play(chunk)
```

`Voice.from_array(np.ndarray)` wraps in-memory `[T, 48]` latents. The chunk sizes follow
`docs/streaming-contract.md`: the first chunk is `samples_per_step - output_delay` samples, later ones
`samples_per_step` (1920), the last one `output_delay` (720) zeros, and the concatenation of all chunks is
sample-for-sample identical to `synthesize` with the same parameters and seed.

## Build

```sh
cd bindings/python
maturin build --release                     # wheel in target/wheels/
maturin develop --release                   # or: install into the active venv
```

The release profile uses fat LTO and strips symbols; the wheel is ~0.4 MB. Runtime dependency: `numpy`.

## Test

The tests need the model assets (gitignored). Point `LINNET_ASSETS` at a directory holding
`linnet-en-35m.fp16.safetensors`, `tokenizer.model` and `voices/linnet-en-f1.npy` (default: `../../local`;
`local/` = the unpacked weights release, see the root README):

```sh
uv venv /tmp/linnet-venv && . /tmp/linnet-venv/bin/activate
uv pip install target/wheels/linnet_tts-*.whl pytest
pytest tests
```

A plain `python -m venv /tmp/linnet-venv` followed by `pip install target/wheels/linnet_tts-*.whl pytest` works
just as well where `uv` is not installed.
