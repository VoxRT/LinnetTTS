# linnet-tts (Python)

Python package for the LinnetTTS native runtime: lightweight on-device text-to-speech at 24 kHz, streaming,
fp16 weights, no PyTorch. The extension module is built from the Rust crate `runtime/linnet-tts` with
[maturin](https://www.maturin.rs) (abi3, one wheel per platform for every CPython >= 3.9).

## Install

```sh
pip install linnet-tts soundfile      # soundfile only for writing WAV files in the snippets
```

Wheels on PyPI: manylinux 2.17 x86_64 and aarch64, macOS 11+ on Apple silicon; no Windows, Intel-macOS or source
distribution in 0.1.1 (build from the repository on other platforms, see "Build"). The model files are not in the
wheel: download the weights release https://github.com/VoxRT/LinnetTTS/releases/tag/weights-v1 (the fp16 weights,
`tokenizer.model` and a voice `.npy`). They are licensed under the Linnet Weights License
(https://github.com/VoxRT/LinnetTTS/blob/main/WEIGHTS_LICENSE.md), not Apache-2.0. Source, docs and the streaming
contract: https://github.com/VoxRT/LinnetTTS.

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

`Voice.from_array(np.ndarray)` wraps in-memory `[T, 48]` latents. The chunk sizes follow the streaming contract
(`docs/streaming-contract.md` in the repository): the first chunk is `samples_per_step - output_delay` samples, later ones
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

With the wheel from PyPI: clone the repository, install `pytest`, and run `pytest bindings/python/tests` from its root (the
tests import the installed `linnet_tts`). The tests need the model assets (gitignored). Point `LINNET_ASSETS` at a directory holding
`linnet-en-35m.fp16.safetensors`, `tokenizer.model` and `voices/linnet-en-f1.npy` (default: `../../local`;
`local/` = the unpacked weights release, see the root README at https://github.com/VoxRT/LinnetTTS):

```sh
uv venv /tmp/linnet-venv && . /tmp/linnet-venv/bin/activate
uv pip install target/wheels/linnet_tts-*.whl pytest
pytest tests
```

A plain `python -m venv /tmp/linnet-venv` followed by `pip install target/wheels/linnet_tts-*.whl pytest` works
just as well where `uv` is not installed.
