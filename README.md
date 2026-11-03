# LinnetTTS

Lightweight on-device text-to-speech by VoxRT: an autoregressive flow LM (1 NFE per frame) over a 25 Hz continuous-latent neural codec,
with a Vocos-style decoder. ~41M parameters end to end (31.5M LM + 9.9M decoder), streaming synthesis, built-in voices, fp16 weights.

- `python/` — `linnet_tts_torch`: reference model definition and inference in PyTorch (no training code; see `python/README.md`). The on-device Python package `linnet-tts` is built from the Rust runtime in `bindings/python/`.
- `runtime/` — Rust runtime: `linnet-ops` (only the kernels this model uses) and `linnet-tts` (model, streaming session, C API).
- `bindings/` — platform wrappers: Android AAR, iOS Swift package (xcframework), Linux C library and Python wheel. Browser (WASM) is deferred to a later release.
- `build/` — build and packaging scripts.
- `tests/goldens/` — shared golden vectors; Python and Rust must agree on them.
- `docs/` — model card, weight container format, streaming contract.

Code is licensed under Apache-2.0 (see `LICENSE`). Model weights are distributed separately under the Linnet Weights License
(see `WEIGHTS_LICENSE.md`); the codec encoder used for voice cloning is not part of this release. The test fixtures in `tests/goldens/` include
copies of the two built-in voices; `tests/goldens/NOTICE.md` states their license and the permission to redistribute them with the repository.

The model is called `linnet-en-35m` after its roughly 35 M parameters outside the 6001×512 text token table (41.4 M in total:
flow LM 31.5 M including the 3.1 M text table, decoder 9.9 M); the fp16 weights file is 83 MB.

## Quick start

1. **Weights.** Download the `weights-v1` release (see "Weights" below) and unpack it into `local/` at the repository root.
   Every test, script and README in this repository assumes this layout:

   ```
   local/linnet-en-35m.fp16.safetensors    runtime weights (what the Rust runtime and the bindings load)
   local/linnet-en-35m.safetensors         fp32 reference weights (PyTorch package, golden tests)
   local/tokenizer.model                   SentencePiece model
   local/voices/linnet-en-f1.npy           built-in voices ([T, 48] float32 .npy); linnet-en-m1.npy as well
   ```

   `local/` is gitignored: nothing you put there is committed.

2. **Rust runtime.** The `synth` example takes the weights, the tokenizer, a voice, the text and an output WAV, followed by
   the optional `temp` (0.3), `cfg` (2.0), `seed` (0) and `threads` (0 = automatic):

   ```sh
   cd runtime
   cargo run --release --example synth -- ../local/linnet-en-35m.fp16.safetensors ../local/tokenizer.model \
       ../local/voices/linnet-en-f1.npy "Hello from Linnet." hello.wav
   ```

3. **Python wheel** (the Rust runtime as a Python extension, no PyTorch): `pip install linnet-tts`, or build it with
   `cd bindings/python && maturin build --release` and install the wheel from `target/wheels/`.

   ```python
   import linnet_tts, soundfile as sf

   model = linnet_tts.Model("local/linnet-en-35m.fp16.safetensors", "local/tokenizer.model")
   voice = model.load_voice("local/voices/linnet-en-f1.npy")
   audio = model.synthesize("Hello from Linnet.", voice, temp=0.3, cfg=2.0, seed=0)   # np.float32 @ 24 kHz
   sf.write("hello.wav", audio, model.sample_rate)
   ```

   Streaming: `session = model.session(voice, seed=0)`, `session.prime(text)`, then iterate `for chunk in session`
   (80 ms per step). See `bindings/python/README.md`.

4. **PyTorch reference** (`python/`, fp32, bit-exact against the training-side vectors, not optimised for speed):
   `pip install ./python`.

   ```python
   import soundfile as sf
   from linnet_tts_torch import Linnet

   lin = Linnet.from_files("local/linnet-en-35m.safetensors", "local/tokenizer.model")
   voice = Linnet.load_voice("local/voices/linnet-en-f1.npy")
   wav = lin.synthesize("Hello from Linnet.", voice, temp=0.3, cfg=2.0)        # torch.float32 [n] @ 24 kHz
   sf.write("hello.wav", wav.numpy(), lin.sample_rate)
   ```

   See `python/README.md`.

5. **Tests.** With the files in `local/` the tests find them on their own; the environment variables below redirect
   them (one file each) or the whole directory:

   - `LINNET_WEIGHTS` — fp32 weights for the Rust golden tests and `tests/test_ref_goldens.py` (default `local/linnet-en-35m.safetensors`)
   - `LINNET_WEIGHTS_FP16` — fp16 weights for the Rust golden tests (default `local/linnet-en-35m.fp16.safetensors`)
   - `LINNET_TOKENIZER` — SentencePiece model (default `local/tokenizer.model`)
   - `LINNET_VOICE` — voice file for the Rust golden tests (default `local/voices/linnet-en-f1.npy`)
   - `LINNET_ASSETS` — directory with all of the above for the wheel, Android and iOS tests (default `local/`)

   `cd runtime && cargo test --release` runs the Rust suite including the golden comparison against `tests/goldens/`
   (a missing input skips a golden test; `cargo test -- --nocapture` shows the `SKIPPED` line);
   `pytest bindings/python/tests` tests the installed wheel; `LINNET_WEIGHTS=local/linnet-en-35m.safetensors
   LINNET_TOKENIZER=local/tokenizer.model pytest tests/test_ref_goldens.py` checks the PyTorch package against
   `tests/goldens/ref`; `python tools/make_goldens.py --weights local/linnet-en-35m.safetensors --tokenizer
   local/tokenizer.model --voice local/voices/linnet-en-f1.npy` regenerates `tests/goldens/{hello,meeting}`.

## Performance

Rust runtime, fp16 weights, one step = 80 ms of audio (an LM step plus two decoder frames). Measured without real-time pacing;
the audio is bit-identical for every thread count and across all these devices.

| device | threads | per step | real-time factor |
|---|---|---|---|
| Apple M5 (macOS) | 1 | 3.3 ms | 5 % |
| iPhone 13 Pro Max, A15 | 1 (auto) / 2 | 5.5 / 4.6 ms | 8.3 / 6.6 % |
| iPhone 11, A13 | 1 (auto) | 6.7–7.1 ms | 9 % |
| desktop x86-64, AVX2 | 1 | 8.8 ms | 11 % |
| Snapdragon 662 (Redmi 9T), Cortex-A73 | 1 / 4 (auto) | 42.4 / 14.3 ms | 62 / 21 % |
| Raspberry Pi Zero 2 W, Cortex-A53 1 GHz | 1 / 4 (auto) | 158 / 59–60 ms | 242 / 88 % |

First audio arrives 50–120 ms after the text on the phones and about 250 ms on the Snapdragon 662 (the prime pass plus one step).
Peak memory is about 160 MB on Linux. Details, the in-app numbers and the effect of playback pacing: `docs/model-card.md`
("Performance") and `bindings/README.md` ("Threads", "Playback and step time").

## Weights

Model weights are not in this repository and are not bundled in any package. Download them from the GitHub release
`weights-v1` of https://github.com/VoxRT/LinnetTTS (mirror: https://huggingface.co/VoxRT/linnet-en-35m):

| file | purpose |
|---|---|
| `linnet-en-35m.fp16.safetensors` | runtime weights (fp16 matrices, ~83 MB); what the Rust runtime and every binding load |
| `linnet-en-35m.safetensors` | fp32 reference weights (~166 MB) for `linnet_tts_torch` and the golden tests |
| `tokenizer.model` | SentencePiece model |
| `voices/<name>.npy` | built-in voices: float32 `[T, 48]` voice-prompt latents |
| `SHA256SUMS` | checksums of the files above; verify with `sha256sum -c SHA256SUMS` before use |
| `WEIGHTS_LICENSE.md`, `model-card.md` | license and model card of the weights |

The runtime needs three files: the fp16 weights, `tokenizer.model` and one voice `.npy`. Check `SHA256SUMS` after
downloading. The weights are licensed under the Linnet Weights License (`WEIGHTS_LICENSE.md`), not under Apache-2.0.
The weight container is documented in `docs/weights-format.md`.

## Versioning

Code releases are tagged `vX.Y.Z` (one version for the crates, the wheel, the AAR and the Swift package; see
`CHANGELOG.md`); weights releases are tagged `weights-vN`; the weights container format is `linnet-tts/1`.
