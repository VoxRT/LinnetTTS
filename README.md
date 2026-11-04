# LinnetTTS

Lightweight on-device text-to-speech by VoxRT: an autoregressive flow LM (1 NFE per frame) over a 25 Hz continuous-latent neural codec,
with a Vocos-style decoder. ~41M parameters end to end (31.5M LM + 9.9M decoder), streaming synthesis, built-in voices, fp16 weights.

- `python/` — `linnet_tts_torch`: reference model definition and inference in PyTorch (no training code; see `python/README.md`). The on-device Python package `linnet-tts` is built from the Rust runtime in `bindings/python/`.
- `runtime/` — Rust runtime: `linnet-ops` (only the kernels this model uses) and `linnet-tts` (model, streaming session, C API).
- `bindings/` — platform wrappers: Android AAR, iOS Swift package (xcframework), Linux C library and Python wheel. Browser (WASM) is deferred to a later release.
- `build/` — build and packaging scripts.
- `tests/goldens/` — shared golden vectors; Python and Rust must agree on them.
- `docs/` — model card, weight container format, streaming contract.

Code is licensed under Apache-2.0 (see `LICENSE`). The model weights, tokenizer and voice files are distributed separately
under the Linnet Weights License (see `WEIGHTS_LICENSE.md`), which is not an open-source license. In short:

- **Free, including commercially:** running the unmodified weights with the built-in voices inside your own products, on
  devices and on servers, including paid and closed-source products; shipping the unmodified weights as part of your
  product; selling audio you generate.
- **Requires a separate written license from VoxRT** (help@voxrt.com): creating or cloning voices (including your own);
  fine-tuning the weights or saving modified, quantised or converted copies of them (converting or quantising in memory at
  load time is allowed); training any model with the weights or their output; offering general-purpose text-to-speech to
  others as an API or service, free or paid; and publishing the weights outside your product.

The license text governs; this summary does not. The test fixtures in `tests/goldens/` include copies of the two built-in
voices; `tests/goldens/NOTICE.md` states their license and the permission to redistribute them with the repository.

The model is called `linnet-en-35m`; the number is a label from the original parameter budget, not an exact count (41.4 M in total:
flow LM 31.5 M including the 3.1 M text table, decoder 9.9 M; 38.4 M outside the 6001×512 text table). The fp16 weights file is 83 MB.

## Quick start

1. **Weights.** Download the archive of the `weights-v1` release (see "Weights" below) and unpack it into `local/` at the
   repository root. The archive has one top-level directory, so strip it:

   ```sh
   mkdir -p local && tar -xzf linnet-en-35m-weights-v1.tar.gz -C local --strip-components=1
   (cd local && sha256sum -c SHA256SUMS)
   ```

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
   (1920 samples = 80 ms per step; the first chunk is 720 samples shorter, the last one is 720 zeros). See `bindings/python/README.md`.

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

5. **Tests.** With the files in `local/` the Rust, wheel, Android and iOS tests find them on their own;
   `tests/test_ref_goldens.py` needs `LINNET_WEIGHTS` and `LINNET_TOKENIZER` set explicitly (it is skipped otherwise). The
   environment variables redirect the tests to one file each or to a whole directory:

   - `LINNET_WEIGHTS` — fp32 weights for the Rust golden tests (default `local/linnet-en-35m.safetensors`) and, with
     `LINNET_TOKENIZER`, for `tests/test_ref_goldens.py` (no default)
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

## Examples

`examples/audio/` holds four utterances made with the default parameters (two voices, a plain sentence and one with simple
numbers), with the texts and the command that reproduces them. The model has no text normalisation: simple digit forms
are usually read correctly, but dense raw text (currency amounts with decimals, percentages, slash or ISO dates, phone
numbers, version strings) is sometimes misread or cut short, so convert such text to words before synthesis.

## Performance

Rust runtime, fp16 weights, one step = 80 ms of audio (an LM step plus two decoder frames). Measured without real-time pacing.
The audio is bit-identical for every thread count; the reference synthesis of the weights release (`MANIFEST.json`,
`reference_md5`) gave the same MD5 on every device listed.

| device | threads | per step | real-time factor |
|---|---|---|---|
| Apple M5 (macOS) | 1 | 3.3 ms | 4 % |
| iPhone 13 Pro Max, A15 | 1 (auto) / 2 | 5.5 / 4.6 ms | 7 / 6 % |
| iPhone 11, A13 | 1 (auto) | 6.7–7.1 ms | 8–9 % |
| desktop x86-64 (Core i9-13900K), AVX2 | 1 / 4 (auto) | 7.0–7.4 / 2.5 ms | 9 / 3 % |
| Snapdragon 662 (Redmi 9T), Cortex-A73 | 1 / 4 (auto) | 42.4 / 14.3 ms | 53 / 18 % |
| Raspberry Pi Zero 2 W, Cortex-A53 1 GHz | 1 / 4 (auto) | 158 / 59–60 ms | 198 / 75 % |

Real-time factor = mean step time / 80 ms (the sentence-level figures in the binding READMEs also include the prime pass).
First audio arrives 50–120 ms after the text on the phones and about 250 ms in the Android sample app on the Snapdragon 662
(the prime pass plus one step). Peak memory is about 160 MB on Linux. Details, the in-app numbers and the effect of playback pacing: `docs/model-card.md`
("Performance") and `bindings/README.md` ("Threads", "Playback and step time").

## Weights

Model weights are not in this repository and are not bundled in any package. Download them from the GitHub release
`weights-v1` of https://github.com/VoxRT/LinnetTTS (mirror: https://huggingface.co/VoxRT/linnet-en-35m):

| file | purpose |
|---|---|
| `linnet-en-35m-weights-v1.tar.gz` | everything below in one archive with the directory layout (`voices/` inside); unpack and run `sha256sum -c SHA256SUMS` |
| `linnet-en-35m.fp16.safetensors` | runtime weights (fp16 matrices, ~83 MB); what the Rust runtime and every binding load |
| `linnet-en-35m.safetensors` | fp32 reference weights (~166 MB) for `linnet_tts_torch` and the golden tests |
| `tokenizer.model` | SentencePiece model |
| `voices/<name>.npy` | built-in voices: float32 `[T, 48]` voice-prompt latents (attached to the release as `linnet-en-f1.npy`, `linnet-en-m1.npy`; put them under `local/voices/`) |
| `SHA256SUMS` | checksums of the files above; verify with `sha256sum -c SHA256SUMS` before use |
| `WEIGHTS_LICENSE.md`, `model-card.md` | license and model card of the weights |

The runtime needs three files: the fp16 weights, `tokenizer.model` and one voice `.npy`. Check `SHA256SUMS` after
downloading. The weights are licensed under the Linnet Weights License (`WEIGHTS_LICENSE.md`), not under Apache-2.0.
Products that ship or use them show the notices in Section 8 of that license.
The weight container is documented in `docs/weights-format.md`.

## Versioning

Code releases are tagged `vX.Y.Z` (one version for the crates, the wheel, the AAR and the Swift package; see
`CHANGELOG.md`); weights releases are tagged `weights-vN`; the weights container format is `linnet-tts/1`.
