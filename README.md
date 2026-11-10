# LinnetTTS

Lightweight on-device text-to-speech with voice cloning, by VoxRT: the model speaks in any voice given to it as a short voice-prompt
latent, streams audio from the first 80 ms step, and runs in real time on a single phone CPU core. Under the hood: an autoregressive
flow LM (1 NFE per frame) over a 25 Hz continuous-latent neural codec with a Vocos-style decoder, ~41M parameters end to end
(31.5M LM + 9.9M decoder), fp16 weights. The release ships two built-in voices; voice prompts for new voices (cloning from a
recording) are produced by VoxRT's encoder, which is not published, and are available under a separate license (see
`WEIGHTS_LICENSE.md`).

**Try it:** [live demo that runs in your browser tab](https://voxrt.com/text-to-speech#live-demo) · [demo Space on Hugging Face](https://huggingface.co/spaces/VoxRT/linnet-tts) · [audio samples, and the same sentence from Piper, Kokoro and pocket-tts](https://huggingface.co/VoxRT/linnet-en-35m)

- `python/` — `linnet_tts_torch`: reference model definition and inference in PyTorch (no training code; see `python/README.md`). The on-device Python package `linnet-tts` is built from the Rust runtime in `bindings/python/`.
- `runtime/` — Rust runtime: `linnet-ops` (only the kernels this model uses) and `linnet-tts` (model, streaming session, C API).
- `bindings/` — platform wrappers: Android AAR, iOS Swift package (xcframework), Linux C library and Python wheel. Browser (WASM) is deferred to a later release.
- `build/` — build and packaging scripts.
- `tests/goldens/` — shared golden vectors; Python and Rust must agree on them.
- `docs/` — model card, weight container format, streaming contract.

Code is licensed under Apache-2.0 (see `LICENSE`). The model weights, tokenizer and voice files are distributed separately
under the Linnet Weights License (see `WEIGHTS_LICENSE.md`), which is not an open-source license. In short:

- **Free, including commercially:** using the unmodified weights with the built-in voices inside your own products, on
  devices and on servers, including paid and closed-source products; speech as part of your product, such as read-aloud,
  assistants and notifications; shipping the unmodified weights with your product; selling audio you generate, subject to
  the restrictions on outputs (no use as a voice or cloning prompt for another system, no model training, and your client
  contracts must say so).
- **Requires a separate written license from VoxRT** (help@voxrt.com): creating or cloning voices (including your own);
  fine-tuning the weights or saving modified, quantised or converted copies of them (converting or quantising in memory at
  load time is allowed); training any model with the weights or their output; offering general-purpose text-to-speech to
  others in any form, free or paid, including an API, SDK, downloadable tool or hosted service; and distributing the weights
  other than as part of your product.

Conditions apply, including attribution and shipping a copy of the license with your product's third-party notices.

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

2. **Rust runtime.** As a dependency: `linnet-tts = { git = "https://github.com/VoxRT/LinnetTTS", tag = "v0.2.0" }` (the crate
   is not on crates.io; see the crate docs with `cargo doc --no-deps` in `runtime/`). The `synth` example takes the weights, the tokenizer, a voice, the text and an output WAV, followed by
   the optional `temp` (0.3), `cfg` (2.0), `seed` (0) and `threads` (0 = automatic):

   ```sh
   cd runtime
   cargo run --release --example synth -- ../local/linnet-en-35m.fp16.safetensors ../local/tokenizer.model \
       ../local/voices/linnet-en-f1.npy "Hello from Linnet." hello.wav
   ```

3. **Python wheel** (the Rust runtime as a Python extension, no PyTorch): `pip install linnet-tts soundfile` (`soundfile` only
   for writing the WAV in the snippet), or build it with
   `cd bindings/python && maturin build --release` and install the wheel from `target/wheels/`.

   ```python
   import linnet_tts, soundfile as sf

   model = linnet_tts.Model("local/linnet-en-35m.fp16.safetensors", "local/tokenizer.model")
   voice = model.load_voice("local/voices/linnet-en-f1.npy")
   audio = model.synthesize("Hello from Linnet.", voice, temp=0.3, cfg=2.0, seed=0)   # np.float32 @ 24 kHz
   sf.write("hello.wav", audio, model.sample_rate)
   ```

   Streaming: `session = model.session(voice, seed=0)`, `session.prime(text)`, then iterate `for chunk in session`
   (1920 samples = 80 ms per step; the first chunk is 720 samples shorter, the last one is 720 zeros). `load_voice` prepares the
   voice for the model, so a prime pass runs only the text rows (one desktop core: 14 ms instead of 67 ms for a short
   sentence); a re-primed session keeps the voice too. See `bindings/python/README.md`.

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

5. **Platform bindings.** The same model, voice and streaming session on every platform, each with its own README,
   install instructions and sample code:

   - Linux C / C++: [`bindings/linux/`](bindings/linux/README.dist.md), tarball `linnet-tts-<version>-linux-{x86_64,aarch64}.tar.gz`
     on the GitHub release (`.so`, `.a`, header, `synth.c` and `stream.c` examples)
   - Android (Kotlin): [`bindings/android/`](bindings/android/README.md), AAR `linnet-tts-<version>.aar` on the GitHub release,
     sample app in `bindings/android/sample/`
   - iOS (Swift; the xcframework also carries a macOS slice for host builds and tests, not a supported product): [`bindings/ios/`](bindings/ios/README.md), `.package(url: "https://github.com/VoxRT/LinnetTTS.git", from: "0.1.0")`,
     SwiftUI sample in `bindings/ios/Sample/`
   - Python: the wheel above ([`bindings/python/`](bindings/python/README.md)); Rust: the `linnet-tts` crate in `runtime/`

6. **Tests.** With the files in `local/` the Rust, wheel, Android and iOS tests find them on their own;
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

Four utterances made with the released weights and the default parameters (`temp` 0.3, `cfg` 2.0, `seed` 0); the
files are in `examples/audio/` together with the command that reproduces them.

| voice | text | audio |
|---|---|---|
| `linnet-en-f1` | Good morning, and welcome back. The meeting starts at nine, and the coffee is already on the table. | [linnet-en-f1_plain.wav](examples/audio/linnet-en-f1_plain.wav) |
| `linnet-en-f1` | There are 24 students in room 12, and the class starts at 9:30. | [linnet-en-f1_numbers.wav](examples/audio/linnet-en-f1_numbers.wav) |
| `linnet-en-m1` | Good morning, and welcome back. The meeting starts at nine, and the coffee is already on the table. | [linnet-en-m1_plain.wav](examples/audio/linnet-en-m1_plain.wav) |
| `linnet-en-m1` | There are 24 students in room 12, and the class starts at 9:30. | [linnet-en-m1_numbers.wav](examples/audio/linnet-en-m1_numbers.wav) |

The model has no text normalisation. Simple digit forms such as the ones above are usually read correctly; dense raw text
(currency amounts with decimals, percentages, slash or ISO dates, phone numbers, version strings) is sometimes misread or
cut short, so convert such text to words before synthesis.

## Performance

Rust runtime, fp16 weights, one step = 80 ms of audio (an LM step plus two decoder frames). Measured without real-time pacing.
On a given platform the audio is bit-identical for every thread count and across runs. Across CPU architectures and C libraries
(x86-64 AVX2, Apple silicon, Android NEON, Raspberry Pi) the samples differ slightly, below 2e-3 absolute (SNR above 60 dB,
inaudible), so the MD5 of the reference synthesis in the weights release (`MANIFEST.json`, `reference_md5`) is reproduced
only by x86-64 builds; use it as a check there.

| device | threads | per step | real-time factor |
|---|---|---|---|
| Apple M5 (macOS) | 1 | 3.3 ms | 4 % |
| iPhone 13 Pro Max, A15 | 1 (auto) / 2 | 5.5 / 4.6 ms | 7 / 6 % |
| iPhone 11, A13 | 1 (auto) | 6.7–7.1 ms | 8–9 % |
| desktop x86-64 (Core i9-13900K), AVX2 | 1 / 4 (auto) | 7.0–7.4 / 2.5 ms | 9 / 3 % |
| Snapdragon 662 (Redmi 9T), Cortex-A73 | 1 / 4 (auto) | 42.4 / 14.3 ms | 53 / 18 % |
| Raspberry Pi Zero 2 W, Cortex-A53 1 GHz | 1 / 4 (auto) | 158 / 59–60 ms | 198 / 75 % |

Real-time factor = mean step time / 80 ms (the sentence-level figures in the binding READMEs also include the prime pass).
First audio = the prime pass plus one step. With the whole prefix in every prime pass (the 0.1.x runtime) it arrived
50–130 ms after the text on the phones and about 250 ms in the Android sample app on the Snapdragon 662. With a prepared
voice (0.2: every binding's `loadVoice` / `load_voice` does it, C `linnet_voice_prepare`) the prime pass runs only the text
rows, and a re-primed session keeps the voice too. Same 44-character sentence: one Core i9 core, prime 14 ms instead of 67 ms,
first audio 22 ms instead of 75 ms; Apple M5, one core, prime 7 ms (first audio about 10 ms, measured prime plus the
3.3 ms step); iPhone 13 Pro Max sample app, one thread, prime 29 ms and first chunk 38 ms instead of 79 / 85 ms; iPhone 11,
first chunk 38–45 ms instead of 107–116 ms; Android
sample app on the Snapdragon 662, four threads, prime 73–77 ms and first chunk 89–92 ms warm (first utterance after
launch: prime 96 ms, first chunk 153 ms) instead of about 250 ms. From there to the first sample at the speaker the
platforms add their own output latency: about 105–110 ms on the iPhones and 65–110 ms on the Snapdragon 662 with the audio
route already open, more when the first chunk has to open it; from a cold process launch with the files in the OS cache,
tap to first sound is about 0.4–0.5 s on the iPhone 11 (the binding READMEs, "Playback timing"). Peak memory is
about 160 MB on Linux. Details, the in-app numbers and the effect of playback pacing: `docs/model-card.md`
("Performance") and `bindings/README.md` ("Threads", "Playback and step time").

### Compared with other CPU TTS models

Same machine (i9-13900K, one performance core per thread), each model through its published Python package, three English
texts, median of three runs. RTF = generation time / audio duration, including text processing and the prime pass. Method,
versions and the reproduction script: `docs/benchmarks.md`, `tools/bench_cpu_tts.py`. Speed only, on English text; quality is not measured here. Linnet is English only; Kokoro serves eight languages from the one model measured, pocket-tts and Piper ship one model per language (their English models are measured).
File sizes are decimal megabytes (10^6 bytes), as in the weights table below.

| model | languages | params | weights | cloning | streaming | RTF, 1 thread | RTF, 4 threads | first audio (short / long text) | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| Linnet `linnet-en-35m` (fp16, Rust; 0.1.1, voice re-read in every prime, see "Performance" for 0.2) | English | 41 M | 83 MB | yes | every 80 ms | 10.6 % | 3.4 % | 87 / 133 ms | 187 MB |
| pocket-tts 4.0.1 (PyTorch) | English (7 languages, one model each) | 100 M | 236 MB | yes | every 80 ms | 23.8 % (2 cores) | 24.1 % | 71 / 82 ms | 968 MB |
| Kokoro-82M (ONNX fp32) | 8 languages in one model | 81 M | 354 MB | no | no | 51.7 % | 17.2 % | 1.5 / 9.6 s | 1246 MB |
| Piper `en_US-ljspeech-medium` (ONNX) | English (one model per voice, many languages) | 16 M | 64 MB | no | per sentence | 5.8 % | 2.1 % | 180 / 283 ms | 503 MB |
| Piper `en_US-ljspeech-high` (ONNX) | English (one model per voice, many languages) | 28 M | 114 MB | no | per sentence | 42.5 % | 12.4 % | 1.4 / 2.1 s | 639 MB |

## Weights

Model weights are not in this repository and are not bundled in any package. Download them from the GitHub release
`weights-v1` of https://github.com/VoxRT/LinnetTTS (mirror: https://huggingface.co/VoxRT/linnet-en-35m):

| file | purpose |
|---|---|
| `linnet-en-35m-weights-v1.tar.gz` | everything below in one archive with the directory layout (`voices/` inside); unpack and run `sha256sum -c SHA256SUMS` |
| `linnet-en-35m.fp16.safetensors` | runtime weights (fp16 matrices, 83 MB = 83 075 620 bytes); what the Rust runtime and every binding load |
| `linnet-en-35m.safetensors` | fp32 reference weights (166 MB) for `linnet_tts_torch` and the golden tests |
| `tokenizer.model` | SentencePiece model |
| `voices/<name>.npy` | built-in voices: float32 `[T, 48]` voice-prompt latents (attached to the release as `linnet-en-f1.npy`, `linnet-en-m1.npy`; put them under `local/voices/`) |
| `SHA256SUMS` | checksums of the files above; verify with `sha256sum -c SHA256SUMS` (macOS: `shasum -a 256 -c`) before use |
| `MANIFEST.json` | contents, sizes and hashes of the files, the metadata of the weight headers, and the x86-64 reference synthesis (text, parameters, MD5) |
| `WEIGHTS_LICENSE.md`, `model-card.md` | license and model card of the weights |

The runtime needs three files: the fp16 weights, `tokenizer.model` and one voice `.npy`. Check `SHA256SUMS` after
downloading. The weights are licensed under the Linnet Weights License (`WEIGHTS_LICENSE.md`), not under Apache-2.0.
Products that ship or use them show the notices in Section 8 of that license. Written permissions given under its
Section 15.2 (at present: storing a prepared voice state inside a product) are collected in `WEIGHTS_PERMISSIONS.md`.
The weight container is documented in `docs/weights-format.md`.

## Versioning

Code releases are tagged `vX.Y.Z` (one version for the crates, the wheel, the AAR and the Swift package; see
`CHANGELOG.md`); weights releases are tagged `weights-vN`; the weights container format is `linnet-tts/1`.
