# bindings/

Platform wrappers around the Rust runtime in `runtime/linnet-tts`. Every channel exposes the same three
objects (model, voice, streaming session) and the same contract (`docs/streaming-contract.md`): one LM step
= 80 ms = 1920 samples at 24 kHz, first chunk 720 samples shorter, last chunk 720 zeros, the chunks
concatenate bit-for-bit to the offline result.

| channel | directory | artefact | status |
|---|---|---|---|
| Linux C / C++ | `linux/` | `linnet-tts-<ver>-linux-{x86_64,aarch64}.tar.gz` (`.so`, `.a`, header) | ready |
| Python | `python/` | wheel `linnet-tts` (import `linnet_tts`), abi3 cp39+ | ready |
| Android | `android/` | AAR (JNI over the C API) | TODO |
| iOS | `ios/` | xcframework + Swift package | TODO |
| Browser | `browser/` | WASM (SIMD128) + npm package | TODO |

Model weights (`linnet-en-35m.fp16.safetensors`, `tokenizer.model`) and voice files are not part of any
channel; see `WEIGHTS_LICENSE.md`.

## Linux C

```sh
bindings/linux/build.sh                                        # x86_64, host toolchain
bindings/linux/build.sh --target aarch64-unknown-linux-gnu     # cross build
```

Output: `bindings/linux/dist/linnet-tts-<version>-linux-<arch>/{lib,include,examples,README.md,LICENSE}` plus a
`.tar.gz` and its SHA-256. The script builds `runtime/linnet-tts` in release mode with fat LTO, strips the
libraries, and reports the glibc symbol versions the `.so` needs. The aarch64 cross build uses
`cargo zigbuild --target aarch64-unknown-linux-gnu.2.17` when cargo-zigbuild and zig are present (glibc 2.17
floor), otherwise a GNU cross compiler (`aarch64-linux-gnu-gcc`) as the linker, otherwise it prints what to
install and exits. The x86_64 kernels pick AVX2+FMA at run time; no `-C target-cpu` is used.

Examples (`bindings/linux/examples/`): `synth.c` writes a sentence to a 16-bit WAV with `linnet_synthesize`;
`stream.c` drives `linnet_session_*` chunk by chunk and reports time to first audio and the worst step.

```sh
cd bindings/linux/examples
make DIST=../dist/linnet-tts-0.0.1-linux-x86_64            # or STATIC=1
./synth  weights.safetensors tokenizer.model voice.npy "Hello there." hello.wav
./stream weights.safetensors tokenizer.model voice.npy "Hello there." hello_stream.wav
```

Linking by hand: `-Iinclude -Llib -llinnet_tts -lm` (shared) or `lib/liblinnet_tts.a -lpthread -ldl -lm`
(static).

## Python

`bindings/python/` is a pyo3 extension (`linnet_tts._native`, abi3 for CPython >= 3.9) plus a thin
NumPy wrapper (`linnet_tts/__init__.py`). Built with maturin:

```sh
cd bindings/python
maturin build --release                      # -> target/wheels/linnet_tts-<ver>-cp39-abi3-<platform>.whl
pip install target/wheels/linnet_tts-*.whl
```

```python
import linnet_tts
model = linnet_tts.Model("linnet-en-35m.fp16.safetensors", "tokenizer.model")
voice = model.load_voice("voice.npy")                 # or linnet_tts.Voice.from_array(latents)
audio = model.synthesize("Hello there.", voice, temp=0.3, cfg=2.0, seed=0)   # np.float32 @ 24 kHz

session = model.session(voice, seed=0)
session.prime("Hello there.")
for chunk in session:                                 # np.float32 chunks, 80 ms each
    ...
```

`Model.sample_rate`, `Model.samples_per_step`, `Model.output_delay`, `Model.tokenize`, `Model.decode`,
`Session.step() -> Optional[np.ndarray]`, `Session.latent()`, `Session.frames`, `Session.finished`.
Long calls release the GIL. Tests: `pytest bindings/python/tests` (needs the model assets, see
`bindings/python/README.md`). For a manylinux wheel build on a modern host use
`maturin build --release --manylinux 2_17 --zig` (needs zig) or build inside a manylinux container.

The PyTorch reference package lives in `python/` as `linnet_tts_torch` (pip `linnet-tts-torch`); the two do
not collide.

## Android, iOS, browser (TODO)

- **Android**: `cargo ndk` build of `linnet-tts` for arm64-v8a / armeabi-v7a / x86_64, a JNI layer over
  `include/linnet_tts.h`, Kotlin `Model` / `Voice` / `Session` mirroring the Python API, packaged as an AAR.
- **iOS**: `aarch64-apple-ios` + simulator static libraries into an xcframework, a Swift package wrapping the
  C API with the same three types.
- **Browser**: `wasm32-unknown-unknown` with `+simd128`, a JS wrapper that feeds an AudioWorklet 80 ms at a
  time; model files fetched by the page.
