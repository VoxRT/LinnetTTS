# bindings/

Platform wrappers around the Rust runtime in `runtime/linnet-tts`. Every channel exposes the same three
objects (model, voice, streaming session) and the same contract (`docs/streaming-contract.md`): one LM step
= 80 ms = 1920 samples at 24 kHz, first chunk 720 samples shorter, last chunk 720 zeros, the chunks
concatenate bit-for-bit to the offline result.

| channel | directory | artefact | status |
|---|---|---|---|
| Linux C / C++ | `linux/` | `linnet-tts-<ver>-linux-{x86_64,aarch64}.tar.gz` (`.so`, `.a`, header) | ready |
| Python | `python/` | wheel `linnet-tts` (import `linnet_tts`), abi3 cp39+ | ready |
| Android | `android/` | AAR `com.voxrt.linnet:linnet-tts` (Kotlin API, JNI in Rust), arm64-v8a + x86_64 | ready (publishing TODO) |
| iOS | `ios/` | `LinnetTTS.xcframework` (static, ios-arm64 + simulator + macos-arm64) + Swift package `LinnetTTS` | ready (publishing TODO) |
| Browser | `browser/` | WASM (SIMD128) + npm package | not planned for this release |

Model weights (`linnet-en-35m.fp16.safetensors`, `tokenizer.model`) and voice files are not part of any
channel; see `WEIGHTS_LICENSE.md`.

## Threads

Every channel exposes a `threads` setting: `1` runs single-threaded, `0` (the default) picks the fast cores of the
device on Android/Linux (the big cluster by `cpuinfo_max_freq`, at most 4) and **one thread on Apple platforms**: an
Apple-silicon step is already 5-7 ms and splits into ~80 tiny regions, so waking a worker on a second performance core
costs more than the work (A13: 2 threads 11.4 ms vs 6.7 ms per step, 4 threads 47 ms); the pool only shortens the
prime pass there, and an app that wants a faster first chunk can set `threads` to the performance-core count. The
model owns one persistent worker pool; the parameter of a session (`SynthParams.threads`, Python/Kotlin keyword
`threads`) is applied to the model when the session is created, and the C API sets it on the model directly
(`linnet_set_threads(model, n)`, `linnet_threads(model)`). Work is split across output rows, heads and columns
with every reduction kept on one thread, so **the audio is bit-identical for every thread count**; only the
speed changes (see `docs/streaming-contract.md`).

## Playback and step time

Synthesis is far faster than real time, so the player helpers (`LinnetPlayer` on iOS, `AudioTrackPlayer` on Android)
queue 8 s of audio ahead by default: an utterance is produced in one burst at full clock and playback drains behind
it. Pacing synthesis to real time with a small queue makes every step 3-4 times slower (the core is clocked down and
the work moved to efficiency cores while the thread sleeps between steps) without improving latency to the first
chunk. Measure throughput without pacing; the sample apps' `-nopace`/`-noaudio` flags exist for that.

## Linux C

```sh
bindings/linux/build.sh                                        # x86_64, host toolchain
bindings/linux/build.sh --target aarch64-unknown-linux-gnu     # cross build
bindings/linux/build.sh --zig                                  # x86_64 with cargo-zigbuild, glibc 2.17 floor
```

Output: `bindings/linux/dist/linnet-tts-<version>-linux-<arch>/{lib,include,examples,README.md,LICENSE}` plus a
`.tar.gz` and its SHA-256. The script builds `runtime/linnet-tts` in release mode with fat LTO, strips the
libraries, and reports the glibc symbol versions the `.so` needs. The aarch64 cross build uses
`cargo zigbuild --target aarch64-unknown-linux-gnu.2.17` when cargo-zigbuild and zig are present (glibc 2.17
floor), otherwise a GNU cross compiler (`aarch64-linux-gnu-gcc`) as the linker, otherwise it prints what to
install and exits. The native x86_64 build requires the glibc of the build host (2.34 on Ubuntu 22.04); `--zig`
builds it with cargo-zigbuild against glibc 2.17 as well, which is what the release workflow does. The x86_64
kernels pick AVX2+FMA at run time; no `-C target-cpu` is used.

Examples (`bindings/linux/examples/`): `synth.c` writes a sentence to a 16-bit WAV with `linnet_synthesize`;
`stream.c` drives `linnet_session_*` chunk by chunk and reports time to first audio and the worst step.

```sh
cd bindings/linux/examples
make DIST=../dist/linnet-tts-<version>-linux-x86_64        # or STATIC=1
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

## Android

`bindings/android/` is a Gradle project: the library module `linnet-tts` (AAR, Kotlin API `LinnetModel` /
`Voice` / `LinnetSession` / `SynthParams` / `AudioTrackPlayer`, package `com.voxrt.linnet`) and a `sample` app.
The JNI layer is Rust (`bindings/android/jni-crate`, `jni` 0.21) over `runtime/linnet-tts`, cross-built with
cargo-ndk into `liblinnet_tts_jni.so` for arm64-v8a and x86_64 (minSdk 24, ~0.7 MB per ABI).

```sh
bindings/android/build-native.sh                 # needs ANDROID_NDK_HOME, cargo-ndk, Android Rust targets
cd bindings/android && ./gradlew :linnet-tts:assembleRelease :sample:assembleDebug
```

```kotlin
val model = LinnetModel(weightsFile, tokenizerFile)
val voice = model.loadVoice(voiceFile)
val audio = model.synthesize("Hello there.", voice, SynthParams(seed = 0))   // FloatArray @ 24 kHz
model.session(voice).use { s -> s.prime("Hello there."); while (true) { val c = s.step() ?: break; player.write(c) } }
```

Model and voice are shareable across threads; a session is single-threaded. See `bindings/android/README.md`
for the AAR consumption, ProGuard notes, the sample app and the device tests.

## iOS

`bindings/ios/` is a Swift package `LinnetTTS` (`LinnetModel` / `Voice` / `LinnetSession` / `SynthParams` /
`LinnetError` / `LinnetPlayer`) over the C API, shipped as a static `LinnetTTS.xcframework` built by
`bindings/ios/build-xcframework.sh` (slices: ios-arm64, ios simulator arm64 + x86_64, macos-arm64 for host
tests), plus a SwiftUI sample app (`bindings/ios/Sample`) and XCTests that run on the macOS host, the simulator
and, hosted by the sample app, on a device.

```sh
bindings/ios/build-xcframework.sh                # needs Xcode and the Apple Rust targets
cd bindings/ios/LinnetTTS && LINNET_ASSETS=../../../local swift test -c release
```

```swift
let model = try LinnetModel(weightsPath: weights, tokenizerPath: tokenizer)
let voice = try model.loadVoice(path: voicePath)
let audio = try model.synthesize(text: "Hello there.", voice: voice, params: SynthParams(seed: 0))  // [Float] @ 24 kHz
let s = try model.session(voice: voice); try s.prime(text: "Hello there.")
while let c = try s.step() { try player.write(c) }
```

Model and voice are shareable across threads; a session is single-threaded. See `bindings/ios/README.md` for
SPM integration, signing notes, the sample app and the tests.

## Browser (deferred)

Deferred to a later release. The runtime keeps compiling for `wasm32-unknown-unknown` (single-threaded) so the
option stays open, but no JS wrapper or npm package is part of this release.
