# bindings/

Platform wrappers around the Rust runtime in `runtime/linnet-tts`. Every channel exposes the same three
objects (model, voice, streaming session) and the same contract (`docs/streaming-contract.md`): one LM step
= 80 ms = 1920 samples at 24 kHz, first chunk 720 samples shorter, last chunk 720 zeros, the chunks
concatenate bit-for-bit to the offline result.

| channel | directory | artefact | status |
|---|---|---|---|
| Linux C / C++ | `linux/` | `linnet-tts-<ver>-linux-{x86_64,aarch64}.tar.gz` (`.so`, `.a`, header) | ready |
| Python | `python/` | wheel `linnet-tts` (import `linnet_tts`), abi3 cp39+ | ready |
| Android | `android/` | AAR `com.voxrt.linnet:linnet-tts` (Kotlin API, JNI in Rust), arm64-v8a + x86_64 | ready; `linnet-tts-<ver>.aar` attached to the GitHub release (no Maven repository yet) |
| iOS | `ios/` | `LinnetTTS.xcframework` (static, ios-arm64 + simulator + macos-arm64) + Swift package `LinnetTTS` | ready; `.package(url: "https://github.com/VoxRT/LinnetTTS.git", from: "<ver>")` |
| Browser | — | WASM (SIMD128) + npm package | not planned for this release |

Model weights (`linnet-en-35m.fp16.safetensors`, `tokenizer.model`) and voice files are not part of any
channel; see `WEIGHTS_LICENSE.md`.

## Threads

Every channel exposes a `threads` setting: `1` runs single-threaded, `0` (the default) picks the fast cores of the
device on Android/Linux (the big cluster by `cpuinfo_max_freq`, at most 4) and **one thread on Apple platforms** (on an
M-series Mac a second thread does pay off, about 1.7×; set `threads` explicitly there): an
Apple-silicon step is already 5-7 ms and splits into ~80 tiny regions, so waking a worker on a second performance core
costs more than the work (A13: 2 threads 11.4 ms vs 6.7 ms per step, 4 threads 47 ms); the pool only shortens the
prime pass there, and an app that wants a faster first chunk can set `threads` to the performance-core count. The
model owns one persistent worker pool; the parameter of a session (`SynthParams.threads`, Python/Kotlin keyword
`threads`) is applied to the model when the session is created, and the C API sets it on the model directly
(`linnet_set_threads(model, n)`, `linnet_threads(model)`). Work is split across output rows, heads and columns
with every reduction kept on one thread, so **the audio is bit-identical for every thread count**; only the
speed changes (see `docs/streaming-contract.md`).

**Prepared voices.** The voice part of the prefix (`[bos_voice | voice]`, 77 positions for the built-in voices) does not
depend on the text, so its trunk state is computed once per voice and model and reused: `loadVoice` / `load_voice` /
`voiceFromNpy` prepare the voice they return (Rust `Linnet::voice_state`, C `linnet_voice_prepare`, Python
`Model.prepare_voice`, Kotlin `LinnetModel.prepare`, Swift `LinnetModel.prepare(voice:)` for voices built from latents or
loaded without a model), and every session created from a prepared voice, as well as every re-prime of an existing
session, runs only the text rows of the prime pass. The audio is bit-identical either way. Preparation costs about one
prime pass over the voice (66 ms on one Core i9 core, 39 ms on an Apple M5 core, 230–340 ms in the Android sample app on
a Snapdragon 662 right after the model load, four threads) and keeps about 2 MB per voice in memory; it is tied to the
model instance that computed it (another model uses the voice unprepared, and preparing the voice for another model
replaces the state). The runtime never writes the state to disk and has no export call; a product that adds one may
store its own copy under the written permission in `WEIGHTS_PERMISSIONS.md`. On one i9 core a 44-character sentence then primes in 14 ms instead of 67 ms (99 characters:
21 instead of 79; 321 characters: 77 instead of 139), and the first chunk follows about 8 ms later.

On Android the runtime also pins its threads to the big cluster: the pool workers for good, and the thread that calls
`prime`, `step` or `decode` only for the duration of that call (its previous affinity mask is restored afterwards).
Without this the scheduler keeps the short bursts of a synthesis on the little cores and migrates them up slowly; the
same sample app on a Snapdragon 662 measured 22-33 ms per step before and 16-18 ms after (14.3 ms for the pinned
command-line build). Nothing is pinned on a device whose cores are all alike, and nothing is pinned on iOS or Linux.

## Playback and step time

Synthesis is far faster than real time, so the player helpers (`LinnetPlayer` on iOS, `AudioTrackPlayer` on Android)
queue 8 s of audio ahead by default: an utterance is produced in one burst at full clock and playback drains behind
it. Pacing synthesis to real time with a small queue makes every step 3-4 times slower (the core is clocked down and
the work moved to efficiency cores while the thread sleeps between steps) without improving latency to the first
chunk. Measure throughput without pacing; the iOS sample app's `-nopace`/`-noaudio` flags exist for that.

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
`stream.c` prepares the voice (`linnet_voice_prepare`, once per voice), drives `linnet_session_*` chunk by chunk and reports
the preparation time, time to first audio and the worst step.

```sh
cd bindings/linux/examples
make DIST=../dist/linnet-tts-<version>-linux-x86_64        # or STATIC=1
./synth  weights.safetensors tokenizer.model voice.npy "Hello there." hello.wav
./stream weights.safetensors tokenizer.model voice.npy "Hello there." hello_stream.wav
```

Linking by hand: `-Iinclude -Llib -llinnet_tts -lm -Wl,-rpath,'$ORIGIN/lib'` (shared; without the rpath the binary
only starts with `LD_LIBRARY_PATH=lib`) or `lib/liblinnet_tts.a -lpthread -ldl -lm` (static).

### Raspberry Pi

The aarch64 tarball runs unchanged on a Raspberry Pi (64-bit OS; the gcc build needs glibc 2.34 or newer, the `--zig`
build glibc 2.17). Measured on a Pi Zero 2 W (4 × Cortex-A53 at 1 GHz, 512 MB, Debian 13, no heatsink) with the fp16
weights and a 7.0 s sentence (RTF here = prime pass plus all steps over the audio duration, as the `synth` example prints it; the
tables in the root README use step / 80 ms): 1 thread 158 ms per step (RTF 242 %); `threads` auto picks all four cores and gives
59–60 ms per step (RTF 88 %, worst step 63–76 ms), i.e. real time with a small margin; six utterances back to back
stayed at 59–60 ms while the SoC warmed from 57 to 67 °C. Peak resident memory 160 MB; loading the model from the SD
card takes about 2.5 s warm (5 s cold from the SD card). The audio differs from the x86-64 build by up to 3e-4 absolute (inaudible;
the released aarch64 library is built against glibc 2.17 whose math functions differ), so the reference MD5 of the weights release
is not reproduced on the Pi. A Pi Zero 2 W is the floor for
real time; a Pi 4 or Pi 5 has several times the margin.

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
for chunk in session:                                 # np.float32 chunks: 1200, then 1920 (80 ms) each, 720 zeros last
    ...
```

`Model.sample_rate`, `Model.samples_per_step`, `Model.output_delay`, `Model.tokenize`, `Model.decode`,
`Session.step() -> Optional[np.ndarray]`, `Session.latent()`, `Session.frames`, `Session.finished`.
Long calls release the GIL. Tests: `pytest bindings/python/tests` (needs the model assets under `../../local/`
or `LINNET_ASSETS`; `local/` = the unpacked weights release, see the root README and `bindings/python/README.md`). For a manylinux wheel build on a modern host use
`maturin build --release --manylinux 2_17 --zig` (needs zig) or build inside a manylinux container.

The PyTorch reference package lives in `python/` as `linnet_tts_torch` (`pip install ./python`; not on PyPI); the two do
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
