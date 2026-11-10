# LinnetTTS for Android

Kotlin library `com.voxrt.linnet:linnet-tts` (AAR) over the Rust runtime in `runtime/linnet-tts`, plus a one-screen
sample app. On-device, streaming, 24 kHz, fp16 weights; the API mirrors the Python package one to one.

```kotlin
import com.voxrt.linnet.*

val model = LinnetModel(File(filesDir, "linnet-en-35m.fp16.safetensors"), File(filesDir, "tokenizer.model"))
val voice = model.loadVoice(File(filesDir, "voice.npy"))          // [T, 48] float32 .npy

val audio: FloatArray = model.synthesize("Hello there.", voice, SynthParams(temp = 0.3f, cfg = 2.0f, seed = 0))

model.session(voice, SynthParams(seed = 0)).use { session ->      // streaming, 80 ms per step
    AudioTrackPlayer(model.sampleRate).use { player ->
        session.prime("Streaming gives the first chunk after one step.")
        while (true) {
            val chunk = session.step() ?: break                   // null = finished
            player.write(chunk)
        }
        player.drain()
    }
}
```

| Kotlin | Python |
|---|---|
| `LinnetModel(weightsPath, tokenizerPath)`, `LinnetModel(File, File)`, `LinnetModel.fromBytes(weights, tokenizer)` | `Model(weights, tokenizer)` |
| `.sampleRate`, `.samplesPerStep`, `.latentDim`, `.outputDelay`, `.tokenize(text)` | same |
| `.loadVoice(path)`, `.voiceFromNpy(bytes)` (both prepare the voice), `.prepare(voice)`, `.isPrepared(voice)`, `Voice.fromLatents(FloatArray, frames, dim = 48)`, `Voice.load(path)`, `Voice.fromNpy(bytes)` | `model.load_voice`, `prepare_voice`, `is_prepared`, `Voice.from_array`, `Voice.load` |
| `.synthesize(text, voice, SynthParams(temp, cfg, maxFrames, eosThreshold, seed)): FloatArray` | `model.synthesize(...)` |
| `.session(voice, params): LinnetSession` with `.prime(text)`, `.step(): FloatArray?`, `.stepInto(buf): Int`, `.chunks()`, `.frames`, `.finished`, `.latent()` | `model.session(...)`, `prime`, `step`, iteration, `frames`, `finished`, `latent` |
| `LinnetException(code, message)` with the C API codes | `IOError` / `ValueError` |

Every class is `AutoCloseable`; use `use { }` or call `close()` (idempotent). Chunk layout follows
`docs/streaming-contract.md`: the first chunk is `samplesPerStep - outputDelay` samples (1200), later ones
`samplesPerStep` (1920), the last one `outputDelay` (720) zeros, and the concatenation is sample for sample the
result of `synthesize` with the same parameters and seed.

**Prepared voices.** `loadVoice` and `voiceFromNpy` run the voice through the model once and keep the result in the
`Voice`, so sessions created from it (and `synthesize`) skip the voice part of the prime pass: the first chunk then costs
the text rows plus one step. `Voice.load` / `Voice.fromNpy` / `Voice.fromLatents` do not know the model; call
`model.prepare(voice)` on those. About 60-100 ms once per voice on a phone, 2 MB per voice; the audio is identical. A
re-primed session keeps the voice as well. See `bindings/README.md`, "Prepared voices".

**Threads.** `LinnetModel` and `Voice` are immutable: share them freely across threads and sessions. A
`LinnetSession` is single-threaded (drive it from one thread at a time; different sessions may run in parallel).
Nothing may be closed while another thread is still using it, but a session keeps the weights alive on its
own, so closing the model before its sessions is safe. Loading (a few hundred ms) and synthesis belong off the
main thread. `AudioTrackPlayer.write` blocks only when the 8 s track buffer is full, so a typical utterance is synthesised in
one burst and playback drains behind it (see `bindings/README.md`, "Playback and step time", for why synthesis is not paced).

**Memory.** The fp16 weights file is 83 MB and stays resident (about 85 MB; roughly twice that transiently while
loading); a session adds about 20 MB of KV caches with the default `maxFrames` and guidance (half without guidance).
`LinnetModel.fromBytes` is convenient for `assets/`, but the images pass through the Java heap and are copied
once more natively; for an 83 MB file copy the assets to `filesDir` on first run and use the path constructor.

## Layout

```
bindings/android/
  build-native.sh          cargo-ndk build of jni-crate -> linnet-tts/src/main/jniLibs/<abi>/liblinnet_tts_jni.so
  jni-crate/               Rust: JNI functions (jni 0.21) over runtime/linnet-tts, Arc-shared model
  linnet-tts/              Gradle library module (AAR), Kotlin API, consumer ProGuard rules, instrumented smoke test
  sample/                  Gradle app module: text field, Speak/Stop, streaming AudioTrack playback
  settings.gradle.kts, build.gradle.kts, gradle/libs.versions.toml, gradlew(.bat), gradle/wrapper/
```

`liblinnet_tts_jni.so` links the runtime statically and also exports the plain C API (`linnet_*` from
`runtime/linnet-tts/include/linnet_tts.h`), so NDK code in the same app can call it directly.

## Build

Prerequisites: Rust with the Android targets, cargo-ndk, an Android NDK (r25+; r27d tested), JDK 17+, Android SDK
with platform 35 (the wrapper fetches Gradle 8.9; AGP 8.6.1, Kotlin 2.0.21).

```sh
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/ndk          # or ANDROID_HOME with ndk/<version> inside

bindings/android/build-native.sh              # arm64-v8a + x86_64 (ABIS="arm64-v8a armeabi-v7a x86_64" for 32-bit ARM too)
cd bindings/android
echo "sdk.dir=/path/to/Android/Sdk" > local.properties   # or export ANDROID_HOME
./gradlew :linnet-tts:assembleRelease         # -> linnet-tts/build/outputs/aar/linnet-tts-release.aar
./gradlew :sample:assembleDebug               # -> sample/build/outputs/apk/debug/sample-debug.apk
```

Consumers of the prebuilt AAR need no NDK; AGP then warns `Unable to strip liblinnet_tts_jni.so` (the library is already
stripped; harmless). `build-native.sh` runs `cargo ndk --platform 24 -t <abi> build --release` (fat LTO, stripped) and removes the
runtime's own cdylib that cargo-ndk copies alongside; the Gradle build fails with a pointer to the script when
the `.so` files are missing. Native library sizes (0.1.0): arm64-v8a 756 KB, x86_64 805 KB. Nothing but
`libc`, `libm` and `libdl` is required at run time; minSdk 24.

The runtime's release profile (fat LTO, one codegen unit) applies to these builds too. No `target-cpu` is set:
the kernels pick NEON at compile time on every AArch64 target and need no further flags. Scheduling for a
specific core is possible with, for example, `RUSTFLAGS="-C target-cpu=cortex-a73" bindings/android/build-native.sh`,
but it is optional: on a Snapdragon 662 (Cortex-A73 class cores) it changed neither the output bytes nor the
speed in our measurements (43.7 vs 42.2 ms per step without the flag, within the run-to-run spread), so the
default stays portable.

Without cargo-ndk the NDK's clang can be used directly:

```sh
export CC_aarch64_linux_android=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$CC_aarch64_linux_android
cargo build --release --target aarch64-linux-android --manifest-path jni-crate/Cargo.toml
cp jni-crate/target/aarch64-linux-android/release/liblinnet_tts_jni.so linnet-tts/src/main/jniLibs/arm64-v8a/
```

### Using the AAR

The AAR `linnet-tts-<version>.aar` is attached to each GitHub release (no Maven repository yet). Drop it into the
app, or build it here and publish it to the local Maven repository:

```sh
./gradlew :linnet-tts:publishToMavenLocal     # com.voxrt.linnet:linnet-tts:0.1.1 in ~/.m2
```

```kotlin
// settings.gradle.kts: repositories { mavenLocal(); google(); mavenCentral() }
dependencies { implementation("com.voxrt.linnet:linnet-tts:0.1.1") }
// or: implementation(files("libs/linnet-tts-0.1.1.aar"))   // the release asset (a local build is linnet-tts-release.aar)
```

The AAR carries `consumer-rules.pro`, so R8 in the app keeps `LinnetNative` (whose `external fun`s are instance
methods of a Kotlin object, resolved by name from the `.so`) and `LinnetException` (constructed from native
code). No extra rules are needed in the app; the sample's release build runs R8 against them.

Maven Central / JitPack publishing is planned for a later release (`maven-publish` is configured with the POM; a
signing step and the repository are missing).

## Sample app

`sample/` is one Activity: a text field, Speak (streams `LinnetSession.step()` chunks into an
`AudioTrackPlayer` from a background thread and prints prime time, time to first chunk, worst step and the
synthesis RTF), Stop. It reads the three model files from the app's private `filesDir`; weights are not bundled.

The commands below read the model files from `../../local/` (`local/` = the unpacked weights release, see the
root README):

```sh
./gradlew :sample:installDebug
adb push ../../local/linnet-en-35m.fp16.safetensors ../../local/tokenizer.model /data/local/tmp/
adb push ../../local/voices/linnet-en-f1.npy /data/local/tmp/voice.npy
adb shell run-as com.voxrt.linnet.sample sh -c 'mkdir -p files && cp /data/local/tmp/linnet-en-35m.fp16.safetensors /data/local/tmp/tokenizer.model /data/local/tmp/voice.npy files/'
```

(`run-as` works for the debug build and needs the device unlocked and the app launched once after installation; before that
`run-as` reports `couldn't stat /data/user/0/<package>`. Model files are distributed separately, see `WEIGHTS_LICENSE.md`.)

Sample app on a Snapdragon 662 (Redmi 9T), 7.0 s sentence, `threads` auto (4 big cores), playback on, warm (the first utterance of a
process is slower: 17-19 ms per step with a 55-70 ms worst step and 300-430 ms to the first chunk): 15.9-18.4 ms per step (RTF 20-23 %,
steps over audio duration as the app computes it; the root README tables use step / 80 ms),
worst step 33-39 ms, first chunk 244-260 ms after the Speak tap; the big-core affinity of the pool (see `bindings/README.md`, "Threads") is
what brings the app close to the 14.3 ms of the command-line build.

## Tests

- Host: `cargo test --release` in `jni-crate/` (handle helpers, parameter validation, error codes, and, when
  `../../local` or `LINNET_ASSETS` holds the model files, a check that a session outlives its model handle and
  streams bit-exactly against the offline result).
- Device/emulator: `linnet-tts/src/androidTest` loads the library, checks the constants, the voice round trip,
  the stream-equals-offline identity and typed errors. It reads the model files from `/data/local/tmp/linnet/`
  and skips when they are absent:

```sh
adb shell mkdir -p /data/local/tmp/linnet
adb push ../../local/linnet-en-35m.fp16.safetensors ../../local/tokenizer.model /data/local/tmp/linnet/
adb push ../../local/voices/linnet-en-f1.npy /data/local/tmp/linnet/voice.npy
./gradlew :linnet-tts:connectedDebugAndroidTest
```
