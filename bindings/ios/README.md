# LinnetTTS for iOS (and macOS)

Swift package `LinnetTTS` over the C API of the Rust runtime in `runtime/linnet-tts`, shipped as a static
`LinnetTTS.xcframework`, plus a one-screen SwiftUI sample. On-device, streaming, 24 kHz, fp16 weights; the API
mirrors the Kotlin and Python packages one to one.

```swift
import LinnetTTS

let model = try LinnetModel(weightsPath: weights, tokenizerPath: tokenizer)   // a few hundred ms, off the main thread
let voice = try model.loadVoice(path: voicePath)                               // [T, 48] float32 .npy

let audio: [Float] = try model.synthesize(text: "Hello there.", voice: voice,
                                          params: SynthParams(temp: 0.3, cfg: 2.0, seed: 0))

let session = try model.session(voice: voice, params: SynthParams(seed: 0))   // streaming, 80 ms per step
let player = try LinnetPlayer(sampleRate: model.sampleRate)                    // AVAudioEngine + AVAudioPlayerNode
try session.prime(text: "Streaming gives the first chunk after one step.")
while let chunk = try session.step() { try player.write(chunk) }               // nil = finished
player.drain()
```

| Swift | Kotlin | Python |
|---|---|---|
| `LinnetModel(weightsPath:tokenizerPath:)`, `LinnetModel(weightsURL:tokenizerURL:)` | `LinnetModel(weightsPath, tokenizerPath)` | `Model(weights, tokenizer)` |
| `.sampleRate`, `.samplesPerStep`, `.outputDelay`, `.latentDim`, `LinnetModel.version` | same | same |
| `.threads` (get/set; 0 = automatic, getter returns the resolved count) | `SynthParams.threads` | `threads=` |
| `.loadVoice(path:)`, `.loadVoice(url:)`, `Voice.load(path:)`, `Voice(latents:frames:dim:)`, `.frames`, `.dim` | `loadVoice`, `Voice.load`, `Voice.fromLatents` | `load_voice`, `Voice.from_array` |
| `.synthesize(text:voice:params:) -> [Float]` | `synthesize(text, voice, params)` | `synthesize(...)` |
| `.session(voice:params:) -> LinnetSession`; `.prime(text:)`, `.step() -> [Float]?`, `.step(into:) -> Int`, `.remaining()`, `.finished`, `.chunksEmitted` | `session`, `prime`, `step`, `stepInto`, `chunks`, `finished` | `session`, `prime`, `step`, iteration, `finished` |
| `SynthParams(temp:cfg:maxFrames:eosThreshold:seed:threads:)` | `SynthParams(...)` | keywords |
| `LinnetError` (`.invalidArgument`, `.io`, `.format`, `.shape`, `.bufferTooSmall`, `.internalError`; `.code`, `.message`) | `LinnetException(code, message)` | `IOError` / `ValueError` |
| `LinnetPlayer(sampleRate:bufferMillis:)`: `.write(_:)`, `.drain()`, `.stop()` | `AudioTrackPlayer` | - |

Not in the C API, hence not here: `tokenize`, `latent()`, `frames`. `outputDelay` (720) and `latentDim` (48) are
constants of the streaming contract.

Chunk layout follows `docs/streaming-contract.md`: the first chunk is `samplesPerStep - outputDelay` samples
(1200), later ones `samplesPerStep` (1920), the last one `outputDelay` (720) zeros, and the concatenation is
sample for sample the result of `synthesize` with the same parameters and seed (the tests assert it).

**Lifetimes.** `LinnetModel`, `Voice` and `LinnetSession` are classes that free their C handle in `deinit`. A
session keeps strong references to its model and voice, so the C API's "free sessions before the model" rule
is enforced by ARC; drop references in any order.

**Threads.** `LinnetModel` and `Voice` are immutable (apart from `threads`) and may be shared across threads and
sessions. A `LinnetSession` is single-threaded: drive it from one queue at a time; different sessions of the
same model may run concurrently. `threads` selects the worker count of the model's persistent pool (1 =
single-threaded, 0 = automatic, which is one thread on iOS and macOS, where a second core does not pay off); `SynthParams.threads` applies it when a
session is created. The audio is bit-identical for every thread count. Change it between utterances, not while
a session steps. `LinnetPlayer.write` blocks while more than `bufferMillis` (default 8000 ms) of audio is queued,
which paces synthesis to real time when the queue is small; `stop()` may be called from any thread.

**Memory.** The fp16 weights file is 83 MB and stays resident (about 85 MB; roughly twice that transiently while loading);
a session adds about 20 MB of KV caches with the default `maxFrames` and guidance (half without guidance).
The static library adds about 2 MB to an arm64 app binary.

## Layout

```
bindings/ios/
  build-xcframework.sh     cargo build (ios, ios-sim arm64+x86_64, macos arm64) -> dist/LinnetTTS.xcframework (+ zip, checksum)
  device-smoke.sh          hosted tests + sample app on an attached iPhone (timings from the log)
  LinnetTTS/               Swift package: Package.swift, Sources/LinnetTTS (API + LinnetPlayer), Tests/LinnetTTSTests
  Sample/                  LinnetSample.xcodeproj: SwiftUI app (text, Speak, Stop, timings) + LinnetSampleTests host target
  dist/                    build output (ignored by git)
```

## Build

Prerequisites: Xcode 15+ (26.1 tested), Rust with `rustup target add aarch64-apple-ios aarch64-apple-ios-sim
x86_64-apple-ios aarch64-apple-darwin`.

```sh
bindings/ios/build-xcframework.sh            # --no-macos to drop the macOS slice
```

The script builds `runtime/linnet-tts` (staticlib, release profile: fat LTO, one codegen unit, NEON picked at
compile time, no `target-cpu`), lipo's the two simulator libraries, strips debug and local symbols, wraps the
C header with a `module.modulemap` (module `CLinnetTTS`) and runs `xcodebuild -create-xcframework`. Output:

```
dist/LinnetTTS.xcframework/{ios-arm64,ios-arm64_x86_64-simulator,macos-arm64}/liblinnet_tts.a
dist/LinnetTTS.xcframework.zip            + .sha256 and .checksum (swift package compute-checksum)
LinnetTTS/LinnetTTS.xcframework           copy for the package's local binary target
```

Slice sizes (stripped; measured on 0.0.1, unchanged in 0.1.0): ios-arm64 12.6 MB, simulator 25 MB (two architectures), macos-arm64 12.8 MB; zip
about 20 MB. Deployment targets: iOS 15, macOS 12 (`IPHONEOS_DEPLOYMENT_TARGET` / `MACOSX_DEPLOYMENT_TARGET`
override them). The macOS slice exists so that `swift build` / `swift test` work on an Apple-silicon host; it is
not a supported product.

## Integrate (SwiftPM)

Local: add `bindings/ios/LinnetTTS` as a package dependency (File > Add Package Dependencies > Add Local) or in a
`Package.swift`: `.package(path: "../LinnetTTS")` and `.product(name: "LinnetTTS", package: "LinnetTTS")`. The
sample does exactly this through `XCLocalSwiftPackageReference`.

Release: the root `Package.swift` points at `LinnetTTS-<version>.xcframework.zip` on the GitHub release `v<version>` with
its SwiftPM checksum; the archive is built by the `xcframework` Actions workflow (see `build/README.md`, "Release
checklist"). Consumers add `.package(url: "https://github.com/VoxRT/LinnetTTS.git", from: "0.1.0")` and the product
`LinnetTTS`.

Code signing: the xcframework holds static libraries, nothing to sign separately; the app's own signature covers
the linked code. Bitcode: not produced (Xcode 14+ does not use it). No equivalent of ProGuard exists on iOS; the
Swift API is kept intact by the linker and `linnet_*` symbols are referenced directly. Nothing but `libSystem`
and `AVFoundation` (for `LinnetPlayer`) is required at run time.

## Sample app

`Sample/LinnetSample.xcodeproj`: a text editor, Speak (streams `LinnetSession.step()` chunks into a `LinnetPlayer`
on a background queue, shows prime time, time to first chunk, mean and worst step and the synthesis RTF), Stop,
and a threads picker. It reads `linnet-en-35m.fp16.safetensors`, `tokenizer.model` and `voice.npy` from the app's
Documents directory; weights are not bundled. `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` are
on, so the folder is visible in Finder (device > Files tab, drag the files onto the app), in the Files app and
in Xcode (Window > Devices and Simulators > the app > Download/Replace Container).

Simulator (the model files are copied from `../../local/`; `local/` = the unpacked weights release, see the root
README):

```sh
cd bindings/ios/Sample
xcodebuild -scheme LinnetSample -configuration Release -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build
xcrun simctl boot "iPhone 17 Pro"
xcrun simctl install booted <DerivedData>/Build/Products/Release-iphonesimulator/LinnetSample.app
C=$(xcrun simctl get_app_container booted com.voxrt.linnet.sample data)
cp ../../local/linnet-en-35m.fp16.safetensors ../../local/tokenizer.model "$C/Documents/"
cp ../../local/voices/linnet-en-f1.npy "$C/Documents/voice.npy"
xcrun simctl launch --console-pty booted com.voxrt.linnet.sample -autospeak "Hello there." -repeat 3
```

Launch arguments `-autospeak "<text>"`, `-threads N`, `-repeat N` make the app speak on start, log the timings
(`[linnet-sample] ...`) and exit, so it can be driven from `simctl` / `devicectl` without touching the screen.

Device: `LINNET_DEVICE=<udid> DEVELOPMENT_TEAM=<team> bindings/ios/device-smoke.sh` builds, installs, copies the
model files with `xcrun devicectl device copy to ... --domain-type appDataContainer` and launches the autospeak
runs. The team id is never committed: pass `DEVELOPMENT_TEAM` on the command line or put
`DEVELOPMENT_TEAM = <team>` into the untracked `bindings/ios/local.xcconfig` (included by `Sample/Config.xcconfig`).

## Tests

`LinnetTTS/Tests/LinnetTTSTests`: version and typed errors (missing file -> `LinnetError.io`), constants and the
voice round trip, `synthesize` length == steps * 1920 and stream concatenation == `synthesize` bit for bit,
threads 1 / 4 / auto bit-exact, seed determinism, a timing report per thread count and a ten-utterance
sustained run with an MD5 of the audio (for cross-device comparison). The model files are looked up in
`$LINNET_ASSETS`, in the test bundle's `Assets/` folder, then in `<repo>/local/`; without them the model tests
are skipped.

```sh
cd bindings/ios/LinnetTTS
LINNET_ASSETS=$PWD/../../../local swift test -c release                                       # macOS host
TEST_RUNNER_LINNET_ASSETS=$PWD/../../../local xcodebuild test -scheme LinnetTTS \
    -destination 'platform=iOS Simulator,name=iPhone 17 Pro'                                  # simulator
```

A Swift package test bundle cannot run on a physical device by itself ("tool-hosted testing is unavailable on
device destinations"), so the same test file is also compiled into `LinnetSampleTests`, hosted by the sample
app, with `LinnetTTS/Tests/LinnetTTSTests/Assets/` copied in as a resource folder (`device-smoke.sh tests` does the copy
and the run; the files are ignored by git).

Results (measured on 0.0.1; unchanged in 0.1.0): all tests pass on the macOS host (Apple M5) and on the iPhone 17 Pro simulator, both as the
package tests and as `LinnetSampleTests`; the audio MD5 is identical everywhere, for every thread count. Host
timings (release, 68-step sentence; RTF in this section = prime pass plus all steps over the audio duration, as the test
prints it, while the root README tables use step / 80 ms): threads 1 mean 3.3 ms per step (RTF 5 %), threads 4 mean 3.7-4.7 ms (the
per-step work is too small on an M5 for the pool to pay off). The simulator runs on the same host CPU and gives
the same numbers in a Release test host; it does not predict a phone. Device runs: `device-smoke.sh` (needs an
unlocked login keychain on the Mac for `codesign`).

Device results (measured on 0.0.1, unchanged in 0.1.0; iPhone 13 Pro Max, A15, iOS 27, hosted `LinnetSampleTests`, release): all 6 tests pass; the audio
MD5s equal the macOS host and simulator values for every thread count (same NEON kernels, thread-count independent
result). Timings on the 68-step sentence (5.44 s of audio): threads 1 — prime 79 ms, first chunk 85 ms, mean step
5.50 ms (worst 6.0), RTF 8.3 %; threads 2 (`auto` resolved to the two performance cores when this was measured; it is
one thread now) — prime 46 ms, first chunk 51 ms, mean step 4.57 ms (worst 7.2), RTF 6.6 %; threads 4 — mean step
17.2 ms (worst 97 ms), RTF 23 %: oversubscribing onto the efficiency cores hurts badly, which is why `auto` never exceeds
the performance-core count. Ten utterances back to back
with `auto`: 333 steps, 26.6 s of audio in 3.9 s (RTF 14.6 %, includes ten primes and sustained-load scheduling).
Recommended default on iOS: `threads = 0`, which the runtime resolves to one thread on Apple platforms (see below).

Step time under real-time pacing: when the sample paces synthesis to playback (`-bufferms 400`: the player blocks
once 400 ms of audio is queued), the CPU idles about 75 ms of every 80 ms step and iOS lowers the core clock and moves the
short bursts to efficiency cores; the same A15 then shows 23 ms per step instead of 5.5 ms. This is power
management, not runtime cost: `-noaudio` or `-nopace` (synthesis never blocks on the player) restores 5.4 ms
with the audio engine running and the UI updating. Measure throughput without pacing; for the tightest
real-time margin on slow phones, synthesize in bursts with a large queue (`LinnetPlayer(bufferMillis:)`; the default
of 8 s, also the sample's default, does this) so the core runs at full clock and then sleeps. `device-pace-experiment.sh` runs these modes on a device.

Baseline device (iPhone 11, A13, iOS 26, sample app, 80-step sentence): threads 1 — prime 97–106 ms, first chunk
107–116 ms, mean step 7.1 ms (worst 9.6) without pacing; threads 2 (auto) — prime 61–74 ms, first chunk 72–86 ms,
mean step 6.9–8.1 ms (the second thread helps the prime, not the step); under real-time pacing both show ≈ 22 ms
per step for the power-management reason described above. The A13 is the oldest device measured (the package
targets iOS 15, which runs on older chips) and leaves a ten-fold margin against the 80 ms step budget. Hosted XCTests on the A13: 6/6, audio MD5s identical to the A15 and
the Mac; threads 1 — 6.7 ms per step, threads 2 — 11.4 ms (worst 31), threads 4 — 47 ms: the pool hurts the step on
Apple silicon, so `threads = 0` now means one thread there (the prime pass is the only part that gains from a second
core, 97 → 68 ms).
