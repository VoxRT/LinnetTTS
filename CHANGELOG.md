# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).
Code releases are tagged `vX.Y.Z`; model weights are released separately as `weights-vN` (see `README.md`, "Weights").

## [Unreleased]

### Added

- Cached voice state: the trunk state of the voice prefix is computed once per voice and model and reused, so a prime
  pass runs only the text rows. A session keeps the voice from its first prime on (re-priming is cheaper), and a
  *prepared voice* gives new sessions the same head start: Rust `Linnet::voice_state` + `Session::with_voice_state`
  (+ `Linnet::synthesize_with_state`, `Session::collect`, `Session::voice_kept`), C `linnet_voice_prepare` /
  `linnet_voice_is_prepared`, Python `Model.prepare_voice` / `is_prepared`, Kotlin `LinnetModel.prepare` / `isPrepared`,
  Swift `LinnetModel.prepare(voice:)` / `isPrepared(voice:)`; `load_voice` / `loadVoice` / `voiceFromNpy` prepare the
  voice they return. New public types and methods in Rust: `VoiceState` (`frames`, `dim`, `prefix_len`, `is_for`),
  `FlowLm::build_voice_rows` / `build_text_rows`, `TrunkState::truncate` / `grow`. The audio is bit-identical to a full
  prime pass, hence to 0.1.x (the golden vectors are unchanged; `kept_voice_prefix_is_bit_exact` in
  `runtime/linnet-tts/tests/golden.rs` checks fp16 and fp32, with and without guidance, one and four threads). One Core
  i9 core, 44-character sentence: prime 14 ms instead of 67 ms, first audio 22 ms instead of 75 ms; preparation 66 ms
  once per voice, about 2 MB per voice in memory. Behaviour change for existing users: `load_voice` / `loadVoice` /
  `voiceFromNpy` now run one prime pass over the voice (tens of milliseconds, off the main thread like the model load)
  and can fail for that reason; `Voice.load` plus an explicit `prepare` keeps the old split.
- `stream.c` prepares the voice and prints the preparation time; the sample apps log the weights and voice load times
  separately.
- `bindings/ios/device-smoke.sh` passes the sample's arguments after `--` (devicectl took `-autospeak` as its own
  flag and the launches failed silently) and keeps the continuation lines of the app's log.
- `WEIGHTS_PERMISSIONS.md`: written permissions under Section 15.2 of the weights license, starting with one that lets a
  product store a prepared voice state on its own device or server.

### Fixed

- Android `AudioTrackPlayer` played nothing on a Snapdragon 662 (Redmi 9T) and most likely on other Qualcomm devices:
  its 8 s track buffer made the audio policy route the track to a "direct PCM" output that waits for the whole buffer
  to fill, so a short utterance never started and `drain()` returned after its 150 ms no-progress timeout (the 0.1.1
  `drain()` change was a symptom of this). The track is now about 160 ms, asks for the low-latency performance mode
  and runs on the mixer's fast path; the 8 s queue lives in the player and is fed by a thread, so `write()` still
  returns without pacing the synthesis. Consequences for callers: `write()` copies the chunk (the caller may reuse
  its buffer at once); a failure of the track surfaces at the next `write()` or `drain()` as an `IllegalStateException`
  with the original error as its cause, rather than from the `write()` that hit it; `write()` after `close()` throws;
  `stopNow()` pauses the track at once and the feeder thread flushes it.

### Changed

- `AudioTrackPlayer.timing` / `LinnetPlayer.timing` report when the first chunk reached the output (onset, first
  chunk fully played), and `prewarm()` opens the output route before the first chunk; the sample apps take
  `prewarm` (Android: intent extras `autospeak`, `prewarm`, `repeat`; iOS: `-prewarm`) and log the figures.
- Python extension built on pyo3 0.30 (was 0.22.6, the version of three Dependabot advisories that the extension did not
  exercise; issue #2). Same API, same `abi3-py39` wheel; the extension releases the GIL around `load`, `load_voice`,
  `prepare_voice`, `prime`, `step`, `synthesize` and `decode`, as 0.1 did for the calls that existed.

## [0.1.1] - 2026-11-04

### Fixed

- C API: every argument error sets the message of `linnet_last_error`; the header states where a voice's width is validated.
- Linux examples: `wav.h` converts float to 16-bit PCM exactly like the runtime's `synth` example (round half away from zero),
  so the C examples reproduce the reference MD5 of the weights release on x86-64.
- Android `AudioTrackPlayer.drain()` returned only at its deadline on devices whose playback head never reports the last frame;
  it now returns as soon as the head stops advancing.
- PyTorch package: blank text, `max_frames < 1`, negative `temp` and a voice of the wrong width raise `ValueError` (same rules
  as the wheel); the wheel raises `ValueError` for negative `threads`.
- Documentation after an outside-developer pass over every channel: how to depend on the Rust crate, `soundfile` in the
  Python snippets, seeds not portable between the wheel and the PyTorch package, fp32-only parity tolerances, rpath for
  hand-linked C programs, iOS device-build prerequisites and the Swift package tests' binary target, Android first-run
  notes, and an honest statement on cross-architecture output: bit-identical per platform, below 2e-3 across
  x86-64 / Apple / Android / Raspberry Pi (the reference MD5 holds on x86-64).
- A stale prebuilt `_native.abi3.so` is no longer tracked in `bindings/python/`.

## [0.1.0] - 2026-11-04

First public release of LinnetTTS: lightweight on-device English text-to-speech (flow LM over a 25 Hz continuous-latent
codec with a Vocos-style decoder, ~41M parameters, 24 kHz, streaming).

### Added

- `python/` — `linnet_tts_torch`, the reference model definition and inference in PyTorch (no training code). Bit-exact
  against the training-side reference vectors in `tests/goldens/ref` (`tests/test_ref_goldens.py`).
- `runtime/` — Rust runtime. `linnet-ops`: the kernels this model uses, scalar reference plus aarch64 NEON, x86_64 AVX2+FMA
  and WASM SIMD128 fast paths, selected at run time on x86_64. `linnet-tts`: loader for the `linnet-tts/1` safetensors
  container (fp16 weight matrices, fp32 accumulation), SentencePiece tokenizer, streaming `Session` (one LM step = 80 ms =
  1920 samples; the chunks concatenate bit-for-bit to the offline result), C API (`include/linnet_tts.h`), and a persistent
  worker pool whose audio output is bit-identical for every thread count.
- `bindings/linux/` — C distribution `linnet-tts-<version>-linux-{x86_64,aarch64}.tar.gz` (`.so`, `.a`, header, examples).
- `bindings/python/` — wheel `linnet-tts` (import `linnet_tts`; abi3, CPython >= 3.9, NumPy in and out, GIL released).
- `bindings/android/` — AAR `com.voxrt.linnet:linnet-tts` (Kotlin API, JNI in Rust, arm64-v8a + x86_64, minSdk 24,
  `AudioTrackPlayer`) and a sample app (package registry publishing follows the first tagged release; see
  `build/README.md`).
- `bindings/ios/` — Swift package `LinnetTTS` over a static `LinnetTTS.xcframework` (ios-arm64, simulator arm64 + x86_64,
  macos-arm64 for host tests; iOS 15 / macOS 12), `LinnetPlayer`, a SwiftUI sample app and XCTests (package
  registry publishing follows the first tagged release; see `build/README.md`).
- `docs/` — model card, weight container format (`linnet-tts/1`), streaming contract. Golden vectors in `tests/goldens/`
  shared by the Python package and the Rust runtime.
- `build/` — `set_version.sh` (one version for every package), `make_weights_release.sh` (assembles and checks a weights
  release directory), git hooks; GitHub Actions workflows for CI and tagged releases.

### Weights

- `linnet-en-35m` v1 (fp16 runtime file, fp32 reference file, tokenizer, built-in voices) is published separately as the
  GitHub release `weights-v1` with a Hugging Face mirror, under the Linnet Weights License (`WEIGHTS_LICENSE.md`).
  No weights are bundled in any package.

### Known limitations

- English only; other languages produce garbage.
- No text normalisation inside the model: write numbers, dates, currencies and similar as words before synthesis.
- A slight roughness is audible on headphones in some utterances.
- Browser (WASM) and Node channels are deferred to a later release; the runtime keeps compiling for `wasm32-unknown-unknown`.
- An int8 (W8A8) weight variant for weaker devices is planned as a separate weights file.

[Unreleased]: https://github.com/VoxRT/LinnetTTS/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/VoxRT/LinnetTTS/releases/tag/v0.1.1
[0.1.0]: https://github.com/VoxRT/LinnetTTS/releases/tag/v0.1.0
