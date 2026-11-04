# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).
Code releases are tagged `vX.Y.Z`; model weights are released separately as `weights-vN` (see `README.md`, "Weights").

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

[0.1.0]: https://github.com/VoxRT/LinnetTTS/releases/tag/v0.1.0
