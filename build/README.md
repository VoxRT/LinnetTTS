# build/

- `git-hooks/` — repository hooks. Enable once per clone: `git config core.hooksPath build/git-hooks`.
- `set_version.sh <version>` — sets one version in every package manifest: the Rust crates (`runtime/`), the Python
  wheel (`bindings/python/`), the PyTorch package (`python/`), the Android JNI crate and `VERSION_NAME` of the AAR
  (`bindings/android/`), and the matching `Cargo.lock` entries. The Linux tarball name, the C API (`linnet_version`),
  the Python module, the Kotlin and the Swift wrappers all read the `linnet-tts` crate version, so this is the only
  place to change. Prints each file it changed and checks that the lock files still match.
- `make_weights_release.sh` — assembles a weights release directory `linnet-en-35m-weights-v<N>/` from the fp32 and
  fp16 safetensors files, the tokenizer and a directory of voice `.npy` files, adds `WEIGHTS_LICENSE.md` and the
  model card, validates the safetensors headers (container format `linnet-tts/1`, dtype policy, identical tensor
  sets and metadata) and the voices (float32 `[T, 48]`), and writes `SHA256SUMS` and `MANIFEST.json`. With
  `--verify-runtime` it also builds the `synth` example and records the MD5 of a fixed utterance (text, seed, temp,
  cfg, threads, voice, runtime version) in the manifest for device smoke tests. Python 3 standard library only.

  ```sh
  build/make_weights_release.sh --fp32 linnet-en-35m.safetensors --fp16 linnet-en-35m.fp16.safetensors \
      --tokenizer tokenizer.model --voices voices/ --out /tmp/weights --version 1 --verify-runtime
  ```

  Upload the contents of the output directory as the assets of the GitHub release `weights-v<N>` and mirror them.

## Release checklist

1. `build/set_version.sh X.Y.Z`, update `CHANGELOG.md`, commit and push.
2. `cd runtime && cargo test --release`; `bindings/linux/build.sh` and `bindings/linux/build.sh --target aarch64-unknown-linux-gnu`.
3. Run the `xcframework` workflow (Actions, "Run workflow", version `X.Y.Z`). It builds `LinnetTTS-X.Y.Z.xcframework.zip`
   and attaches it with its SwiftPM checksum to a draft release `vX.Y.Z`.
4. Put that checksum and the version into `Package.swift` (`let version`, `let checksum`), commit, tag `vX.Y.Z` and push
   the tag. `release.yml` builds the wheels, the Linux tarballs and the AAR, verifies the checksum in `Package.swift`
   against the draft asset, attaches everything to the release and publishes it.
5. Publish the wheels to PyPI and the weights release `weights-vN` (see `make_weights_release.sh`).

The xcframework is built once and never rebuilt by the tag workflow: SwiftPM consumers resolve the binary target by
URL and checksum, and a rebuilt archive would not match.
