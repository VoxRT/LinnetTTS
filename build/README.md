# build/

- `git-hooks/` — repository hooks. Enable once per clone: `git config core.hooksPath build/git-hooks`.
- `set_version.sh <version>` — sets one version in every package manifest: the Rust crates (`runtime/`), the Python
  wheel (`bindings/python/`), the PyTorch package (`python/`), the Android JNI crate and `VERSION_NAME` of the AAR
  (`bindings/android/`), `let version` of the root `Package.swift`, `MARKETING_VERSION` of the iOS sample, and the
  matching `Cargo.lock` entries. The Linux tarball name, the C API (`linnet_version`),
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

  Pack the directory (`tar -czf linnet-en-35m-weights-v<N>.tar.gz linnet-en-35m-weights-v<N>/`) and attach the archive plus
  the individual files as the assets of the GitHub release `weights-v<N>`; mirror the directory to Hugging Face.

## Release checklist

1. `build/set_version.sh X.Y.Z`, update `CHANGELOG.md`, commit and push.
2. `cd runtime && cargo test --release`; `bindings/linux/build.sh` and `bindings/linux/build.sh --target aarch64-unknown-linux-gnu`.
3. Run the `xcframework` workflow (Actions, "Run workflow", version `X.Y.Z`). It builds `LinnetTTS-X.Y.Z.xcframework.zip`
   and attaches it with its SwiftPM checksum to a draft release `vX.Y.Z`.
4. Put that checksum into `Package.swift` (`let checksum`; `let version` is set by `set_version.sh`), commit, tag `vX.Y.Z`
   and push the tag. `release.yml` builds the wheels, the Linux tarballs and the AAR, verifies the checksum in
   `Package.swift` against the draft asset, publishes the wheels to PyPI (trusted publishing; **not reversible**: a version
   can never be re-uploaded), then attaches everything to the release and publishes it with the CHANGELOG section as notes.
5. Weights release `weights-vN`: `make_weights_release.sh`, pack the archive, `gh release create weights-vN` with the archive
   and the individual files, mirror to Hugging Face (see above).

The xcframework is built once and never rebuilt by the tag workflow: SwiftPM consumers resolve the binary target by
URL and checksum, and a rebuilt archive would not match.
