# LinnetTTS

Lightweight on-device text-to-speech by VoxRT: an autoregressive flow LM (1 NFE per frame) over a 25 Hz continuous-latent neural codec,
with a Vocos-style decoder. ~41M parameters end to end (31.5M LM + 9.9M decoder), streaming synthesis, built-in voices, fp16 weights.

- `python/` — `linnet_tts_torch`: reference model definition and inference in PyTorch (no training code). The on-device Python package `linnet-tts` is built from the Rust runtime in `bindings/python/`.
- `runtime/` — Rust runtime: `linnet-ops` (only the kernels this model uses) and `linnet-tts` (model, streaming session, C API).
- `bindings/` — platform wrappers: Android AAR, iOS Swift package (xcframework), Linux C library and Python wheel. Browser (WASM) is deferred to a later release.
- `build/` — build and packaging scripts.
- `tests/goldens/` — shared golden vectors; Python and Rust must agree on them.
- `docs/` — model card, weight container format, streaming contract.

Code is licensed under Apache-2.0 (see `LICENSE`). Model weights are distributed separately under the Linnet Weights License
(see `WEIGHTS_LICENSE.md`); the codec encoder used for voice cloning is not part of this release.

## Weights

Model weights are not in this repository and are not bundled in any package. Download them from the GitHub release
`weights-v1` of https://github.com/VoxRT/LinnetTTS (mirror: https://huggingface.co/VoxRT/linnet-en-35m):

| file | purpose |
|---|---|
| `linnet-en-35m.fp16.safetensors` | runtime weights (fp16 matrices, ~83 MB); what the Rust runtime and every binding load |
| `linnet-en-35m.safetensors` | fp32 reference weights (~166 MB) for `linnet_tts_torch` and the golden tests |
| `tokenizer.model` | SentencePiece model |
| `voices/<name>.npy` | built-in voices: float32 `[T, 48]` voice-prompt latents |
| `SHA256SUMS` | checksums of the files above; verify with `sha256sum -c SHA256SUMS` before use |
| `WEIGHTS_LICENSE.md`, `model-card.md` | license and model card of the weights |

The runtime needs three files: the fp16 weights, `tokenizer.model` and one voice `.npy`. Check `SHA256SUMS` after
downloading. The weights are licensed under the Linnet Weights License (`WEIGHTS_LICENSE.md`), not under Apache-2.0.
The weight container is documented in `docs/weights-format.md`.

## Versioning

Code releases are tagged `vX.Y.Z` (one version for the crates, the wheel, the AAR and the Swift package; see
`CHANGELOG.md`); weights releases are tagged `weights-vN`; the weights container format is `linnet-tts/1`.
