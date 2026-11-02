# LinnetTTS

Lightweight on-device text-to-speech by VoxRT: an autoregressive flow LM (1 NFE per frame) over a 25 Hz continuous-latent neural codec,
with a Vocos-style decoder. ~35M parameters end to end, streaming synthesis, built-in voices, fp16 weights.

- `python/` — model definition and inference (pure PyTorch, no training code).
- `runtime/` — Rust runtime: `linnet-ops` (only the kernels this model uses) and `linnet-tts` (model, streaming session, C API).
- `bindings/` — platform wrappers (Android AAR, iOS xcframework, Linux C/Python/Node, browser WASM).
- `build/` — build and packaging scripts.
- `tests/goldens/` — shared golden vectors; Python and Rust must agree on them.
- `docs/` — model card, weight container format, streaming contract.

Code is licensed under Apache-2.0 (see `LICENSE`). Model weights are distributed separately under the Linnet Weights License
(see `WEIGHTS_LICENSE.md`); the codec encoder used for voice cloning is not part of this release.
