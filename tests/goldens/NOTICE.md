# License of the test fixtures

The files under `tests/goldens/` are test fixtures and are not covered by the Apache-2.0 license of the source code
(`LICENSE` at the repository root).

- `ref/voice_f1_latent.bin` and `ref/voice_m1_latent.bin` are copies of the built-in Voice Files `linnet-en-f1` and
  `linnet-en-m1` from the `weights-v1` release. They are governed by the Linnet Weights License (`WEIGHTS_LICENSE.md`).
  VoxRT, as the Licensor, expressly permits redistributing these two files as part of this repository and of any fork,
  clone, mirror or source archive of it, for the purpose of testing the code, without the notices otherwise required by
  Section 5 of that license. Every other use of the files, including extracting them for use as product voices, is under the
  Linnet Weights License as published with the weights.
- `ref/voice_*_prompt.bin` are the synthetic designed-voice utterances the two voices were encoded from (generated with
  Qwen3-TTS VoiceDesign, Apache-2.0); no recording of a real person is included.
- All remaining files (`ref/*_latents.bin`, `ref/*_noise.bin`, `ref/*_wav.bin`, `ref/voice_*_roundtrip.bin`,
  `ref/dec_*.bin`, `hello/*`, `meeting/*`) are model Outputs in the sense of the Linnet Weights License or random inputs; the
  Licensor claims no rights in them and they may be used under Apache-2.0 together with the code.
