# License of the test fixtures

The files under `tests/goldens/` are test fixtures and are not covered by the Apache-2.0 license of the source code
(`LICENSE` at the repository root).

- Six files are Voice Files under the Linnet Weights License (`WEIGHTS_LICENSE.md`):
  - `ref/voice_f1_latent.bin` and `ref/voice_m1_latent.bin`, copies of the built-in voices `linnet-en-f1` and `linnet-en-m1`
    from the `weights-v1` release;
  - `ref/voice_f1_prompt.bin` and `ref/voice_m1_prompt.bin`, the synthetic designed-voice utterances the two voices were
    encoded from (generated with Qwen3-TTS VoiceDesign); no recording of a real person is included;
  - `ref/voice_f1_roundtrip.bin` and `ref/voice_m1_roundtrip.bin`, the codec reconstructions of those utterances.

  VoxRT, as the Licensor, expressly permits redistributing these six files as part of this repository and of any fork, clone,
  mirror or source archive of it, for the purpose of testing the code, without the notices otherwise required by Section 5(a)
  and (b) of that license. Every other use of them, including extracting them for use as product voices, is subject to the
  Linnet Weights License as published with the weights.
- All remaining files (`ref/*_latents.bin`, `ref/*_noise.bin`, `ref/*_wav.bin`, `ref/dec_*.bin`, `hello/*`, `meeting/*`) may
  be redistributed with this repository and its forks, clones, mirrors and source archives, and used for testing the code.
  The Licensor claims no ownership in these files. Files that are Outputs remain subject to the Output restrictions of the
  Linnet Weights License, including the prohibition of model training; independently generated random inputs (`*_noise.bin`)
  are not Outputs merely because they accompany the fixtures.
