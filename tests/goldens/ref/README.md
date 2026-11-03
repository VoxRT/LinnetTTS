Reference input/output vectors for the released `linnet-en-35m` weights, produced by the training-side implementation of the model.
The two voices are the released synthetic built-in voices (`f1` = linnet-en-f1, `m1` = linnet-en-m1); `voice_*_prompt.bin` are the
designed-voice utterances (24 kHz, 6 s, peak −3 dBFS) they were encoded from. No recording of a real person is included.
All `.bin` files are float32 little-endian; shapes and sha256 are in `manifest.json`.

Check protocol (`tests/test_ref_goldens.py`): tokens equal; with the recorded per-step noise, generated latents and audio must match
bit for bit on the same machine and PyTorch build (4 threads); on other CPUs only matmul summation-order noise (~1e-6) is acceptable.

License: see `tests/goldens/NOTICE.md`; `voice_*_latent.bin` are copies of the built-in Voice Files (Linnet Weights License) with
express permission to redistribute them as part of this repository for testing.
