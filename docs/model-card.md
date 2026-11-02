# Model card: `linnet-en-35m` (draft)

> Draft for review. Items marked **TODO** must be settled before the weights are published.

## Summary

Linnet is a lightweight on-device text-to-speech model: an autoregressive flow language model that emits one continuous 48-dimensional
latent per 80 ms step (two 24-dimensional codec frames at 25 Hz) with a single flow evaluation per step, followed by a Vocos-style neural
codec decoder that turns latents into 24 kHz audio. The model speaks English, clones a voice from a short prompt (as voice latents produced
by a separate encoder that is not part of this release), streams audio from the first step, and runs in real time on a single CPU core.

| | |
|---|---|
| Parameters | 41.4 M total: flow LM 31.5 M (text table 3.1 M), codec decoder 9.9 M |
| Sample rate | 24 kHz, mono |
| Latent grid | 25 Hz codec frames (24-dim); LM step 12.5 Hz (48-dim) |
| Context | voice prompt (≈5 s → 63 LM positions) + text tokens + generated steps; tested up to ~11 s of output |
| Text frontend | SentencePiece unigram, 6000 pieces, no phonemizer, no text normalisation inside the model |
| Sampling | 1 flow step per frame; `temp` scales the Gaussian draw (default 0.3); classifier-free guidance `cfg` on the trunk output (default 2.0); EOS head |
| Weights | safetensors `linnet-tts/1`; fp16 matrices for the runtime, fp32 reference |
| Languages | English only |

## Architecture

- **Trunk**: 6 pre-LayerNorm transformer layers, width 512, 8 heads, rotary positions (interleaved pairs, max period 10⁴), GELU (tanh) FFN
  with 4× expansion and no biases. Inputs: a learned "begin voice" embedding, the projected voice-prompt latents, the text token embeddings,
  then the projected previously generated latents. KV cache for streaming.
- **Flow head**: 6 AdaLN MLP blocks of width 512 conditioned on the trunk state and on two sinusoidal time embeddings; at inference the head is
  evaluated once per step on `x0 ~ N(0, temp·I)` and the latent is `x0 + v`. Latents are generated in a normalised space and de-normalised with
  stored mean/std.
- **EOS head**: a single logit on the trunk state; the step on which it fires is not emitted.
- **Codec decoder**: 1×1 projection 24→384, 2 causal ConvNeXt blocks at 25 Hz, ×4 upsampling to the 100 Hz STFT-column grid, 8 causal ConvNeXt blocks,
  a linear head producing 513 log-magnitudes and 513 phases per column, inverse FFT (1024) with a 960-sample Hann window, hop 240, causal overlap-add.
  The decoder reproduces its input delayed by 720 samples; the runtime compensates for this.

Every operator is causal; there is no lookahead anywhere in the chain. See `docs/streaming-contract.md` and `docs/weights-format.md`.

## Training data

**Codec** (encoder + decoder, trained first and then frozen): about 5 600 hours in 8 languages (German, English, Spanish, French, Chinese,
Russian, Italian, Portuguese) from public corpora: LibriVox-derived multilingual TTS corpora (CC BY 4.0), HiFiTTS-2 (CC BY 4.0), LibriTTS-R
(CC BY 4.0), Emilia-YODAS (CC BY 4.0), AISHELL-3 (Apache-2.0), and smaller public sets. **TODO**: finish the license audit of the smaller sets
(HUI German corpus, the Russian LibriVox subset, Google LatAm Spanish — CC BY-SA 4.0) and list every source with its license here.

**Flow LM**: about 6 000 hours of English speech with word-level alignments: Emilia-YODAS English (≈4 200 h, CC BY 4.0), HiFiTTS-2 (≈1 200 h, CC BY 4.0),
LibriTTS-R (≈530 h, CC BY 4.0) and ≈75 h of synthetic speech for numbers, dates, currencies and similar formats. About 10 % of utterances also
carried an alternative written form of the transcript (digits vs words). **TODO**: confirm the terms of use of the synthetic-speech generator's outputs.

No proprietary or user data was used. Common Voice was used for validation only.

## Evaluation

- Intelligibility: on held-out voices the generated speech is transcribed by an independent ASR model with word error rates of 15–30 % on short
  sentences, dominated by ASR tokenisation differences (contractions, "ad"/"add", "route"); canonical test sentences are reproduced verbatim.
- Listening: in blind comparisons against the codec's own reconstruction of real speech, listeners rank the model's output slightly rougher than
  the reconstruction in roughly half of the pairs; against real speech, the codec reconstruction itself is distinguishable. Overall quality was judged
  acceptable for release; the remaining roughness is a known limitation (below).
- fp16 vs fp32 weights: no audible difference in blind listening; the two produce different but equally valid samples because the one-step flow
  amplifies tiny numerical differences into a different trajectory.
- Numerical parity: the PyTorch package reproduces the training-side implementation bit for bit on the reference vectors in `tests/goldens/ref`;
  the Rust runtime matches the PyTorch package to 1e-5 on latents.

## Performance (Rust runtime, single thread, fp16 weights)

| host | per LM step | decoder per frame | first audio | real-time factor |
|---|---|---|---|---|
| desktop x86-64 (AVX2) | 8.1 ms | 1.0 ms | 74 ms after prime | 11 % |

**TODO**: mobile (aarch64 NEON) and browser (WASM SIMD) numbers; int8 variant.

## Intended use and limitations

- English only; other languages produce garbage.
- Numbers, dates, currencies, phone numbers, version strings: the model reads written-out words reliably, raw digit formats only sometimes
  (for example "1359" or "$19.99" are often misread). Normalise text to words before synthesis.
- Voice prompts: 5 s of clean speech, peak-normalised to −3 dBFS, from one speaker; the voice encoder is not included. Short or spontaneous
  prompts can drift in timbre; prompts with strong room acoustics or noise degrade the clone.
- Audio quality: a slight roughness is audible on headphones in some utterances; not audible on laptop speakers.
- The model may be used to synthesise speech in the shipped voices or in voices you have the right to clone. See `WEIGHTS_LICENSE.md` for the
  limits on voice-cloning services. Do not use it to impersonate people without their consent.

## Versioning

`linnet-en-35m` v1: trained in October 2026. The weights file records the training step and whether EMA weights are stored in its metadata.
