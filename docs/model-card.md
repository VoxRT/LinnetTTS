# Model card: `linnet-en-35m`

## Summary

Linnet is a lightweight on-device text-to-speech model: an autoregressive flow language model that emits one continuous 48-dimensional
latent per 80 ms step (two 24-dimensional codec frames at 25 Hz) with a single flow evaluation per step, followed by a Vocos-style neural
codec decoder that turns latents into 24 kHz audio. The model speaks English in a voice given as a voice-prompt latent,
streams audio from the first step, and runs in real time on a single CPU core. It ships with two built-in voices; the encoder that produces
latents from recordings is not part of this release, and creating new voices requires a separate license (see `WEIGHTS_LICENSE.md`).

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

## Training data and licenses

All training data comes from public corpora whose licenses permit commercial use. No proprietary or user data was used. Common Voice was used for
validation only. Each corpus was resampled to 24 kHz, filtered for bandwidth and noise, and cut into utterances; the recordings themselves are not
redistributed with this model. The copyright of the recordings remains with their original creators.

**Codec** (encoder + decoder, trained first and then frozen; the decoder is part of this release): 5 612 hours in 8 languages.

| corpus | language | hours | license | source |
|---|---|---|---|---|
| CML-TTS (de, es, fr, it, pt) | de/es/fr/it/pt | 2 302 | CC BY 4.0 | [OpenSLR 146](https://www.openslr.org/146/), LibriVox recordings |
| HiFiTTS-2 | en | 1 220 | CC BY 4.0 (manifests); audio from LibriVox (public domain) | [nvidia/hifitts-2](https://huggingface.co/datasets/nvidia/hifitts-2) |
| Emilia-YODAS (de, fr, zh) | de/fr/zh | 956 | CC BY 4.0 (the base Emilia set, CC BY-NC 4.0, was not used) | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset), derived from YODAS2 (YouTube, CC BY 3.0) |
| LibriTTS-R | en | 507 | CC BY 4.0 | [OpenSLR 141](https://www.openslr.org/141/) |
| HUI-Audio-Corpus-German (clean) | de | 248 | CC0 (as stated by the authors, arXiv 2106.06309); LibriVox recordings | [opendata.iisys.de](https://opendata.iisys.de/dataset/hui-audio-corpus-german/) |
| YO-CPT-ru | ru | 163 | CC BY 4.0 (annotations); audio CC BY 3.0 (YouTube, via YODAS2) | [NCSpeech/YO-CPT-ru](https://huggingface.co/datasets/NCSpeech/YO-CPT-ru) |
| AISHELL-3 | zh | 82 | Apache-2.0 | [OpenSLR 93](https://www.openslr.org/93/) |
| Thorsten-Voice TV-44kHz-Full | de | 39 | CC0 1.0 | [Thorsten-Voice/TV-44kHz-Full](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full) |
| Crowdsourced Latin American Spanish | es | 36 | CC BY-SA 4.0 | [OpenSLR 61, 71–75](https://www.openslr.org/61/) |
| Vibravox (speech_clean, headset) | fr | 31 | CC BY 4.0 | [Cnam-LMSSC/vibravox](https://huggingface.co/datasets/Cnam-LMSSC/vibravox) |
| SIWIS French Speech Synthesis Database | fr | 11 | CC BY 4.0 | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2353) |
| Other public speech data | ru | 17 | commercial use permitted | — |

**Flow LM** (this release): 6 012 hours of English speech with word-level alignments.

| corpus | hours | license | source |
|---|---|---|---|
| Emilia-YODAS (en) | 4 192 | CC BY 4.0 | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) |
| HiFiTTS-2 | 1 220 | CC BY 4.0; audio from LibriVox (public domain) | [nvidia/hifitts-2](https://huggingface.co/datasets/nvidia/hifitts-2) |
| LibriTTS-R | 527 | CC BY 4.0 | [OpenSLR 141](https://www.openslr.org/141/) |
| synthetic speech for numbers, dates, currencies and similar formats | 74 | generated with Qwen3-TTS (Apache-2.0, no restrictions on outputs), in the voices of corpus speakers | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |

About 10 % of the LM utterances also carried an alternative written form of the transcript (digits instead of words), generated with a text model.

Notes on specific licenses:

- *CC BY-SA 4.0* (Latin American Spanish, 0.6 % of the codec data): whether trained weights are an "adaptation" under ShareAlike is legally unsettled.
  The set is credited here; the weights are published under the Linnet weights license.
- *YouTube-derived corpora* (Emilia-YODAS, YO-CPT-ru): the CC BY licenses were declared by the video authors and collected by the corpus authors,
  who honour removal requests.

Training-time tools that are not part of the release: a WavLM model (MIT) served as a feature teacher for the codec encoder; the decoder follows the
Vocos architecture (Vocos: Siuzdak, 2023) in our own implementation; the LM was trained with a training library whose code is not included here.
Please cite the Emilia papers (arXiv 2407.05361, 2501.15907) when citing the data.

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

## Performance (Rust runtime, fp16 weights)

One step produces 80 ms of audio (one LM step plus two decoder frames). Output is bit-identical for any thread count.

| device | threads | per step | first audio after text | real-time factor |
|---|---|---|---|---|
| desktop x86-64 (AVX2) | 1 | 8.8 ms | 74 ms | 11 % |
| Snapdragon 662, Cortex-A73 (Android) | 1 | 42.4 ms | — | 62 % |
| Snapdragon 662, 4 big cores (Android, auto) | 4 | 14.3 ms | — | 21 % |
| Raspberry Pi Zero 2 W, Cortex-A53 1 GHz (Linux) | 1 | 158 ms | — | 242 % |
| Raspberry Pi Zero 2 W, 4 cores (Linux, auto) | 4 | 59–60 ms | — | 88 % |
| iPhone 11, A13 (iOS) | 1 (auto) | 6.7–7.1 ms | 107–116 ms | 9 % |
| iPhone 13 Pro Max, A15 (iOS) | 1 (auto) | 5.5 ms | 85 ms | 8.3 % |
| iPhone 13 Pro Max, A15 (iOS) | 2 | 4.6 ms | 51 ms | 6.6 % |
| Apple M5 (macOS) | 1 | 3.3 ms | — | 5 % |

Measured without real-time pacing; when synthesis is paced to playback, mobile power management raises the step time 3–4× (see
`bindings/README.md`, "Playback and step time"). An integer (W8A8) variant for weaker devices and a browser build are planned for a later release.

## Built-in voices

Both built-in voices are synthetic: they were designed from a text description with Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0,
no restrictions on outputs), so they are not clones of any real person. A 6 s utterance of each designed voice was resampled to 24 kHz,
peak-normalised to −3 dBFS and encoded into the voice latent that ships here; no denoising was applied.

| voice | file | description used for the design |
|---|---|---|
| `linnet-en-f1` | `voices/linnet-en-f1.npy` (76 × 48) | a bright, confident female voice in her late twenties, American accent, natural energy, news-presenter clarity |
| `linnet-en-m1` | `voices/linnet-en-m1.npy` (76 × 48) | a deep, intense male voice with an American accent, documentary narrator, calm and authoritative |

The voices may be used as product voices under `WEIGHTS_LICENSE.md`. Because they are synthetic, no voice-performer credit applies.

## Intended use and limitations

- English only; other languages produce garbage.
- Numbers, dates, currencies, phone numbers, version strings: the model reads written-out words reliably, raw digit formats only sometimes
  (for example "1359" or "$19.99" are often misread). Normalise text to words before synthesis.
- Voice latents: the model is conditioned on a latent of about 5 s of clean speech from one speaker, peak-normalised to −3 dBFS; short or
  spontaneous prompts drift in timbre, and room acoustics or noise degrade the voice. This describes the model's behaviour, not a permission:
  the standard license allows only the built-in voices and voices VoxRT has expressly authorised in writing.
- Audio quality: a slight roughness is audible on headphones in some utterances; not audible on laptop speakers.
- Under the standard license the model may be used with the built-in voices and with voices VoxRT has expressly authorised in writing.
  Cloning any voice (including your own, with consent), designing new voices, fine-tuning the weights or saving modified, quantised or
  converted copies of them, training models with the weights or their output, and offering general-purpose text-to-speech to others as an
  API or service, free or paid, require a separate written license (help@voxrt.com). It may not be used to impersonate a real person
  deceptively or to present its output as a genuine recording. See `WEIGHTS_LICENSE.md`.
- Not designed or certified as the sole means of conveying safety-critical or emergency information; see `WEIGHTS_LICENSE.md`, Section 10.

## Versioning

`linnet-en-35m` v1: trained in October 2026. The weights file records the training step and whether EMA weights are stored in its metadata.
