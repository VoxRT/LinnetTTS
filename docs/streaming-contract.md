# Streaming contract

Both implementations (Python `linnet_tts.synth.Linnet.stream()` and the Rust runtime session) follow the same contract, and the
shared golden vectors in `tests/goldens/` pin it down.

## Grid

| unit | duration | samples @ 24 kHz |
|---|---|---|
| codec frame | 40 ms (25 Hz) | 960 |
| LM step | 80 ms (12.5 Hz) = 2 codec frames | 1920 |
| STFT column | 10 ms (hop 240) | 240 |
| decoder overlap tail (`win − hop`) | 30 ms | 720 |

## Session

1. **Prime**: tokenize the text; feed `[bos_voice] + voice_proj(voice latents) + text tokens` through the trunk once (KV cache). The
   voice latents come from a voice file (`[T, 48]`, 12.5 Hz packing); the package never computes them.
2. **Step** (repeat): draw `x0`, run the trunk for the previous latent (or `bos_latent` for the first step), evaluate the EOS logit, run
   the flow head once (conditional and, when `cfg ≠ 1`, unconditional), get one 48-dim latent. If EOS fires on this step, the latent is
   discarded and the session ends.
3. **Decode**: unpack the latent to two 24-dim frames and run `decoder.stream_step` twice → 1920 new samples.
4. **Output delay**: the decoder reproduces its input delayed by `output_delay = 720` samples. The session drops the first 720 samples
   it produces and, when the session ends, appends 720 zeros, so that the concatenation of all streamed chunks equals the offline
   `decode()` of the same latents sample for sample (up to fp32 overlap-add reassociation, ≤ 1e-5 abs). For live playback the trailing
   zeros may be omitted.

## Determinism

Given the same weights, text, voice file, `temp`, `cfg`, `max_frames` and the same sequence of Gaussian draws (one `[48]` vector per
step, drawn before the EOS check), Python and Rust must produce identical latents (fp32, tolerance 1e-4) and audio within 1e-4 abs.
The Rust runtime reads the noise from a seeded generator; the goldens store the noise so both sides can be checked against each other.

## Latency

First audio is available after the prime pass plus one LM step and two decoder frames; there is no lookahead anywhere in the chain.
Every operator is causal (left-padded convolutions, causal attention over the KV cache, causal overlap-add).
