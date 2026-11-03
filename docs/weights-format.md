# Weight container: `linnet-tts/1`

LinnetTTS weights are a single [safetensors](https://github.com/huggingface/safetensors) file. The same file is read by the
Python package and by the Rust runtime; there is no separate "runtime format". The JSON header carries the model configuration
in `__metadata__`, so a loader needs nothing but the file.

## Metadata (`__metadata__`, all values are strings)

| key | example | meaning |
|---|---|---|
| `format` | `linnet-tts/1` | container version; loaders must refuse unknown values |
| `sample_rate` | `24000` | output sample rate |
| `lm_step` | `800000` | training step the LM weights come from (informational) |
| `lm_ema` | `True` | whether the LM tensors are EMA weights (informational) |
| `output_delay` | `720` | samples: the decoder reproduces its input delayed by this amount; synthesis drops the first `output_delay` samples and appends as many zeros |
| `lm_config` | JSON | `latent_dim 48, d_model 512, n_layers 6, n_heads 8, ff_mult 4, rope_max_period 1e4, vocab_size 6000, head_dim 512, head_depth 6, head_time_dim 256, head_time_max_period 1e4` (the token table has `vocab_size + 1` rows: the last one is the padding id) |
| `decoder_config` | JSON | `latent_dim 24, fps_x2 50, d_low 384, n_low 2, d_high 384, n_high 8, expand 3, kernel 7, hop 240, win 960, n_fft 1024, layer_scale_init 1e-6, mag_bias_init -4.0, logmag_min -12.0, logmag_max ln(100)` |

Shapes are derived from the config; loaders validate every tensor against the derived shape and fail closed on a mismatch.

## Tensors (223 in `linnet-en-35m`, 41.4 M parameters)

Dtype policy: the reference file is fp32. The shipped runtime file is fp16 for all weight matrices and fp32 for biases, norms,
`latent_mean`/`latent_std` and the layer-scale vectors; all accumulation in the runtime is fp32. An int8 per-row variant of the
matmul weights (with fp32 scales) is planned as a separate file, not a flag inside this one.

### Flow LM (`lm.*`)

| name | shape | role |
|---|---|---|
| `lm.text_embed.weight` | `[6001, 512]` | SentencePiece token table (6000 pieces + 1 padding row, the last) |
| `lm.bos_voice` | `[512]` | embedding inserted before the voice prompt |
| `lm.voice_proj.weight` | `[512, 48]` | projection of voice-prompt latents into the trunk |
| `lm.bos_latent` | `[48]` | the "start of audio" latent |
| `lm.latent_in.weight` | `[512, 48]` | projection of sampled latents into the trunk |
| `lm.latent_mean`, `lm.latent_std` | `[48]` | latent normalisation (the LM works in normalised latent space; outputs are de-normalised) |
| `lm.trunk.layers.N.norm_attn.{weight,bias}` | `[512]` | pre-attention LayerNorm |
| `lm.trunk.layers.N.attn.qkv.weight` | `[1536, 512]` | fused Q,K,V (no bias); RoPE on interleaved pairs, max period 1e4 |
| `lm.trunk.layers.N.attn.out.weight` | `[512, 512]` | attention output (no bias) |
| `lm.trunk.layers.N.norm_ff.{weight,bias}` | `[512]` | pre-FFN LayerNorm |
| `lm.trunk.layers.N.ff_in.weight` | `[2048, 512]` | FFN (GELU, tanh approximation; no biases) |
| `lm.trunk.layers.N.ff_out.weight` | `[512, 2048]` | FFN |
| `lm.out_norm.{weight,bias}` | `[512]` | final LayerNorm of the trunk |
| `lm.eos.{weight,bias}` | `[1, 512]`, `[1]` | end-of-speech logit |
| `lm.head.cond_proj.{weight,bias}` | `[512, 512]`, `[512]` | trunk state → head conditioning |
| `lm.head.in_proj.{weight,bias}` | `[512, 48]`, `[512]` | noisy latent → head width |
| `lm.head.time_embed.N.fc1.{weight,bias}` (N = 0,1) | `[512, 256]`, `[512]` | two sinusoidal time embeddings (s, t): MLP … |
| `lm.head.time_embed.N.fc2.{weight,bias}` | `[512, 512]`, `[512]` | … |
| `lm.head.time_embed.N.norm.scale` | `[512]` | … + RMSNorm |
| `lm.head.blocks.N.norm.{weight,bias}` | `[512]` | AdaLN block (N = 0…5): LayerNorm, then `(1 + scale) · h + shift`, gated residual |
| `lm.head.blocks.N.ada.{weight,bias}` | `[1536, 512]`, `[1536]` | shift/scale/gate from the conditioning |
| `lm.head.blocks.N.fc1.{weight,bias}` | `[512, 512]`, `[512]` | block MLP (SiLU), first layer |
| `lm.head.blocks.N.fc2.{weight,bias}` | `[512, 512]`, `[512]` | block MLP, second layer |
| `lm.head.out.ada.{weight,bias}` | `[1024, 512]`, `[1024]` | final shift/scale |
| `lm.head.out.linear.{weight,bias}` | `[48, 512]`, `[48]` | velocity output |

Sampling (1 NFE): `x0 ~ N(0, temp · I)`, `v = head(z=x0, s=0, t=1, cond)`, latent `= x0 + v`; classifier-free guidance mixes the
trunk outputs of the conditional pass and of an unconditional pass whose context is `[bos_voice]` only,
`z = z_uncond + cfg · (z_cond − z_uncond)`, and the mixed `z` feeds both the EOS logit and the head (one head evaluation per step).
One LM step produces one 48-dim latent = two consecutive 24-dim codec frames (25 Hz), i.e. 80 ms of audio.

### Codec decoder (`decoder.*`)

| name | shape | role |
|---|---|---|
| `decoder.in_proj.{weight,bias}` | `[384, 24]`, `[384]` | latent → channels (linear, per frame) |
| `decoder.low.N.dw.{weight,bias}` (N = 0,1) | `[384, 1, 7]`, `[384]` | causal ConvNeXt blocks at 25 Hz: depthwise conv, kernel 7 … |
| `decoder.low.N.norm.{weight,bias}` | `[384]` | … LayerNorm … |
| `decoder.low.N.pw1.{weight,bias}` | `[1152, 384]`, `[1152]` | … pointwise expansion ×3 (GELU) … |
| `decoder.low.N.pw2.{weight,bias}` | `[384, 1152]`, `[384]` | … pointwise projection back … |
| `decoder.low.N.gamma` | `[384]` | … layer scale |
| `decoder.up.{weight,bias}` | `[1536, 384]`, `[1536]` | ×4 upsampling to the 100 Hz STFT-column grid (hop 240): linear, 4 × 384 outputs per frame |
| `decoder.high.N.*` (N = 0…7) | as `low` | causal ConvNeXt blocks at 100 Hz, same five tensors per block as `low` |
| `decoder.head_norm.{weight,bias}` | `[384]` | LayerNorm before the head |
| `decoder.head.{weight,bias}` | `[1026, 384]`, `[1026]` | per column: 513 log-magnitudes (clamped to `[logmag_min, logmag_max]`) + 513 phases |

The head output is turned into a complex spectrum `exp(logmag) · exp(i·phase)`, inverse-FFT'ed per column (n_fft 1024, Hann window
960, hop 240) and overlap-added causally. The decoder is streamable frame by frame: each 25 Hz latent frame emits 960 samples, and the
only state is the 720-sample (`win − hop`) overlap tail plus the per-block convolution histories.
