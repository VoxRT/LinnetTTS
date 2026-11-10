# LinnetTTS @VERSION@ — C library for Linux (@ARCH@)

Lightweight on-device text-to-speech: a flow language model over a 25 Hz continuous-latent codec with a
Vocos-style decoder, ~41M parameters, streaming synthesis at 24 kHz. Built from the Rust runtime for
`@TARGET@`; the library has no dependencies beyond libc, libm, libpthread, libdl and libgcc_s; the x86_64 build picks
AVX2+FMA kernels at run time, the aarch64 build uses NEON.

```
lib/liblinnet_tts.so    shared library
lib/liblinnet_tts.a     static library
include/linnet_tts.h    C header (the whole API)
examples/               synth.c (offline), stream.c (streaming), Makefile
```

## Use

```c
#include <stdlib.h>
#include "linnet_tts.h"

linnet_model *model; linnet_voice *voice;
linnet_model_load("linnet-en-35m.fp16.safetensors", "tokenizer.model", &model);
linnet_voice_load("voice.npy", &voice);
linnet_voice_prepare(model, voice);   /* once per voice: sessions then skip the voice part of the prime pass */

size_t cap = 300 * linnet_samples_per_step(model), n;
float *pcm = malloc(cap * sizeof(float));
linnet_synthesize(model, voice, "Hello there.", NULL, pcm, cap, &n);   /* n samples at 24 kHz */

linnet_voice_free(voice);
linnet_model_free(model);
```

Streaming: `linnet_session_new` → `linnet_session_prime(text)` → `linnet_session_step` until it writes 0
samples; a prepared voice makes the prime pass cost only the text rows, and a re-primed session keeps the voice too. Each step is 80 ms of audio (1920 samples); the first chunk is 720 samples shorter and the last is
720 zeros (an utterance that ends on its first step yields no chunk), so the chunks concatenate to exactly what
`linnet_synthesize` returns. Every fallible function returns `LINNET_OK` (0) or a negative `LINNET_ERR_*` code;
`linnet_last_error()` has the message.

Parameters (`linnet_params`, pass NULL for defaults): `temp` 0.3, `cfg` 2.0, `eos_threshold` -1.0,
`max_frames` 300 (24 s), `seed` 0. The same seed and parameters always give the same audio.

## Link

```sh
cc -O2 -Iinclude examples/synth.c -Llib -llinnet_tts -Wl,-rpath,'$ORIGIN/lib' -lm -o synth      # shared
cc -O2 -Iinclude examples/synth.c lib/liblinnet_tts.a -lpthread -ldl -lm -o synth              # static
cc -O2 -Iinclude examples/stream.c -Llib -llinnet_tts -Wl,-rpath,'$ORIGIN/lib' -lm -o stream    # streaming example
```

Model weights are distributed separately under the Linnet Weights License (https://github.com/VoxRT/LinnetTTS).
The code in this archive is Apache-2.0 (see LICENSE).
