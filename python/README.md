# linnet_tts_torch (PyTorch reference)

`linnet_tts_torch` is the reference implementation of LinnetTTS in PyTorch: the model definition (flow LM, codec
decoder, tokenizer wrapper) and inference, no training code. It exists to define the model precisely and to produce
the golden vectors the Rust runtime is tested against; it runs on the CPU in fp32, is bit-exact against the
training-side reference vectors in `tests/goldens/ref`, and is **not optimised for speed**. For on-device use take
the Rust runtime (`runtime/`) or the wheel built from it (`bindings/python/`, `pip install linnet-tts`); the two
packages have different names and do not collide.

## Install

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch (the default index pulls the CUDA build)
pip install ./python            # from the repository root; needs torch >= 2.1, numpy, sentencepiece, safetensors, soundfile
```

## Use

The weights are not in the repository; unpack the weights release into `local/` at the repository root (see the
root README, "Quick start"). The reference package loads the fp32 file.

```python
import soundfile as sf
from linnet_tts_torch import Linnet

lin = Linnet.from_files("local/linnet-en-35m.safetensors", "local/tokenizer.model")
voice = Linnet.load_voice("local/voices/linnet-en-f1.npy")                 # [T, 48] float32 tensor
wav = lin.synthesize("Hello from Linnet.", voice, temp=0.3, cfg=2.0)       # torch.float32 [n] @ 24 kHz
sf.write("hello.wav", wav.numpy(), lin.sample_rate)
```

`synthesize` = `generate` (LM sampling, `[T, 48]` latents; keywords `temp`, `cfg`, `max_frames`, `eos_threshold`,
`generator`) followed by `decode` (latents to waveform with the 720-sample output delay applied). `stream` yields
the audio chunk by chunk per `docs/streaming-contract.md`: 1200 samples after the first step, 1920 after every
later one, 720 zeros at the end; the concatenation equals `synthesize` up to float rounding. Pass a seeded
`torch.Generator` for reproducible sampling. The fp16 runtime file loads as well, but the reference numbers are
defined on the fp32 file. Seeds are not portable to the `linnet-tts` wheel (different random generators): the same seed
gives a different, equally valid utterance there. Blank text, `max_frames < 1`, negative `temp` and a voice of the wrong width
raise `ValueError`, as in the wheel.

## Tests

```sh
LINNET_WEIGHTS=local/linnet-en-35m.safetensors LINNET_TOKENIZER=local/tokenizer.model pytest tests/test_ref_goldens.py -q
```

checks the package bit for bit against `tests/goldens/ref` (the training-side reference set; skipped when the
variables are unset; `LINNET_GOLDEN_TOL` allows summation-order noise on another CPU or torch build).

```sh
python tools/make_goldens.py --weights local/linnet-en-35m.safetensors --tokenizer local/tokenizer.model \
    --voice local/voices/linnet-en-f1.npy
```

regenerates `tests/goldens/hello` and `tests/goldens/meeting`, the runtime parity vectors that
`cd runtime && cargo test --release` compares the Rust implementation against.
