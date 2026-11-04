# CPU speed compared with other small TTS models

Measured on 2026-11-04 with the released `linnet-en-35m` weights against three popular CPU text-to-speech models, every one
driven from Python through its published package on the same machine. Quality is not measured here; this page is about
speed, latency and footprint.

## Setup

- Host: desktop x86-64, Core i9-13900K, Linux (WSL2). Each process is pinned to performance cores with `taskset`:
  1 thread → one core, 4 threads → four physical cores. pocket-tts runs as its README prescribes, one torch thread plus
  its own decoder thread, pinned to two cores; that row is marked "(2 cores)".
- Packages: `linnet-tts` 0.1.1 (the Rust wheel), `pocket-tts` 4.0.1 (PyTorch 2.8, CPU build), `kokoro-onnx` 0.6.1
  (onnxruntime 1.23, voice `af_heart`), `piper-tts` 1.8.0 (onnxruntime, voices `en_US-lessac` medium and high).
- Texts: three English texts of 44, 99 and 321 characters (about 2.6, 5.4 and 17 s of speech). One warm-up run, then the
  median of three timed runs per text.
- RTF (real-time factor) = generation wall time / seconds of audio, both summed over the three texts. It includes text
  processing and, for Linnet, the prime pass and the Python loop, so it is higher than the per-step figure in the README.
- First audio = time from the call to the first chunk the API hands back, short text / long text, 1 thread.
- Peak RSS = `ru_maxrss` of the Python process (interpreter and framework included).

## Results

| model | params | weights on disk | voice cloning | streaming | RTF, 1 thread | RTF, 4 threads | first audio, 1 thread | peak RSS |
|---|---|---|---|---|---|---|---|---|
| Linnet `linnet-en-35m`, fp16, Rust runtime | 41 M | 79 MB | yes | every 80 ms | 10.6 % | 3.4 % | 87 / 133 ms | 187 MB |
| pocket-tts 4.0.1 (Kyutai), PyTorch | 100 M | 224 MB | yes | every 80 ms | 23.8 % (2 cores) | 24.1 % | 71 / 82 ms | 968 MB |
| Kokoro-82M v1.0, ONNX fp32 | 81 M | 310 MB + 28 MB voices | no | no | 51.7 % | 17.2 % | 1.5 / 9.6 s | 1246 MB |
| Piper `en_US-lessac-medium`, ONNX | 16 M | 60 MB | no | per sentence | 5.7 % | 2.1 % | 149 / 232 ms | 392 MB |
| Piper `en_US-lessac-high`, ONNX | 28 M | 108 MB | no | per sentence | 41.1 % | 11.9 % | 1.1 / 1.8 s | 502 MB |

Notes:

- Kokoro returns the whole utterance from one call (`kokoro-onnx` `create_stream` also returned a single chunk here). A
  caller that splits the text into sentences gets the first audio after the first sentence, roughly the short-text figure.
  The int8 ONNX of Kokoro measured slower than fp32 on this CPU (74 % at 1 thread, 59 % at 4), so the fp32 row is shown.
- pocket-tts does not get faster with more torch threads (by design, two cores). Its first-audio latency at one thread is a
  little lower than Linnet's because its voice state is computed once per voice, while Linnet re-reads the voice prompt in
  the prime pass of every session.
- Piper medium is the fastest and smallest model in the table; it is a VITS model without voice cloning and without
  streaming inside a sentence.
- Linnet's step alone takes 7.0–7.4 ms at 1 thread on this CPU (README, "Performance"); the mobile and Raspberry Pi
  numbers are in the README and the model card.

## Reproduce

```bash
python -m venv bench && . bench/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install linnet-tts==0.1.1 pocket-tts kokoro-onnx onnxruntime piper-tts soundfile
mkdir models && cd models
# Linnet weights (release weights-v1 of this repository)
curl -LO https://github.com/VoxRT/LinnetTTS/releases/download/weights-v1/linnet-en-35m-weights-v1.tar.gz
mkdir linnet && tar xzf linnet-en-35m-weights-v1.tar.gz -C linnet --strip-components=1
# Piper voices
for q in medium high; do for ext in onnx onnx.json; do
  curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/$q/en_US-lessac-$q.$ext; done; done
# Kokoro ONNX
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..
# pocket-tts downloads its own weights on first use.
for s in linnet pocket kokoro-fp32 piper-medium piper-high; do
  taskset -c 2 python tools/bench_cpu_tts.py $s --threads 1
  taskset -c 2,4,6,8 python tools/bench_cpu_tts.py $s --threads 4
done
```

`tools/bench_cpu_tts.py` prints one JSON line per run (per-text audio seconds, generation seconds, first-chunk seconds, RTF,
peak RSS). Pin pocket-tts to two cores for its 1-thread run.
