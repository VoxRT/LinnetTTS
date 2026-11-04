"""CPU speed benchmark of Linnet against other small TTS models (docs/benchmarks.md).

Usage: python tools/bench_cpu_tts.py {linnet,pocket,kokoro-fp32,kokoro-int8,piper-medium,piper-high} --threads N
One system, one thread count, three texts -> one JSON line on stdout.

Metrics per sentence (median of RUNS after one warm-up):
  gen_s      wall-clock seconds to produce the whole utterance
  audio_s    seconds of audio produced
  rtf        gen_s / audio_s  (fraction; x100 = percent)
  first_s    seconds until the first audio chunk is available (streaming systems); = gen_s otherwise
"""
import argparse, json, os, resource, statistics, sys, time

SENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Good morning, and welcome back. The meeting starts at nine, and the coffee is already on the table.",
    ("Linnet is a small text to speech model that runs on a single processor core. It starts speaking after the first "
     "eighty milliseconds of audio, and it keeps going without pauses while the text is being read. The voice is cloned "
     "from a few seconds of speech, and the same text always gives the same audio on a given machine."),
]
RUNS = 3
M = os.environ.get("BENCH_MODELS", "models")  # directory layout: see docs/benchmarks.md


def timed(gen):
    """gen: iterator of float32 chunks (numpy). Returns (gen_s, first_s, n_samples)."""
    t0 = time.perf_counter(); first = None; n = 0
    for chunk in gen:
        if first is None:
            first = time.perf_counter() - t0
        n += len(chunk)
    return time.perf_counter() - t0, (first if first is not None else time.perf_counter() - t0), n


def run_linnet(threads):
    import linnet_tts, numpy as np
    m = linnet_tts.Model(f"{M}/linnet/linnet-en-35m.fp16.safetensors", f"{M}/linnet/tokenizer.model")
    v = m.load_voice(f"{M}/linnet/voices/linnet-en-f1.npy")
    sr = m.sample_rate
    def gen(text):
        s = m.session(v, temp=0.3, cfg=2.0, seed=0, threads=threads)
        s.prime(text)
        while True:
            c = s.step()
            if c is None:
                return
            yield c
    return gen, sr


def run_piper(threads, quality):
    from piper import PiperVoice
    import onnxruntime as ort
    import numpy as np
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    path = f"{M}/en_US-lessac-{quality}.onnx"
    try:
        voice = PiperVoice.load(path, use_cuda=False)
        voice.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    except TypeError:
        voice = PiperVoice.load(path)
    sr = voice.config.sample_rate
    def gen(text):
        if hasattr(voice, "synthesize"):
            for ch in voice.synthesize(text):
                if hasattr(ch, "audio_float_array"):
                    yield ch.audio_float_array
                else:
                    yield np.frombuffer(ch, dtype=np.int16).astype(np.float32) / 32768
        else:
            for raw in voice.synthesize_stream_raw(text):
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    return gen, sr


def run_kokoro(threads, variant):
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    path = f"{M}/kokoro-v1.0{'.int8' if variant == 'int8' else ''}.onnx"
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, f"{M}/voices-v1.0.bin")
    sr = 24000
    import asyncio
    def gen(text):
        # create_stream splits on sentences and yields each one as soon as it is synthesised
        loop = asyncio.new_event_loop(); it = k.create_stream(text, voice="af_heart", speed=1.0, lang="en-us").__aiter__()
        try:
            while True:
                try:
                    samples, _sr = loop.run_until_complete(it.__anext__())
                except StopAsyncIteration:
                    return
                yield samples
        finally:
            loop.close()
    return gen, sr


def run_pocket(threads):
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(threads)
    m = TTSModel.load_model()
    torch.set_num_threads(threads)
    st = m.get_state_for_audio_prompt("alba")
    sr = m.sample_rate
    def gen(text):
        for ch in m.generate_audio_stream(st, text):
            yield ch.numpy()
    return gen, sr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("system"); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.system == "linnet": gen, sr = run_linnet(a.threads)
    elif a.system.startswith("piper-"): gen, sr = run_piper(a.threads, a.system.split("-", 1)[1])
    elif a.system.startswith("kokoro-"): gen, sr = run_kokoro(a.threads, a.system.split("-", 1)[1])
    elif a.system == "pocket": gen, sr = run_pocket(a.threads)
    else: sys.exit("unknown system")
    load_s = time.perf_counter() - t0
    out = {"system": a.system, "threads": a.threads, "sr": sr, "load_s": load_s, "sentences": []}
    for i, text in enumerate(SENTS):
        timed(gen(text))  # warm-up
        rs = [timed(gen(text)) for _ in range(RUNS)]
        gen_s = statistics.median(r[0] for r in rs); first_s = statistics.median(r[1] for r in rs)
        n = rs[0][2]
        out["sentences"].append({"i": i, "chars": len(text), "audio_s": n / sr, "gen_s": gen_s, "first_s": first_s,
                                 "rtf": gen_s / (n / sr) if n else None})
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(out))


if __name__ == "__main__":
    main()
