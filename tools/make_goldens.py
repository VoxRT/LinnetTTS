#!/usr/bin/env python
"""Write golden vectors for the runtime parity test (tests/goldens/<case>/).

    python tools/make_goldens.py --weights W.safetensors --tokenizer sp.model --voice voice.npy [--out tests/goldens]

Each case directory holds:
    meta.json     text, token ids, sampling parameters, counts, the voice file used
    noise.bin     [n_noise, 48] f32: the scaled Gaussian draws, one per LM step (the step
                  on which end-of-speech fired included), so another implementation can
                  inject the same noise instead of its own generator
    latents.bin   [n_steps, 48] f32 codec-space latents
    wav.bin       [n_steps * 1920] f32 audio from the offline decode (output delay applied)

Pure CPU, float32. The files are small (texts of a few words) and are meant to be committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from linnet_tts_torch import OUTPUT_DELAY, Linnet  # noqa: E402

CASES = [
    ("hello", "Hello there.", 11),
    ("meeting", "The meeting is at nine.", 7),
]
TEMP, CFG, MAX_FRAMES, EOS = 0.3, 2.0, 60, -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "goldens"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    lin = Linnet.from_files(args.weights, args.tokenizer)
    voice = Linnet.load_voice(args.voice)
    out_root = Path(args.out)
    # Recorded relative to the unpacked weights release (`local/` at the repository root),
    # e.g. "voices/linnet-en-f1.npy"; the bare file name when the voice lives elsewhere.
    voice_path = Path(args.voice).resolve()
    local_dir = Path(__file__).resolve().parents[1] / "local"
    voice_name = voice_path.relative_to(local_dir).as_posix() if voice_path.is_relative_to(local_dir) else voice_path.name

    for name, text, seed in CASES:
        tokens = lin.tokenizer.encode(text)
        # Record every draw the sampler makes.
        drawn: list[torch.Tensor] = []
        orig_randn = torch.randn

        def recording_randn(*a, **k):
            t = orig_randn(*a, **k)
            drawn.append(t.clone())
            return t

        g = torch.Generator().manual_seed(seed)
        torch.randn = recording_randn
        try:
            latents = lin.generate(text, voice, temp=TEMP, cfg=CFG, max_frames=MAX_FRAMES,
                                   eos_threshold=EOS, generator=g)
        finally:
            torch.randn = orig_randn
        n_steps = latents.shape[0]
        eos_fired = n_steps < MAX_FRAMES
        assert len(drawn) == n_steps + (1 if eos_fired else 0), (len(drawn), n_steps)
        noise = torch.cat([TEMP ** 0.5 * d for d in drawn], 0)  # same op as sample_step
        # The same sequence must be reproducible from the seed alone.
        g2 = torch.Generator().manual_seed(seed)
        redraw = torch.cat([TEMP ** 0.5 * torch.randn(1, 48, generator=g2) for _ in drawn], 0)
        assert torch.equal(noise, redraw)

        wav = lin.decode(latents)
        assert wav.shape[0] == n_steps * 1920, wav.shape
        g3 = torch.Generator().manual_seed(seed)
        stream = torch.cat(list(lin.stream(text, voice, temp=TEMP, cfg=CFG, max_frames=MAX_FRAMES,
                                           eos_threshold=EOS, generator=g3)))
        stream_delta = float((stream - wav).abs().max())

        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        noise.numpy().astype("<f4").tofile(d / "noise.bin")
        latents.numpy().astype("<f4").tofile(d / "latents.bin")
        wav.numpy().astype("<f4").tofile(d / "wav.bin")
        meta = {
            "text": text,
            "voice": voice_name,
            "tokens": tokens,
            "temp": TEMP,
            "cfg": CFG,
            "max_frames": MAX_FRAMES,
            "eos_threshold": EOS,
            "seed": seed,
            "n_steps": n_steps,
            "n_noise": noise.shape[0],
            "eos_fired": eos_fired,
            "latent_dim": 48,
            "wav_len": int(wav.shape[0]),
            "sample_rate": lin.sample_rate,
            "output_delay": OUTPUT_DELAY,
            "stream_vs_offline_max_abs": stream_delta,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"{name}: {len(tokens)} tokens, {n_steps} steps ({n_steps * 0.08:.2f} s), eos={eos_fired}, "
              f"wav peak {float(wav.abs().max()):.3f}, stream-vs-offline {stream_delta:.2e}")


if __name__ == "__main__":
    main()
