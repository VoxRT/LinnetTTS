#!/usr/bin/env python
"""Export the runtime weight file: every 2-D `*.weight` matrix as float16, everything else
(biases, norms, depthwise kernels, layer scales, latent statistics) kept as float32. The
metadata is copied unchanged.

    python tools/export_fp16.py --src local/linnet-en-35m.safetensors --dst local/linnet-en-35m.fp16.safetensors
"""

from __future__ import annotations

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def is_matrix_weight(name: str, t: torch.Tensor) -> bool:
    return t.ndim == 2 and name.endswith(".weight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    with safe_open(args.src, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    tensors = load_file(args.src, device="cpu")
    out = {}
    n16 = n32 = 0
    for name, t in tensors.items():
        if is_matrix_weight(name, t):
            out[name] = t.to(torch.float16).contiguous()
            n16 += t.numel()
        else:
            out[name] = t.to(torch.float32).contiguous()
            n32 += t.numel()
    save_file(out, args.dst, metadata=meta)
    print(f"{args.dst}: {n16 / 1e6:.2f} M params as float16, {n32 / 1e6:.3f} M as float32")


if __name__ == "__main__":
    main()
