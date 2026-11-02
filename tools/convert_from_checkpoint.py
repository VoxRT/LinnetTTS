#!/usr/bin/env python
"""Convert training checkpoints (LM + codec) into a LinnetTTS safetensors weight file.

    python tools/convert_from_checkpoint.py --lm <lm_checkpoint.pt> --codec <codec_checkpoint.ckpt> \
        --out local/linnet-en-35m.safetensors [--no-ema]

The LM checkpoint is a dict with "model" (state dict) and optionally "ema" (EMA shadow of every
parameter); EMA weights are applied by default, as at sampling time. The codec checkpoint is a
Lightning-style dict whose "state_dict" carries the decoder under the prefix "base.decoder.".
Only the decoder tensors are taken from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from linnet_tts import DecoderConfig, FlowLM, Linnet, LMConfig, VocosDecoder  # noqa: E402

# training-checkpoint key -> LinnetTTS key (top-level LM tensors)
LM_TOP = {
    "flow_lm.conditioner.embed.weight": "text_embed.weight",
    "flow_lm.bos_before_voice": "bos_voice",
    "flow_lm.speaker_proj_weight": "voice_proj.weight",
    "flow_lm.bos_emb": "bos_latent",
    "flow_lm.input_linear.weight": "latent_in.weight",
    "flow_lm.out_norm.weight": "out_norm.weight",
    "flow_lm.out_norm.bias": "out_norm.bias",
    "flow_lm.out_eos.weight": "eos.weight",
    "flow_lm.out_eos.bias": "eos.bias",
    "flow_lm.emb_mean": "latent_mean",
    "flow_lm.emb_std": "latent_std",
    "flow_lm.flow_net.cond_embed.weight": "head.cond_proj.weight",
    "flow_lm.flow_net.cond_embed.bias": "head.cond_proj.bias",
    "flow_lm.flow_net.input_proj.weight": "head.in_proj.weight",
    "flow_lm.flow_net.input_proj.bias": "head.in_proj.bias",
    "flow_lm.flow_net.final_layer.linear.weight": "head.out.linear.weight",
    "flow_lm.flow_net.final_layer.linear.bias": "head.out.linear.bias",
    "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight": "head.out.ada.weight",
    "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias": "head.out.ada.bias",
}
# per transformer layer: suffix -> suffix
LM_LAYER = {
    "self_attn.in_proj.weight": "attn.qkv.weight",
    "self_attn.out_proj.weight": "attn.out.weight",
    "norm1.weight": "norm_attn.weight",
    "norm1.bias": "norm_attn.bias",
    "norm2.weight": "norm_ff.weight",
    "norm2.bias": "norm_ff.bias",
    "linear1.weight": "ff_in.weight",
    "linear2.weight": "ff_out.weight",
}
# per flow-head residual block
HEAD_BLOCK = {
    "in_ln.weight": "norm.weight",
    "in_ln.bias": "norm.bias",
    "mlp.0.weight": "fc1.weight",
    "mlp.0.bias": "fc1.bias",
    "mlp.2.weight": "fc2.weight",
    "mlp.2.bias": "fc2.bias",
    "adaLN_modulation.1.weight": "ada.weight",
    "adaLN_modulation.1.bias": "ada.bias",
}
# per time embedder
HEAD_TIME = {
    "mlp.0.weight": "fc1.weight",
    "mlp.0.bias": "fc1.bias",
    "mlp.2.weight": "fc2.weight",
    "mlp.2.bias": "fc2.bias",
    "mlp.3.alpha": "norm.scale",
}
# Non-persistent buffers recomputed from the config (dropped on conversion).
LM_DROP_SUFFIXES = (".freqs",)


def map_lm_key(key: str) -> str | None:
    if key in LM_TOP:
        return LM_TOP[key]
    if key.endswith(LM_DROP_SUFFIXES):
        return None
    pre = "flow_lm.transformer.layers."
    if key.startswith(pre):
        idx, rest = key[len(pre):].split(".", 1)
        return f"trunk.layers.{idx}.{LM_LAYER[rest]}"
    pre = "flow_lm.flow_net.res_blocks."
    if key.startswith(pre):
        idx, rest = key[len(pre):].split(".", 1)
        return f"head.blocks.{idx}.{HEAD_BLOCK[rest]}"
    pre = "flow_lm.flow_net.time_embed."
    if key.startswith(pre):
        idx, rest = key[len(pre):].split(".", 1)
        return f"head.time_embed.{idx}.{HEAD_TIME[rest]}"
    raise KeyError(f"unmapped LM tensor {key!r}")


def convert_lm(payload: dict, apply_ema: bool = True) -> tuple[dict[str, torch.Tensor], int]:
    model = dict(payload["model"])
    n_ema = 0
    if apply_ema and payload.get("ema"):
        for k, v in payload["ema"].items():
            if k in model:
                model[k] = v
                n_ema += 1
    out: dict[str, torch.Tensor] = {}
    for k, v in model.items():
        if not k.startswith("flow_lm."):
            continue  # training-only modules (loss weighting etc.)
        nk = map_lm_key(k)
        if nk is None:
            continue
        if nk == "bos_voice":
            v = v.reshape(-1)
        out[nk] = v.detach().clone().float()
    return out, n_ema


def convert_decoder(blob: dict) -> dict[str, torch.Tensor]:
    sd = blob["state_dict"]
    pre = "base.decoder."
    out = {k[len(pre):]: v.detach().clone().float() for k, v in sd.items() if k.startswith(pre)}
    if not out:
        raise KeyError(f"no tensors under {pre!r} in codec checkpoint")
    return out


def decoder_config_from_blob(blob: dict) -> DecoderConfig:
    hp = dict(blob.get("hyper_parameters", {}))
    kw = {}
    if hp.get("dec_n_low") is not None:
        kw["n_low"] = int(hp["dec_n_low"])
    if hp.get("dec_n_high") is not None:
        kw["n_high"] = int(hp["dec_n_high"])
    cfg = DecoderConfig(**kw)
    shift = int(hp.get("target_shift", 0) or 0)
    if shift and shift != cfg.ola_tail:
        raise ValueError(f"checkpoint target_shift {shift} != expected output delay {cfg.ola_tail}")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", required=True, type=Path, help="LM training checkpoint (.pt)")
    ap.add_argument("--codec", required=True, type=Path, help="codec training checkpoint (.ckpt)")
    ap.add_argument("--out", required=True, type=Path, help="output .safetensors")
    ap.add_argument("--no-ema", action="store_true", help="use raw (non-EMA) LM weights")
    a = ap.parse_args(argv)

    lm_payload = torch.load(a.lm, map_location="cpu", weights_only=False)
    codec_blob = torch.load(a.codec, map_location="cpu", weights_only=False)

    lm_sd, n_ema = convert_lm(lm_payload, apply_ema=not a.no_ema)
    dec_sd = convert_decoder(codec_blob)
    dec_cfg = decoder_config_from_blob(codec_blob)
    lm_cfg = LMConfig()

    lm, dec = Linnet.build_from_config(lm_cfg, dec_cfg)
    lm.load_state_dict(lm_sd, strict=True)
    dec.load_state_dict(dec_sd, strict=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Linnet.save_weights(a.out, lm, dec, extra_meta={
        "lm_step": lm_payload.get("step", ""), "lm_ema": str(bool(n_ema)),
    })
    print(f"LM params      {lm.n_params():>12,d}  (EMA tensors applied: {n_ema})")
    print(f"decoder params {dec.n_params():>12,d}")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
