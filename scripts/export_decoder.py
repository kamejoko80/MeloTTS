#!/usr/bin/env python3
import argparse
import inspect
import pathlib
import sys
import torch


def _resolve_infer_target(tts):
    for cand in [getattr(tts, "model", None), getattr(getattr(tts, "model", None), "model", None)]:
        if cand is not None and hasattr(cand, "infer") and callable(getattr(cand, "infer")):
            return cand
    raise RuntimeError("Cannot find MeloTTS infer() target (tried tts.model and tts.model.model).")


class DecoderWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z_p_slice: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y = self.dec(z_p_slice, g=g)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y


def _torch_onnx_export(model, args, f, **kwargs):
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    return torch.onnx.export(model, args, f, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="EN")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--z-ch", type=int, default=None, help="override z channels (default tries to infer)")
    ap.add_argument("--g-ch", type=int, default=None, help="override speaker/g channels (default tries to infer)")
    ap.add_argument("--out", default="decoder_T128.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from melo.api import TTS

    tts = TTS(language=args.language, device=args.device)
    infer_target = _resolve_infer_target(tts)

    if not hasattr(infer_target, "dec"):
        raise RuntimeError("infer_target.dec not found. This exporter expects a VITS-style model with .dec")

    dec = infer_target.dec.eval().to(args.device)

    z_ch = args.z_ch
    if z_ch is None:
        z_ch = getattr(infer_target, "inter_channels", None)
    if z_ch is None:
        z_ch = 192

    g_ch = args.g_ch
    if g_ch is None:
        g_ch = getattr(infer_target, "gin_channels", None)
    if g_ch is None:
        g_ch = 256

    T = int(args.T)
    out_path = pathlib.Path(args.out).resolve()

    wrapper = DecoderWrapper(dec).eval().to(args.device)

    z = torch.zeros((1, int(z_ch), T), dtype=torch.float32, device=args.device)
    g = torch.zeros((1, int(g_ch), 1), dtype=torch.float32, device=args.device)

    _torch_onnx_export(
        wrapper,
        (z, g),
        str(out_path),
        opset_version=args.opset,
        input_names=["z_p_slice", "g"],
        output_names=["audio"],
        do_constant_folding=True,
    )

    print("OK: wrote", out_path)
    print(" z shape:", tuple(z.shape), " g shape:", tuple(g.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
