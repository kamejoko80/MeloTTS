#!/usr/bin/env python3
import argparse
import inspect
import pathlib
import torch


def resolve_infer_target(tts):
    for cand in [getattr(tts, "model", None), getattr(getattr(tts, "model", None), "model", None)]:
        if cand is not None and hasattr(cand, "infer") and callable(getattr(cand, "infer")):
            return cand
    raise RuntimeError("Cannot find MeloTTS infer() target.")


class EncP_G_Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, x_lengths, tone, language, bert, ja_bert, g):
        out = self.m.enc_p(x, x_lengths, tone=tone, language=language, bert=bert, ja_bert=ja_bert, g=g)
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise RuntimeError("enc_p output must be (x, m_p, logs_p, x_mask)")
        x_h, m_p, logs_p, x_mask = out[0], out[1], out[2], out[3]
        return x_h, m_p, logs_p, x_mask


def torch_onnx_export(model, args, f, **kwargs):
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    return torch.onnx.export(model, args, f, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="EN")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from melo.api import TTS
    from melo.download_utils import load_or_download_config

    tts = TTS(language=args.language, device=args.device)
    m = resolve_infer_target(tts)
    m.eval()

    hps = load_or_download_config(args.language)
    bert_dim = getattr(getattr(hps, "model", None), "bert_dim", None) or getattr(hps, "bert_dim", None) or 1024
    ja_bert_dim = getattr(getattr(hps, "model", None), "ja_bert_dim", None) or getattr(hps, "ja_bert_dim", None) or 768
    gin_channels = getattr(m, "gin_channels", 256)

    L = int(args.L)
    dev = torch.device(args.device)

    x = torch.zeros((1, L), dtype=torch.int64, device=dev)
    x_lengths = torch.tensor([L], dtype=torch.int64, device=dev)
    tone = torch.zeros((1, L), dtype=torch.int64, device=dev)
    language = torch.zeros((1, L), dtype=torch.int64, device=dev)
    bert = torch.zeros((1, int(bert_dim), L), dtype=torch.float32, device=dev)
    ja_bert = torch.zeros((1, int(ja_bert_dim), L), dtype=torch.float32, device=dev)
    g = torch.zeros((1, int(gin_channels), 1), dtype=torch.float32, device=dev)

    wrapper = EncP_G_Wrapper(m).eval().to(dev)
    out_path = pathlib.Path(args.out or f"enc_p_g_L{L}.onnx").resolve()

    with torch.no_grad():
        torch_onnx_export(
            wrapper,
            (x, x_lengths, tone, language, bert, ja_bert, g),
            str(out_path),
            opset_version=args.opset,
            input_names=["x", "x_lengths", "tone", "language", "bert", "ja_bert", "g"],
            output_names=["x_h", "m_p", "logs_p", "x_mask"],
            do_constant_folding=True,
        )

    print("OK:", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
