#!/usr/bin/env python3
import argparse
import re
import wave
import numpy as np
import onnxruntime as ort
import torch

from melo.api import TTS
from melo.download_utils import load_or_download_config
from melo import utils


def write_wav_i16(path: str, audio_f32: np.ndarray, sr: int):
    audio_f32 = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def hparams_spk2id_to_dict(spk2id_obj):
    if spk2id_obj is None:
        return {}
    if isinstance(spk2id_obj, dict):
        return spk2id_obj
    if hasattr(spk2id_obj, "__dict__") and isinstance(spk2id_obj.__dict__, dict) and len(spk2id_obj.__dict__) > 0:
        return dict(spk2id_obj.__dict__)
    try:
        if hasattr(spk2id_obj, "items"):
            return dict(spk2id_obj.items())
    except Exception:
        pass
    return {}


def pick_speaker_id(spk2id: dict, prefer_key="EN-Default") -> int:
    if not spk2id:
        return 0
    if prefer_key in spk2id:
        return int(spk2id[prefer_key])
    return int(next(iter(spk2id.values())))


def resolve_infer_target(tts):
    for cand in [getattr(tts, "model", None), getattr(getattr(tts, "model", None), "model", None)]:
        if cand is not None and hasattr(cand, "infer") and callable(getattr(cand, "infer")):
            return cand
    raise RuntimeError("Cannot find MeloTTS infer() target.")


class DecTap(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        self.last_z = None
        self.last_g = None

    def forward(self, z, g=None, **kwargs):
        self.last_z = z.detach()
        if g is not None:
            self.last_g = g.detach()
        return self.dec(z, g=g, **kwargs)


def decode_with_decoder_onnx(dec_sess, z_full: np.ndarray, g: np.ndarray, T: int) -> np.ndarray:
    z_len = z_full.shape[-1]
    slice_num = int(np.ceil(z_len / T))
    audio_chunks = []
    for i in range(slice_num):
        z_slice = z_full[..., i * T : (i + 1) * T]
        sub_T = z_slice.shape[-1]
        sub_audio_len = 512 * sub_T

        if sub_T < T:
            pad = np.zeros((*z_slice.shape[:-1], T - sub_T), dtype=np.float32)
            z_slice = np.concatenate([z_slice, pad], axis=-1)

        feeds = {}
        in_names = [i.name for i in dec_sess.get_inputs()]
        if "z_p_slice" in in_names:
            feeds["z_p_slice"] = z_slice.astype(np.float32)
        else:
            feeds[in_names[0]] = z_slice.astype(np.float32)

        if "g" in in_names:
            feeds["g"] = g.astype(np.float32)
        else:
            feeds[in_names[1]] = g.astype(np.float32)

        out = dec_sess.run(None, feeds)[0]
        out = np.asarray(out)
        if out.ndim == 3:
            out_1d = out[0, 0, :]
        elif out.ndim == 2:
            out_1d = out[0, :]
        else:
            out_1d = out.reshape(-1)

        audio_chunks.append(out_1d[:sub_audio_len])

    return np.concatenate(audio_chunks, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, help="decoder.onnx")
    ap.add_argument("--language", default="EN")
    ap.add_argument("--text", default="Hello world")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--out-pytorch", default="pytorch_ref.wav")
    ap.add_argument("--out-onnx", default="onnx_decoder.wav")
    args = ap.parse_args()

    device = "cpu"

    hps = load_or_download_config(args.language)
    symbols = hps.symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    lang_code = args.language.split("_")[0]
    lang_code = "ZH_MIX_EN" if lang_code == "ZH" else lang_code

    text = args.text
    if lang_code in ["EN", "ZH_MIX_EN"]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    spk2id_obj = getattr(getattr(hps, "data", None), "spk2id", None)
    spk2id = hparams_spk2id_to_dict(spk2id_obj)
    speaker_id = pick_speaker_id(spk2id, prefer_key="EN-Default")

    tts = TTS(language=args.language, device=device)
    model = resolve_infer_target(tts)
    model.eval()

    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
        text, lang_code, hps, device, symbol_to_id
    )

    x = phones.to(torch.int64).unsqueeze(0)
    tone = tones.to(torch.int64).unsqueeze(0)
    lang = lang_ids.to(torch.int64).unsqueeze(0)
    sid = torch.tensor([int(speaker_id)], dtype=torch.int64)
    x_lengths = torch.tensor([int(phones.numel())], dtype=torch.int64)
    bert_b = bert.to(torch.float32).unsqueeze(0)
    ja_bert_b = ja_bert.to(torch.float32).unsqueeze(0)

    dec_orig = model.dec
    dec_tap = DecTap(dec_orig).eval()
    model.dec = dec_tap

    with torch.no_grad():
        # Use deterministic settings (same style many deployments use for encoder-side ONNX):
        noise_scale = 0.0
        noise_scale_w = 0.0
        sdp_ratio = 0.0
        length_scale = 1.0 / float(args.speed)

        y_pt = model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tone,
            language=lang,
            bert=bert_b,
            ja_bert=ja_bert_b,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
        )
        if isinstance(y_pt, (tuple, list)):
            y_pt = y_pt[0]

    model.dec = dec_orig

    z = dec_tap.last_z
    g = dec_tap.last_g
    if z is None or g is None:
        raise RuntimeError("Failed to capture (z, g) passed into model.dec().")

    y_pt_np = y_pt.detach().cpu().numpy()
    if y_pt_np.ndim == 3:
        y_pt_1d = y_pt_np[0, 0, :]
    elif y_pt_np.ndim == 2:
        y_pt_1d = y_pt_np[0, :]
    else:
        y_pt_1d = y_pt_np.reshape(-1)

    z_np = z.cpu().numpy().astype(np.float32)
    g_np = g.cpu().numpy().astype(np.float32)

    dec_sess = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])
    y_onnx_1d = decode_with_decoder_onnx(dec_sess, z_np, g_np, int(args.T))

    sr = int(getattr(getattr(hps, "data", None), "sampling_rate", 44100))
    write_wav_i16(args.out_pytorch, y_pt_1d, sr)
    write_wav_i16(args.out_onnx, y_onnx_1d, sr)

    n = min(len(y_pt_1d), len(y_onnx_1d))
    if n > 0:
        diff = y_pt_1d[:n] - y_onnx_1d[:n]
        rmse = float(np.sqrt(np.mean(diff * diff)))
    else:
        rmse = float("nan")

    print("OK: captured real (z, g) from PyTorch infer() and decoded with decoder ONNX")
    print("text:", args.text)
    print("speaker_id:", speaker_id)
    print("z:", z_np.shape, "g:", g_np.shape, "T:", args.T)
    print("pytorch wav:", args.out_pytorch, "len:", len(y_pt_1d))
    print("onnx wav:", args.out_onnx, "len:", len(y_onnx_1d))
    print("RMSE (aligned):", rmse)


if __name__ == "__main__":
    main()
