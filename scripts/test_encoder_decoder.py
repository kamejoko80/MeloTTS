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
    peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
    if peak > 0 and np.isfinite(peak):
        audio_f32 = audio_f32 * (0.95 / peak)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


def resolve_infer_target(tts):
    for cand in [getattr(tts, "model", None), getattr(getattr(tts, "model", None), "model", None)]:
        if cand is not None and hasattr(cand, "infer") and callable(getattr(cand, "infer")):
            return cand
    raise RuntimeError("Cannot find MeloTTS infer() target.")


def hparams_spk2id_to_dict(hps):
    spk2id_obj = getattr(getattr(hps, "data", None), "spk2id", None)
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


def pick_speaker_id(spk2id: dict) -> int:
    if not spk2id:
        return 0
    if "EN-Default" in spk2id:
        return int(spk2id["EN-Default"])
    return int(next(iter(spk2id.values())))


def pad_or_trunc_1d_int(x: torch.Tensor, L: int, pad_value: int = 0) -> torch.Tensor:
    x = x.to(torch.int64).view(-1)
    n = int(x.numel())
    if n == L:
        return x
    if n > L:
        return x[:L]
    pad = torch.full((L - n,), int(pad_value), dtype=torch.int64, device=x.device)
    return torch.cat([x, pad], dim=0)


def pad_or_trunc_bert(x: torch.Tensor, L: int) -> torch.Tensor:
    x = x.to(torch.float32)
    C, T = x.shape
    if T == L:
        return x
    if T > L:
        return x[:, :L]
    pad = torch.zeros((C, L - T), dtype=torch.float32, device=x.device)
    return torch.cat([x, pad], dim=1)


def dec_onnx_stitch(sess, z_full: np.ndarray, g: np.ndarray, chunk_T: int) -> np.ndarray:
    z_full = np.asarray(z_full, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    total_T = z_full.shape[-1]
    num = int(np.ceil(total_T / chunk_T))
    chunks = []
    in_names = [i.name for i in sess.get_inputs()]
    z_name = "z_p_slice" if "z_p_slice" in in_names else in_names[0]
    g_name = "g" if "g" in in_names else in_names[1]
    for i in range(num):
        z_slice = z_full[..., i * chunk_T : (i + 1) * chunk_T]
        sub_T = z_slice.shape[-1]
        if sub_T < chunk_T:
            pad = np.zeros((*z_slice.shape[:-1], chunk_T - sub_T), dtype=np.float32)
            z_in = np.concatenate([z_slice, pad], axis=-1)
        else:
            z_in = z_slice
        out = sess.run(None, {z_name: z_in, g_name: g})[0]
        out = np.asarray(out)
        if out.ndim == 3:
            out_1d = out[0, 0, :]
        elif out.ndim == 2:
            out_1d = out[0, :]
        else:
            out_1d = out.reshape(-1)
        chunks.append(out_1d[: 512 * sub_T])
    return np.concatenate(chunks, axis=-1)


class EncPFromOnnxG(torch.nn.Module):
    def __init__(self, enc_sess: ort.InferenceSession, L: int):
        super().__init__()
        self.sess = enc_sess
        self.L = int(L)

    def forward(self, x, x_lengths, tone=None, language=None, bert=None, ja_bert=None, g=None):
        L = self.L
        x_np = x.detach().cpu().numpy().astype(np.int64)
        xl_np = x_lengths.detach().cpu().numpy().astype(np.int64)
        tone_np = tone.detach().cpu().numpy().astype(np.int64)
        lang_np = language.detach().cpu().numpy().astype(np.int64)
        bert_np = bert.detach().cpu().numpy().astype(np.float32)
        ja_bert_np = ja_bert.detach().cpu().numpy().astype(np.float32)
        g_np = g.detach().cpu().numpy().astype(np.float32)

        enc_in = [i.name for i in self.sess.get_inputs()]
        feeds = {
            ("x" if "x" in enc_in else enc_in[0]): x_np,
            ("x_lengths" if "x_lengths" in enc_in else enc_in[1]): xl_np,
            ("tone" if "tone" in enc_in else enc_in[2]): tone_np,
            ("language" if "language" in enc_in else enc_in[3]): lang_np,
            ("bert" if "bert" in enc_in else enc_in[4]): bert_np,
            ("ja_bert" if "ja_bert" in enc_in else enc_in[5]): ja_bert_np,
            ("g" if "g" in enc_in else enc_in[6]): g_np,
        }

        x_h_np, m_p_np, logs_p_np, x_mask_np = self.sess.run(None, feeds)

        x_h = torch.from_numpy(np.asarray(x_h_np)).to(dtype=torch.float32, device=x.device)
        m_p = torch.from_numpy(np.asarray(m_p_np)).to(dtype=torch.float32, device=x.device)
        logs_p = torch.from_numpy(np.asarray(logs_p_np)).to(dtype=torch.float32, device=x.device)
        x_mask = torch.from_numpy(np.asarray(x_mask_np)).to(dtype=torch.float32, device=x.device)

        if x_mask.dim() == 3 and x_mask.shape[1] != 1 and x_mask.shape[0] == 1 and x_mask.shape[-1] == L:
            x_mask = x_mask[:, :1, :]

        return x_h, m_p, logs_p, x_mask


class DecFromOnnx(torch.nn.Module):
    def __init__(self, dec_sess: ort.InferenceSession, chunk_T: int):
        super().__init__()
        self.sess = dec_sess
        self.chunk_T = int(chunk_T)

    def forward(self, z, g=None, **kwargs):
        z_np = z.detach().cpu().numpy().astype(np.float32)
        g_np = g.detach().cpu().numpy().astype(np.float32)
        audio = dec_onnx_stitch(self.sess, z_np, g_np, self.chunk_T)
        y = torch.from_numpy(audio).to(dtype=torch.float32, device=z.device).view(1, 1, -1)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--language", default="EN")
    ap.add_argument("--text", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--chunk-T", type=int, default=128)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-scale-w", type=float, default=0.8)
    ap.add_argument("--out", default="onnx_encoder_decoder.wav")
    args = ap.parse_args()

    device = "cpu"
    hps = load_or_download_config(args.language)
    sr = int(getattr(getattr(hps, "data", None), "sampling_rate", 44100))
    symbols = hps.symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    spk2id = hparams_spk2id_to_dict(hps)
    speaker_id = pick_speaker_id(spk2id)

    lang_code = args.language.split("_")[0]
    lang_code = "ZH_MIX_EN" if lang_code == "ZH" else lang_code

    text = args.text
    if lang_code in ["EN", "ZH_MIX_EN"]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
        text, lang_code, hps, device, symbol_to_id
    )

    L = int(args.L)
    x_len = min(int(phones.numel()), L)

    x = pad_or_trunc_1d_int(phones, L, 0).unsqueeze(0).to(torch.int64)
    tone = pad_or_trunc_1d_int(tones, L, 0).unsqueeze(0).to(torch.int64)
    language = pad_or_trunc_1d_int(lang_ids, L, 0).unsqueeze(0).to(torch.int64)
    x_lengths = torch.tensor([x_len], dtype=torch.int64)
    sid = torch.tensor([speaker_id], dtype=torch.int64)
    bert_b = pad_or_trunc_bert(bert, L).unsqueeze(0).to(torch.float32)
    ja_bert_b = pad_or_trunc_bert(ja_bert, L).unsqueeze(0).to(torch.float32)

    tts = TTS(language=args.language, device=device)
    model = resolve_infer_target(tts)
    model.eval()

    g = model.emb_g(sid).unsqueeze(-1)

    enc_sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])

    orig_enc_p = model.enc_p
    orig_dec = model.dec

    model.enc_p = EncPFromOnnxG(enc_sess, L=L).eval()
    model.dec = DecFromOnnx(dec_sess, chunk_T=int(args.chunk_T)).eval()

    with torch.no_grad():
        length_scale = 1.0 / max(1e-6, float(args.speed))
        wav = model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tone,
            language=language,
            bert=bert_b,
            ja_bert=ja_bert_b,
            noise_scale=float(args.noise_scale),
            length_scale=float(length_scale),
            noise_scale_w=float(args.noise_scale_w),
            sdp_ratio=0.0,
        )[0]

    model.enc_p = orig_enc_p
    model.dec = orig_dec

    wav_np = wav.detach().cpu().numpy().astype(np.float32).reshape(-1)
    write_wav_i16(args.out, wav_np, sr)
    print("OK:", args.out, "sr:", sr)


if __name__ == "__main__":
    main()
