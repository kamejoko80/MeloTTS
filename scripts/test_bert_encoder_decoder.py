#!/usr/bin/env python3

# Running flow: BERT → ENC → SDP/FLOW → DEC → WAV

import argparse
import os
import re
import time
import wave
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("Please install onnxruntime: pip install onnxruntime") from e

from melo.api import TTS
from melo.download_utils import load_or_download_config
from melo import utils


def write_wav_i16(path: str, audio_f32: np.ndarray, sr: int, auto_gain: bool = True):
    audio_f32 = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if auto_gain and audio_f32.size:
        peak = float(np.max(np.abs(audio_f32)))
        if peak > 0 and np.isfinite(peak):
            audio_f32 = audio_f32 * (0.95 / peak)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_f32 * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, T_x = duration.shape
    _, _, T_y, T_x2 = mask.shape
    if T_x2 != T_x:
        raise RuntimeError("duration/mask T_x mismatch")
    duration = torch.clamp_min(duration, 0)
    cum = torch.cumsum(duration, dim=1)
    start = cum - duration
    t = torch.arange(T_y, device=duration.device).view(1, 1, T_y)
    end = cum.unsqueeze(-1)
    st = start.unsqueeze(-1)
    path = ((t >= st) & (t < end)).to(mask.dtype)
    path = path.permute(0, 2, 1).unsqueeze(1)
    return path * mask


def fmt_ms(sec: float) -> str:
    return f"{sec * 1000.0:.2f} ms"


def make_ort_session(path: str, providers: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    prov = [p.strip() for p in providers.split(",") if p.strip()]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=prov)


def ort_np_dtype_from_onnx_type(onnx_type: str):
    if onnx_type == "tensor(int64)":
        return np.int64
    if onnx_type == "tensor(int32)":
        return np.int32
    if onnx_type == "tensor(float)":
        return np.float32
    if onnx_type == "tensor(float16)":
        return np.float16
    return None


def cast_np(x: np.ndarray, dtype):
    if dtype is None:
        return x
    if x.dtype == dtype:
        return x
    return x.astype(dtype, copy=False)


def enc_ort_run(enc_sess: ort.InferenceSession,
                x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np):
    inputs = enc_sess.get_inputs()
    in_names = [i.name for i in inputs]
    in_types = {i.name: i.type for i in inputs}

    def want(name, fallback_idx):
        return name if name in in_names else in_names[fallback_idx]

    n_x = want("x", 0)
    n_xl = want("x_lengths", 1)
    n_t = want("tone", 2)
    n_l = want("language", 3)
    n_b = want("bert", 4)
    n_j = want("ja_bert", 5)
    n_g = want("g", 6)

    feeds = {
        n_x:  cast_np(x_np,     ort_np_dtype_from_onnx_type(in_types.get(n_x))),
        n_xl: cast_np(xlen_np,  ort_np_dtype_from_onnx_type(in_types.get(n_xl))),
        n_t:  cast_np(tone_np,  ort_np_dtype_from_onnx_type(in_types.get(n_t))),
        n_l:  cast_np(lang_np,  ort_np_dtype_from_onnx_type(in_types.get(n_l))),
        n_b:  cast_np(bert_np,  ort_np_dtype_from_onnx_type(in_types.get(n_b)) or np.float32),
        n_j:  cast_np(ja_bert_np, ort_np_dtype_from_onnx_type(in_types.get(n_j)) or np.float32),
        n_g:  cast_np(g_np,     ort_np_dtype_from_onnx_type(in_types.get(n_g)) or np.float32),
    }
    return enc_sess.run(None, feeds)


def dec_ort_stitch(dec_sess: ort.InferenceSession, z_full: np.ndarray, g: np.ndarray, chunk_T: int) -> np.ndarray:
    z_full = np.asarray(z_full, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    total_T = z_full.shape[-1]
    num = int(np.ceil(total_T / chunk_T))
    chunks = []

    inputs = dec_sess.get_inputs()
    in_names = [i.name for i in inputs]
    in_types = {i.name: i.type for i in inputs}

    z_name = "z_p_slice" if "z_p_slice" in in_names else in_names[0]
    g_name = "g" if "g" in in_names else in_names[1]
    z_dt = ort_np_dtype_from_onnx_type(in_types.get(z_name)) or np.float32
    g_dt = ort_np_dtype_from_onnx_type(in_types.get(g_name)) or np.float32

    for i in range(num):
        z_slice = z_full[..., i * chunk_T : (i + 1) * chunk_T]
        sub_T = z_slice.shape[-1]
        if sub_T < chunk_T:
            pad = np.zeros((*z_slice.shape[:-1], chunk_T - sub_T), dtype=np.float32)
            z_in = np.concatenate([z_slice, pad], axis=-1)
        else:
            z_in = z_slice
        out = dec_sess.run(None, {z_name: cast_np(z_in, z_dt), g_name: cast_np(g, g_dt)})[0]
        y = np.asarray(out)
        if y.ndim == 3:
            y1 = y[0, 0, :]
        elif y.ndim == 2:
            y1 = y[0, :]
        else:
            y1 = y.reshape(-1)
        chunks.append(y1[: 512 * sub_T])
    return np.concatenate(chunks, axis=-1)


class BertOnnxHook:
    def __init__(self, bert_onnx: str, providers: str, tokenizer_name: str, seq: int = 128):
        from transformers import AutoTokenizer

        self._cache = {}
        self.seq = int(seq)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.sess = make_ort_session(bert_onnx, providers)
        self.time_sec = 0.0
        self.calls = 0

        self.inp = self.sess.get_inputs()
        self.in_names = [i.name for i in self.inp]
        self.in_types = {i.name: i.type for i in self.inp}

        def find(name, idx):
            return name if name in self.in_names else self.in_names[idx]

        self.n_ids = find("input_ids", 0)
        self.n_mask = "attention_mask" if "attention_mask" in self.in_names else (self.in_names[1] if len(self.in_names) > 1 else None)
        self.n_ttype = "token_type_ids" if "token_type_ids" in self.in_names else (self.in_names[2] if len(self.in_names) > 2 else None)

        self.dt_ids = ort_np_dtype_from_onnx_type(self.in_types.get(self.n_ids)) or np.int64
        self.dt_mask = ort_np_dtype_from_onnx_type(self.in_types.get(self.n_mask)) if self.n_mask else None
        self.dt_ttype = ort_np_dtype_from_onnx_type(self.in_types.get(self.n_ttype)) if self.n_ttype else None

    def _tok_np_fixed(self, text: str):
        out = self.tok(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.seq,
        )
        ids = cast_np(out["input_ids"], self.dt_ids)
        mask = out.get("attention_mask", None)
        mask = cast_np(mask, self.dt_mask) if mask is not None and self.dt_mask is not None else mask
        ttype = out.get("token_type_ids", None)
        ttype = cast_np(ttype, self.dt_ttype) if ttype is not None and self.dt_ttype is not None else ttype
        return ids, mask, ttype

    def infer_token_features(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        if text in self._cache:
            self.calls += 1
            return self._cache[text]

        ids, mask, ttype = self._tok_np_fixed(text)
        if mask is None:
            mask = (ids != 0).astype(ids.dtype)
        if ttype is None:
            ttype = np.zeros_like(ids)

        feed = {self.n_ids: ids}
        if self.n_mask is not None:
            feed[self.n_mask] = cast_np(mask, self.dt_mask) if self.dt_mask is not None else mask
        if self.n_ttype is not None:
            feed[self.n_ttype] = cast_np(ttype, self.dt_ttype) if self.dt_ttype is not None else ttype

        outs = self.sess.run(None, feed)
        if outs is None or len(outs) == 0:
            raise RuntimeError("BERT ONNX inference returned None/empty.")

        hid = np.asarray(outs[0], dtype=np.float32)
        if hid.ndim == 3:
            hid = hid[0]
        elif hid.ndim != 2:
            raise RuntimeError(f"Unexpected BERT output shape: {hid.shape}")

        if hid.shape[0] != self.seq:
            raise RuntimeError(f"Unexpected BERT token dim: got {hid.shape[0]} but expect {self.seq}")

        self.time_sec += (time.perf_counter() - t0)
        self.calls += 1
        self._cache[text] = hid
        return hid

    def phone_level_feature_T(self, text: str, word2ph: list[int]) -> torch.Tensor:
        hid = self.infer_token_features(text)
        n = min(len(word2ph), hid.shape[0])
        reps = []
        for i in range(n):
            r = int(word2ph[i])
            if r <= 0:
                continue
            reps.append(np.repeat(hid[i : i + 1, :], r, axis=0))
        if not reps:
            return torch.zeros((hid.shape[1], 1), dtype=torch.float32)
        phone_feat = np.concatenate(reps, axis=0)
        return torch.from_numpy(phone_feat).to(torch.float32).T


def install_en_bert_onnx_hook(hook: BertOnnxHook):
    import melo.text.english_bert as en_bert

    def get_bert_feature_onnx(text, word2ph, device=None):
        return hook.phone_level_feature_T(text, word2ph)

    en_bert.get_bert_feature = get_bert_feature_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--dec", required=True)

    ap.add_argument("--bert-tokenizer", default="bert-base-uncased")
    ap.add_argument("--bert-seq", type=int, default=128)

    ap.add_argument("--language", default="EN")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="melo_bert_ort.wav")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--chunk-T", type=int, default=128)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-scale-w", type=float, default=0.8)

    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--speaker-id", type=int, default=None)
    ap.add_argument("--list-speakers", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = "cpu"
    t0_total = time.perf_counter()

    bert_hook = BertOnnxHook(
        bert_onnx=args.bert,
        providers=args.providers,
        tokenizer_name=args.bert_tokenizer,
        seq=int(args.bert_seq),
    )
    install_en_bert_onnx_hook(bert_hook)

    hps = load_or_download_config(args.language)
    sr = int(getattr(getattr(hps, "data", None), "sampling_rate", 44100))
    symbols = hps.symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    spk2id = hparams_spk2id_to_dict(hps)

    if args.list_speakers:
        if not spk2id:
            print("No spk2id found in hparams.")
        else:
            for k, v in spk2id.items():
                print(f"{k}: {v}")
        return 0

    speaker_id = int(args.speaker_id) if args.speaker_id is not None else pick_speaker_id(spk2id)

    lang_code = args.language.split("_")[0]
    lang_code = "ZH_MIX_EN" if lang_code == "ZH" else lang_code

    text = args.text
    if lang_code in ["EN", "ZH_MIX_EN"]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    t0 = time.perf_counter()
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(text, lang_code, hps, device, symbol_to_id)
    t1 = time.perf_counter()
    t_front = t1 - t0

    L = int(args.L)
    x_len = min(int(phones.numel()), L)

    x = pad_or_trunc_1d_int(phones, L, 0)
    tone = pad_or_trunc_1d_int(tones, L, 0)
    language = pad_or_trunc_1d_int(lang_ids, L, 0)
    bert_b = pad_or_trunc_bert(bert, L)
    ja_bert_b = pad_or_trunc_bert(ja_bert, L)

    x_np = x.unsqueeze(0).cpu().numpy()
    xlen_np = np.array([x_len], dtype=np.int64)
    tone_np = tone.unsqueeze(0).cpu().numpy()
    lang_np = language.unsqueeze(0).cpu().numpy()
    bert_np = bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)
    ja_bert_np = ja_bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)

    tts = TTS(language=args.language, device=device)
    model = resolve_infer_target(tts)
    model.eval()

    sid = torch.tensor([speaker_id], dtype=torch.int64)
    g_t = model.emb_g(sid).unsqueeze(-1).to(torch.float32)
    g_np = g_t.detach().cpu().numpy().astype(np.float32)

    enc_sess = make_ort_session(args.enc, args.providers)
    dec_sess = make_ort_session(args.dec, args.providers)

    for _ in range(max(0, int(args.warmup))):
        _ = enc_ort_run(enc_sess, x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np)
        _ = dec_ort_stitch(dec_sess, np.zeros((1, 192, int(args.chunk_T)), np.float32), g_np, int(args.chunk_T))
        _ = bert_hook.infer_token_features(text)

    t0_enc = time.perf_counter()
    enc_out = enc_ort_run(enc_sess, x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np)
    t_enc = time.perf_counter() - t0_enc

    if len(enc_out) < 4:
        raise RuntimeError(f"encoder outputs unexpected len={len(enc_out)}")

    m_p_np, logs_p_np = enc_out[1], enc_out[2]
    m_p = torch.from_numpy(np.asarray(m_p_np)).to(torch.float32)
    logs_p = torch.from_numpy(np.asarray(logs_p_np)).to(torch.float32)

    x_mask = torch.zeros((1, 1, L), dtype=torch.float32)
    x_mask[:, :, :x_len] = 1.0

    t0_cpu = time.perf_counter()
    with torch.no_grad():
        length_scale = 1.0 / max(1e-6, float(args.speed))
        logw = model.sdp(m_p, x_mask, g=g_t, reverse=True, noise_scale=float(args.noise_scale_w))
        w = torch.exp(logw) * x_mask * float(length_scale)
        w_ceil = torch.clamp_min(torch.ceil(w), 1.0) * x_mask

        T_y = int(torch.sum(w_ceil, dim=[1, 2]).long().item())
        T_y = max(1, T_y)
        y_mask = torch.ones((1, 1, T_y), dtype=torch.float32)

        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil.squeeze(1).long(), attn_mask)

        m_p_t = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_t = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p_t + torch.randn_like(m_p_t) * torch.exp(logs_p_t) * float(args.noise_scale)
        z = model.flow(z_p, y_mask, g=g_t, reverse=True)
    t_cpu = time.perf_counter() - t0_cpu

    z_np = z.detach().cpu().numpy().astype(np.float32)

    t0_dec = time.perf_counter()
    audio = dec_ort_stitch(dec_sess, z_np, g_np, int(args.chunk_T))
    t_dec = time.perf_counter() - t0_dec

    write_wav_i16(args.out, audio, sr, auto_gain=True)
    t_total = time.perf_counter() - t0_total

    audio_sec = float(audio.shape[0]) / float(sr) if sr > 0 else 0.0
    rtf_total = (t_total / audio_sec) if audio_sec > 0 else float("inf")
    rtf_front = (t_front / audio_sec) if audio_sec > 0 else float("inf")
    rtf_enc = (t_enc / audio_sec) if audio_sec > 0 else float("inf")
    rtf_cpu = (t_cpu / audio_sec) if audio_sec > 0 else float("inf")
    rtf_dec = (t_dec / audio_sec) if audio_sec > 0 else float("inf")
    bert_rtf = (bert_hook.time_sec / audio_sec) if audio_sec > 0 else float("inf")

    print("speaker_id:", speaker_id)
    print("==== Timing ====")
    print("bert ORT :", fmt_ms(bert_hook.time_sec), "calls=", bert_hook.calls, "RTF:", f"{bert_rtf:.3f}")
    print("frontend:", fmt_ms(t_front), "RTF:", f"{rtf_front:.3f}")
    print("enc ORT :", fmt_ms(t_enc), "RTF:", f"{rtf_enc:.3f}")
    print("cpu sdp+flow:", fmt_ms(t_cpu), "RTF:", f"{rtf_cpu:.3f}")
    print("dec ORT :", fmt_ms(t_dec), "RTF:", f"{rtf_dec:.3f}")
    print("TOTAL  :", fmt_ms(t_total), "RTF:", f"{rtf_total:.3f}")
    print("audio_sec:", f"{audio_sec:.3f}", "samples:", int(audio.shape[0]), "sr:", sr)
    print("x_len:", x_len, "T_y:", int(z_np.shape[-1]))
    print("wrote:", args.out)

    if args.verbose:
        print("bert inputs:", [(i.name, i.type, i.shape) for i in bert_hook.sess.get_inputs()])
        print("bert outputs:", [(o.name, o.type, o.shape) for o in bert_hook.sess.get_outputs()])
        print("enc inputs:", [(i.name, i.type, i.shape) for i in enc_sess.get_inputs()])
        print("enc outputs:", [(o.name, o.type, i.shape) for i, o in zip(enc_sess.get_outputs(), enc_sess.get_outputs())])
        print("dec inputs:", [(i.name, i.type, i.shape) for i in dec_sess.get_inputs()])
        print("dec outputs:", [(o.name, o.type, o.shape) for o in dec_sess.get_outputs()])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
