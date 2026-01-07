#!/usr/bin/env python3
import argparse
import re
import wave
import time
import numpy as np
import torch
from rknnlite.api import RKNNLite

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


def run_rknn_init(rknn_path: str, core_mask: str):
    r = RKNNLite()
    ret = r.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {rknn_path}, ret={ret}")
    if core_mask == "012":
        cm = RKNNLite.NPU_CORE_0_1_2
    elif core_mask == "01":
        cm = RKNNLite.NPU_CORE_0_1
    elif core_mask == "0":
        cm = RKNNLite.NPU_CORE_0
    elif core_mask == "1":
        cm = RKNNLite.NPU_CORE_1
    elif core_mask == "2":
        cm = RKNNLite.NPU_CORE_2
    else:
        cm = RKNNLite.NPU_CORE_0_1_2
    ret = r.init_runtime(core_mask=cm)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed ret={ret}")
    return r


def to_idx_dtype(arr: np.ndarray, idx_dtype: str):
    if idx_dtype == "int32":
        return arr.astype(np.int32)
    if idx_dtype == "int64":
        return arr.astype(np.int64)
    raise RuntimeError("idx_dtype must be int32 or int64")


def dec_rknn_stitch(dec_rknn, z_full: np.ndarray, g: np.ndarray, chunk_T: int) -> np.ndarray:
    z_full = np.asarray(z_full, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    total_T = z_full.shape[-1]
    num = int(np.ceil(total_T / chunk_T))
    chunks = []
    for i in range(num):
        z_slice = z_full[..., i * chunk_T : (i + 1) * chunk_T]
        sub_T = z_slice.shape[-1]
        if sub_T < chunk_T:
            pad = np.zeros((*z_slice.shape[:-1], chunk_T - sub_T), dtype=np.float32)
            z_in = np.concatenate([z_slice, pad], axis=-1)
        else:
            z_in = z_slice
        out = dec_rknn.inference(inputs=[z_in, g])
        y = np.asarray(out[0])
        if y.ndim == 3:
            y1 = y[0, 0, :]
        elif y.ndim == 2:
            y1 = y[0, :]
        else:
            y1 = y.reshape(-1)
        chunks.append(y1[: 512 * sub_T])
    return np.concatenate(chunks, axis=-1)


def fmt_ms(sec: float) -> str:
    return f"{sec * 1000.0:.2f} ms"


class BertRKNNHook:
    def __init__(self, bert_rknn_path: str, core_mask: str, idx_dtype: str, tokenizer_name: str, seq: int = 128):
        from transformers import AutoTokenizer

        self._cache = {}
        self.idx_dtype = idx_dtype
        self.seq = int(seq)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rknn = run_rknn_init(bert_rknn_path, core_mask)
        self.time_sec = 0.0
        self.calls = 0

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass

    def _tok_np_fixed(self, text: str):
        out = self.tok(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.seq,
        )
        dt = np.int32 if self.idx_dtype == "int32" else np.int64
        ids = out["input_ids"].astype(dt)
        mask = out.get("attention_mask", None)
        mask = mask.astype(dt) if mask is not None else None
        ttype = out.get("token_type_ids", None)
        ttype = ttype.astype(dt) if ttype is not None else None
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

        outs = self.rknn.inference(inputs=[ids, mask, ttype])
        if outs is None or len(outs) == 0:
            raise RuntimeError("BERT RKNN inference returned None/empty. Check BERT model inputs and shapes.")

        hid = np.asarray(outs[0], dtype=np.float32)
        if hid.ndim != 3:
            raise RuntimeError(f"Unexpected BERT output shape: {hid.shape}")
        hid = hid[0]  # (SEQ, H)

        if hid.shape[0] != self.seq:
            raise RuntimeError(f"Unexpected BERT token dim: got {hid.shape[0]} but expect {self.seq}")

        self.time_sec += (time.perf_counter() - t0)
        self.calls += 1
        self._cache[text] = hid
        
        return hid

    def phone_level_feature_T(self, text: str, word2ph: list[int]) -> torch.Tensor:
        hid = self.infer_token_features(text)  # (SEQ, H)
        n = min(len(word2ph), hid.shape[0])

        reps = []
        for i in range(n):
            r = int(word2ph[i])
            if r <= 0:
                continue
            reps.append(np.repeat(hid[i : i + 1, :], r, axis=0))

        if not reps:
            return torch.zeros((hid.shape[1], 1), dtype=torch.float32)

        phone_feat = np.concatenate(reps, axis=0)  # (N_phone, H)
        return torch.from_numpy(phone_feat).to(torch.float32).T  # (H, N_phone)


def install_en_bert_rknn_hook(hook: BertRKNNHook):
    import melo.text.english_bert as en_bert

    def get_bert_feature_rknn(text, word2ph, device=None):
        return hook.phone_level_feature_T(text, word2ph)

    en_bert.get_bert_feature = get_bert_feature_rknn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-rknn", required=True)
    ap.add_argument("--dec-rknn", required=True)

    ap.add_argument("--bert-rknn", default=None, help="Optional: BERT .rknn to accelerate EN frontend.")
    ap.add_argument("--bert-tokenizer", default="bert-base-uncased", help="Tokenizer name/path for EN BERT.")
    ap.add_argument("--bert-seq", type=int, default=128, help="BERT static seq length (must match bert.rknn).")
    ap.add_argument("--bert-idx-dtype", default="int32", choices=["int32", "int64"])

    ap.add_argument("--language", default="EN")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="melo_rknn.wav")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--chunk-T", type=int, default=128)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-scale-w", type=float, default=0.8)
    ap.add_argument("--idx-dtype", default="int32", choices=["int32", "int64"])
    ap.add_argument("--core-mask", default="012", choices=["012", "01", "0", "1", "2"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="Override speaker id. If omitted, uses EN-Default (or first in spk2id).",
    )
    ap.add_argument("--list-speakers", action="store_true", help="Print spk2id mapping and exit.")
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = "cpu"

    bert_hook = None
    if args.bert_rknn is not None:
        bert_hook = BertRKNNHook(
            bert_rknn_path=args.bert_rknn,
            core_mask=args.core_mask,
            idx_dtype=args.bert_idx_dtype,
            tokenizer_name=args.bert_tokenizer,
            seq=args.bert_seq,
        )
        install_en_bert_rknn_hook(bert_hook)

    t0_total = time.perf_counter()

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
        if bert_hook is not None:
            bert_hook.release()
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

    x_np = to_idx_dtype(x.unsqueeze(0).cpu().numpy(), args.idx_dtype)
    xlen_np = to_idx_dtype(np.array([x_len], dtype=np.int64), args.idx_dtype)
    tone_np = to_idx_dtype(tone.unsqueeze(0).cpu().numpy(), args.idx_dtype)
    lang_np = to_idx_dtype(language.unsqueeze(0).cpu().numpy(), args.idx_dtype)
    bert_np = bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)
    ja_bert_np = ja_bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)

    tts = TTS(language=args.language, device=device)
    model = resolve_infer_target(tts)
    model.eval()

    sid = torch.tensor([speaker_id], dtype=torch.int64)
    g_t = model.emb_g(sid).unsqueeze(-1).to(torch.float32)
    g_np = g_t.detach().cpu().numpy().astype(np.float32)

    enc_rknn = run_rknn_init(args.enc_rknn, args.core_mask)
    dec_rknn = run_rknn_init(args.dec_rknn, args.core_mask)

    for _ in range(max(0, int(args.warmup))):
        _ = enc_rknn.inference(inputs=[x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np])
        _ = dec_rknn.inference(inputs=[np.zeros((1, 192, int(args.chunk_T)), np.float32), g_np])
        if bert_hook is not None:
            _ = bert_hook.infer_token_features(text)

    t0_enc = time.perf_counter()
    enc_out = enc_rknn.inference(inputs=[x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np])
    t_enc = time.perf_counter() - t0_enc

    if len(enc_out) < 4:
        raise RuntimeError(f"enc outputs unexpected len={len(enc_out)}")
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
    audio = dec_rknn_stitch(dec_rknn, z_np, g_np, int(args.chunk_T))
    t_dec = time.perf_counter() - t0_dec

    write_wav_i16(args.out, audio, sr, auto_gain=True)

    t_total = time.perf_counter() - t0_total

    try:
        enc_rknn.release()
    except Exception:
        pass
    try:
        dec_rknn.release()
    except Exception:
        pass
    if bert_hook is not None:
        bert_hook.release()

    audio_sec = float(audio.shape[0]) / float(sr) if sr > 0 else 0.0
    rtf_total = (t_total / audio_sec) if audio_sec > 0 else float("inf")
    rtf_front = (t_front / audio_sec) if audio_sec > 0 else float("inf")
    rtf_enc = (t_enc / audio_sec) if audio_sec > 0 else float("inf")
    rtf_cpu = (t_cpu / audio_sec) if audio_sec > 0 else float("inf")
    rtf_dec = (t_dec / audio_sec) if audio_sec > 0 else float("inf")

    print("speaker_id:", speaker_id)
    print("==== Timing ====")
    if bert_hook is not None:
        bert_rtf = (bert_hook.time_sec / audio_sec) if audio_sec > 0 else float("inf")
        print(f"bert_rknn: {fmt_ms(bert_hook.time_sec)} calls={bert_hook.calls} RTF={bert_rtf:.3f}")      
    print("get_text_for_tts_infer:", (t1 - t0) * 1000, "ms")
    print("bert_hook_time:", bert_hook.time_sec * 1000, "ms")        
    print("frontend:", fmt_ms(t_front), "RTF:", f"{rtf_front:.3f}")
    print("enc NPU :", fmt_ms(t_enc), "RTF:", f"{rtf_enc:.3f}")
    print("cpu sdp+flow:", fmt_ms(t_cpu), "RTF:", f"{rtf_cpu:.3f}")
    print("dec NPU :", fmt_ms(t_dec), "RTF:", f"{rtf_dec:.3f}")
    print("TOTAL  :", fmt_ms(t_total), "RTF:", f"{rtf_total:.3f}")
    print("audio_sec:", f"{audio_sec:.3f}", "samples:", int(audio.shape[0]), "sr:", sr)
    print("x_len:", x_len, "T_y:", int(z_np.shape[-1]))
    print("wrote:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
