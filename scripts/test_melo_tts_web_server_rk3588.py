#!/usr/bin/env python3
import argparse
import re
import time
import io
import wave
import threading
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from rknnlite.api import RKNNLite

from melo.api import TTS
from melo.download_utils import load_or_download_config
from melo import utils

# -------------------------
# Logging
# -------------------------
def _fix_broken_logging_levels():
    import logging
    std = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }
    for name, val in std.items():
        try:
            if logging.getLevelName(val) != name:
                logging.addLevelName(val, name)
        except Exception:
            pass
    try:
        logging._nameToLevel.update(std)
        logging._levelToName.update({v: k for k, v in std.items()})
    except Exception:
        pass


_UVICORN_LOG_CONFIG_NUMERIC = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(levelname)s: %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {"class": "logging.StreamHandler", "formatter": "default"},
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": 20, "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": 20, "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": 20, "propagate": False},
    },
    "root": {"handlers": ["default"], "level": 20},
}


# -------------------------
# WAV helpers
# -------------------------
def wav_bytes_i16(audio_f32: np.ndarray, sr: int, auto_gain: bool = True) -> bytes:
    x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if auto_gain and x.size:
        peak = float(np.max(np.abs(x)))
        if peak > 0 and np.isfinite(peak):
            x = x * (0.95 / peak)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# -------------------------
# Melo helpers
# -------------------------
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




def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving newlines."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ 	]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?。！？;:])\s+|\n+")

def split_text_basic(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[,，])\s+", t) if p.strip()]
    return parts or [t]


def split_in_half(text: str) -> tuple[str, str]:
    t = normalize_text(text)
    if not t:
        return "", ""
    mid = len(t) // 2
    left = t.rfind(" ", 0, mid)
    right = t.find(" ", mid)
    candidates = [x for x in (left, right) if x != -1]
    if candidates:
        cut = min(candidates, key=lambda x: abs(x - mid))
        a = t[:cut].strip()
        b = t[cut:].strip()
        if a and b:
            return a, b
    for ch in [".", "?", "!", "；", ";", ",", "，", "。", "？", "！", "\n"]:
        pos = t.rfind(ch, 0, mid)
        if pos != -1 and pos + 1 < len(t):
            a = t[: pos + 1].strip()
            b = t[pos + 1 :].strip()
            if a and b:
                return a, b
    a = t[:mid].strip()
    b = t[mid:].strip()
    return a, b


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


# -------------------------
# RKNN helpers
# -------------------------
def run_rknn_init(rknn_path: str, core_mask: str) -> RKNNLite:
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


def dec_rknn_stitch(dec_rknn: RKNNLite, z_full: np.ndarray, g: np.ndarray, chunk_T: int) -> np.ndarray:
    z_full = np.asarray(z_full, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    total_T = int(z_full.shape[-1])
    num = int(np.ceil(total_T / int(chunk_T)))
    chunks = []
    for i in range(num):
        z_slice = z_full[..., i * chunk_T : (i + 1) * chunk_T]
        sub_T = int(z_slice.shape[-1])
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


# -------------------------
# BERT RKNN hook (same idea as your working script)
# -------------------------
class BertRKNNHook:
    def __init__(self, bert_rknn_path: str, core_mask: str, idx_dtype: str, tokenizer_name: str, seq: int):
        from transformers import AutoTokenizer
        self.idx_dtype = idx_dtype
        self.seq = int(seq)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.rknn = run_rknn_init(bert_rknn_path, core_mask)
        self._cache: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

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
        with self.lock:
            if text in self._cache:
                return self._cache[text]

        ids, mask, ttype = self._tok_np_fixed(text)
        if mask is None:
            mask = (ids != 0).astype(ids.dtype)
        if ttype is None:
            ttype = np.zeros_like(ids)

        with self.lock:
            outs = self.rknn.inference(inputs=[ids, mask, ttype])

        if outs is None or len(outs) == 0:
            raise RuntimeError("BERT RKNN inference returned empty outputs.")

        hid = np.asarray(outs[0], dtype=np.float32)
        if hid.ndim != 3:
            raise RuntimeError(f"Unexpected BERT output shape: {hid.shape}")

        hid = hid[0]  # (SEQ, H)
        if hid.shape[0] != self.seq:
            raise RuntimeError(f"BERT seq mismatch: got {hid.shape[0]} expected {self.seq}")

        with self.lock:
            self._cache[text] = hid
        return hid

    def phone_level_feature_T(self, text: str, word2ph: List[int]) -> torch.Tensor:
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


# -------------------------
# Engine (RKNN enc/dec + CPU sdp/flow)
# -------------------------
class MeloRKNNEngine:
    def __init__(
        self,
        enc_rknn_path: str,
        dec_rknn_path: str,
        language: str,
        L: int,
        chunk_T: int,
        idx_dtype: str,
        core_mask: str,
        seed: int,
        bert_rknn_path: Optional[str] = None,
        bert_tokenizer: str = "bert-base-uncased",
        bert_seq: int = 256,
        torch_threads: int = 4,
        warmup: int = 1,
    ):
        torch.set_num_threads(int(torch_threads))
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.language = language
        self.L = int(L)
        self.chunk_T = int(chunk_T)
        self.idx_dtype = idx_dtype
        self.core_mask = core_mask

        self.hps = load_or_download_config(language)
        self.sr = int(getattr(getattr(self.hps, "data", None), "sampling_rate", 44100))
        self.symbols = self.hps.symbols
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.spk2id = hparams_spk2id_to_dict(self.hps)

        self.tts = TTS(language=language, device="cpu")
        self.model = resolve_infer_target(self.tts)
        self.model.eval()

        # Melo frontend utilities expect a device string.
        self.device = "cpu"

        self._g_cache: Dict[int, np.ndarray] = {}
        self._g_lock = threading.Lock()

        self.bert_hook: Optional[BertRKNNHook] = None
        if bert_rknn_path is not None:
            self.bert_hook = BertRKNNHook(
                bert_rknn_path=bert_rknn_path,
                core_mask=core_mask,
                idx_dtype=idx_dtype,
                tokenizer_name=bert_tokenizer,
                seq=int(bert_seq),
            )
            install_en_bert_rknn_hook(self.bert_hook)

        self.enc_rknn = run_rknn_init(enc_rknn_path, core_mask)
        self.dec_rknn = run_rknn_init(dec_rknn_path, core_mask)

        self.enc_lock = threading.Lock()
        self.dec_lock = threading.Lock()

        # RKNNLite inference is not thread-safe; serialize end-to-end synthesis.
        self.lock = threading.Lock()

        for _ in range(max(0, int(warmup))):
            self._warmup_once()

    def _warmup_once(self):
        L = int(self.L)
        chunk_T = int(self.chunk_T)
        x_np = to_idx_dtype(np.zeros((1, L), dtype=np.int64), self.idx_dtype)
        xlen_np = to_idx_dtype(np.array([1], dtype=np.int64), self.idx_dtype)
        tone_np = to_idx_dtype(np.zeros((1, L), dtype=np.int64), self.idx_dtype)
        lang_np = to_idx_dtype(np.zeros((1, L), dtype=np.int64), self.idx_dtype)
        bert_np = np.zeros((1, 1024, L), dtype=np.float32)
        ja_bert_np = np.zeros((1, 768, L), dtype=np.float32)
        g_np = np.zeros((1, 256, 1), dtype=np.float32)

        with self.enc_lock:
            _ = self.enc_rknn.inference(inputs=[x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np])
        with self.dec_lock:
            _ = self.dec_rknn.inference(inputs=[np.zeros((1, 192, chunk_T), np.float32), g_np])

    def shutdown(self):
        try:
            self.enc_rknn.release()
        except Exception:
            pass
        try:
            self.dec_rknn.release()
        except Exception:
            pass
        if self.bert_hook is not None:
            self.bert_hook.release()

    def list_speakers(self) -> Dict[str, int]:
        return dict(self.spk2id) if self.spk2id else {}

    def _get_g_np(self, speaker_id: int) -> np.ndarray:
        with self._g_lock:
            if speaker_id in self._g_cache:
                return self._g_cache[speaker_id]
        sid = torch.tensor([int(speaker_id)], dtype=torch.int64)
        with torch.no_grad():
            g = self.model.emb_g(sid).unsqueeze(-1).to(torch.float32)
        g_np = g.detach().cpu().numpy().astype(np.float32)
        with self._g_lock:
            self._g_cache[speaker_id] = g_np
        return g_np


    def _synthesize_one(self, text: str, speaker_id: int, speed: float, noise_scale: float, noise_scale_w: float):
        """Synthesize a single segment. Returns (audio_f32_1d, stats_dict)."""
        t_total0 = time.perf_counter()

        lang_code = self.language.split("_")[0]
        lang_code = "ZH_MIX_EN" if lang_code == "ZH" else lang_code

        seg_text = text
        if lang_code in ["EN", "ZH_MIX_EN"]:
            seg_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", seg_text)

        t0 = time.perf_counter()
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            seg_text, lang_code, self.hps, self.device, self.symbol_to_id
        )
        t_front = time.perf_counter() - t0

        L = int(self.L)
        phones_full_len = int(phones.numel())
        x_len = min(phones_full_len, L)
        truncated = phones_full_len > L

        x = pad_or_trunc_1d_int(phones, L, 0)
        tone = pad_or_trunc_1d_int(tones, L, 0)
        language = pad_or_trunc_1d_int(lang_ids, L, 0)
        bert_b = pad_or_trunc_bert(bert, L)
        ja_bert_b = pad_or_trunc_bert(ja_bert, L)

        x_np = to_idx_dtype(x.unsqueeze(0).cpu().numpy(), self.idx_dtype)
        xlen_np = to_idx_dtype(np.array([x_len], dtype=np.int64), self.idx_dtype)
        tone_np = to_idx_dtype(tone.unsqueeze(0).cpu().numpy(), self.idx_dtype)
        lang_np = to_idx_dtype(language.unsqueeze(0).cpu().numpy(), self.idx_dtype)
        bert_np = bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)
        ja_bert_np = ja_bert_b.unsqueeze(0).cpu().numpy().astype(np.float32)

        sid = torch.tensor([int(speaker_id)], dtype=torch.int64)
        g_t = self.model.emb_g(sid).unsqueeze(-1).to(torch.float32)
        g_np = g_t.detach().cpu().numpy().astype(np.float32)

        t0_enc = time.perf_counter()
        with self.lock:
            enc_out = self.enc_rknn.inference(inputs=[x_np, xlen_np, tone_np, lang_np, bert_np, ja_bert_np, g_np])
        t_enc = time.perf_counter() - t0_enc

        if enc_out is None or len(enc_out) < 4:
            raise RuntimeError(f"enc outputs unexpected: {type(enc_out)} len={0 if enc_out is None else len(enc_out)}")

        m_p_np, logs_p_np = enc_out[1], enc_out[2]
        m_p = torch.from_numpy(np.asarray(m_p_np)).to(torch.float32)
        logs_p = torch.from_numpy(np.asarray(logs_p_np)).to(torch.float32)

        x_mask = torch.zeros((1, 1, L), dtype=torch.float32)
        x_mask[:, :, :x_len] = 1.0

        t0_cpu = time.perf_counter()
        with torch.no_grad():
            length_scale = 1.0 / max(1e-6, float(speed))

            # SDP + flow remain on CPU torch (sdp/flow are not exported to RKNN in this demo)
            logw = self.model.sdp(m_p, x_mask, g=g_t, reverse=True, noise_scale=float(noise_scale_w))
            w = torch.exp(logw) * x_mask * float(length_scale)
            w_ceil = torch.clamp_min(torch.ceil(w), 1.0) * x_mask

            T_y = int(torch.sum(w_ceil, dim=[1, 2]).long().item())
            T_y = max(1, T_y)
            y_mask = torch.ones((1, 1, T_y), dtype=torch.float32)

            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
            attn = generate_path(w_ceil.squeeze(1).long(), attn_mask)

            m_p_t = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p_t = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            z_p = m_p_t + torch.randn_like(m_p_t) * torch.exp(logs_p_t) * float(noise_scale)
            z = self.model.flow(z_p, y_mask, g=g_t, reverse=True)
        t_cpu = time.perf_counter() - t0_cpu

        z_np = z.detach().cpu().numpy().astype(np.float32)

        t0_dec = time.perf_counter()
        with self.lock:
            audio = dec_rknn_stitch(self.dec_rknn, z_np, g_np, self.chunk_T)
        t_dec = time.perf_counter() - t0_dec

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        t_total = time.perf_counter() - t_total0

        stats = {
            "segments": 1,
            "phones_full_len": phones_full_len,
            "x_len": x_len,
            "truncated": bool(truncated),
            "T_y": int(z_np.shape[-1]),
            "audio_samples": int(audio.shape[0]),
            "front_ms": t_front * 1000.0,
            "enc_ms": t_enc * 1000.0,
            "cpu_ms": t_cpu * 1000.0,
            "dec_ms": t_dec * 1000.0,
            "total_ms": t_total * 1000.0,
        }
        return audio, stats

    def synthesize_long_text(self, text: str, speaker_id: int, speed: float, noise_scale: float, noise_scale_w: float, *,
                             silence_ms: float = 60.0, max_recursion: int = 6):
        """Synthesize long text by splitting into segments to avoid L truncation."""
        t0 = time.perf_counter()
        parts = split_text_basic(text)
        if not parts:
            parts = [""]

        audios: list[np.ndarray] = []
        agg = {"segments": 0, "phones_full_len": 0, "x_len": 0, "T_y": 0,
               "front_ms": 0.0, "enc_ms": 0.0, "cpu_ms": 0.0, "dec_ms": 0.0, "total_ms": 0.0,
               "truncated_segments": 0}

        def _run(seg: str, depth: int):
            seg = normalize_text(seg)
            if not seg:
                return
            audio, st = self._synthesize_one(seg, speaker_id, speed, noise_scale, noise_scale_w)
            if st.get("truncated") and depth < max_recursion:
                a, b = split_in_half(seg)
                # If we can split, retry instead of emitting truncated audio
                if a and b and a != seg and b != seg:
                    _run(a, depth + 1)
                    _run(b, depth + 1)
                    return

            audios.append(audio)
            agg["segments"] += 1
            agg["phones_full_len"] += int(st.get("phones_full_len", 0))
            agg["x_len"] += int(st.get("x_len", 0))
            agg["T_y"] += int(st.get("T_y", 0))
            agg["front_ms"] += float(st.get("front_ms", 0.0))
            agg["enc_ms"] += float(st.get("enc_ms", 0.0))
            agg["cpu_ms"] += float(st.get("cpu_ms", 0.0))
            agg["dec_ms"] += float(st.get("dec_ms", 0.0))
            agg["total_ms"] += float(st.get("total_ms", 0.0))
            if st.get("truncated"):
                agg["truncated_segments"] += 1

        for p in parts:
            _run(p, 0)

        if not audios:
            return np.zeros((0,), dtype=np.float32), agg

        gap = int(max(0.0, float(silence_ms)) * float(self.sr) / 1000.0)
        if gap > 0 and len(audios) > 1:
            sil = np.zeros((gap,), dtype=np.float32)
            out = []
            for i, a in enumerate(audios):
                out.append(a)
                if i != len(audios) - 1:
                    out.append(sil)
            audio_all = np.concatenate(out, axis=-1)
        else:
            audio_all = np.concatenate(audios, axis=-1)

        agg["audio_samples"] = int(audio_all.shape[0])
        agg["total_ms"] = max(agg["total_ms"], (time.perf_counter() - t0) * 1000.0)
        return audio_all, agg

    def synthesize(self, text: str, speaker_id: int, speed: float, noise_scale: float, noise_scale_w: float) -> np.ndarray:
        audio, _ = self.synthesize_long_text(text, speaker_id, speed, noise_scale, noise_scale_w)
        return audio
HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MeloTTS RKNN Web Demo</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#0b0f17; color:#e6e8ef; margin:0; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
    .card { background:#121826; border:1px solid #223049; border-radius:14px; padding:16px; box-shadow: 0 8px 30px rgba(0,0,0,.35); }
    h1 { font-size: 20px; margin: 0 0 14px 0; font-weight: 650; }
    textarea { width:100%; min-height: 140px; resize: vertical; background:#0b1020; color:#e6e8ef; border:1px solid #223049; border-radius:10px; padding:12px; font-size:14px; }
    .row { display:flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
    .col { flex: 1; min-width: 220px; }
    label { display:block; font-size:12px; opacity:.9; margin-bottom:6px; }
    select, input[type="range"] {
      width:100%; background:#0b1020; color:#e6e8ef; border:1px solid #223049; border-radius:10px; padding:10px; font-size:14px;
    }
    .btn { background:#3b82f6; border:none; color:white; padding: 10px 14px; border-radius: 10px; cursor:pointer; font-weight: 650; }
    .btn:disabled { opacity:.6; cursor:not-allowed; }
    .muted { font-size: 12px; opacity: .75; margin-top: 10px; }
    audio { width:100%; margin-top: 12px; }
    .top { display:flex; align-items:center; justify-content:space-between; gap: 12px; margin-bottom: 12px; }
    .pill { font-size: 12px; padding: 6px 10px; border:1px solid #223049; border-radius: 999px; background:#0b1020; opacity:.9;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h1>MeloTTS RKNN Web Demo (RK3588)</h1>
      <div class="pill" id="status">idle</div>
    </div>
    <div class="card">
      <label>Text</label>
      <textarea id="text">Did you ever hear a folk tale about a giant turtle?</textarea>

      <div class="row">
        <div class="col">
          <label>Speaker</label>
          <select id="speaker"></select>
        </div>
        <div class="col">
          <label>Speed (1.0 = normal)</label>
          <input id="speed" type="range" min="0.6" max="1.4" step="0.05" value="1.0"/>
          <div class="muted"><span id="speedv">1.0</span></div>
        </div>
        <div class="col">
          <label>noise_scale</label>
          <input id="noise" type="range" min="0.2" max="1.2" step="0.05" value="0.667"/>
          <div class="muted"><span id="noisev">0.667</span></div>
        </div>
        <div class="col">
          <label>noise_scale_w</label>
          <input id="noisew" type="range" min="0.2" max="1.2" step="0.05" value="0.8"/>
          <div class="muted"><span id="noisewv">0.8</span></div>
        </div>
      </div>

      <div class="row" style="align-items:center;">
        <button class="btn" id="go">Generate</button>
        <div class="muted" id="msg"></div>
      </div>

      <audio id="player" controls></audio>
      <div class="row">
        <a id="download" class="btn" href="#" download="tts.wav" style="text-decoration:none; display:none;">Download WAV</a>
      </div>
    </div>
    <div class="muted">If you enabled bert.rknn, EN frontend will be faster. SDP+Flow remains CPU.</div>
  </div>

<script>
const el = (id)=>document.getElementById(id);
const statusPill = el("status");
const msg = el("msg");
const go = el("go");
const player = el("player");
const dl = el("download");

function setStatus(s){ statusPill.textContent = s; }

async function loadSpeakers(){
  const r = await fetch("/api/speakers");
  const j = await r.json();
  const spk = el("speaker");
  spk.innerHTML = "";
  const entries = Object.entries(j.speakers || {});
  if(entries.length === 0){
    const opt = document.createElement("option");
    opt.value = "0";
    opt.textContent = "0";
    spk.appendChild(opt);
    return;
  }
  for(const [name, id] of entries){
    const opt = document.createElement("option");
    opt.value = String(id);
    opt.textContent = `${name} (${id})`;
    spk.appendChild(opt);
  }
}

function bindRanges(){
  const speed = el("speed"), noise = el("noise"), noisew = el("noisew");
  const speedv = el("speedv"), noisev = el("noisev"), noisewv = el("noisewv");
  function upd(){
    speedv.textContent = speed.value;
    noisev.textContent = noise.value;
    noisewv.textContent = noisew.value;
  }
  speed.addEventListener("input", upd);
  noise.addEventListener("input", upd);
  noisew.addEventListener("input", upd);
  upd();
}

async function synth(){
  msg.textContent = "";
  dl.style.display = "none";
  player.removeAttribute("src");

  const payload = {
    text: el("text").value,
    speaker_id: parseInt(el("speaker").value || "0"),
    speed: parseFloat(el("speed").value),
    noise_scale: parseFloat(el("noise").value),
    noise_scale_w: parseFloat(el("noisew").value),
  };

  go.disabled = true;
  setStatus("generating...");
  const t0 = performance.now();
  try{
    const r = await fetch("/api/tts", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if(!r.ok){
      const e = await r.text();
      throw new Error(e || `HTTP ${r.status}`);
    }
    const getH = (k) => r.headers.get(k);
    const srvMs = getH("X-Gen-ms");
    const audioSec = getH("X-Audio-sec");
    const rtf = getH("X-RTF");
    const perf = getH("X-Perf");
    const segs = getH("X-Segments");

    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    player.src = url;
    dl.href = url;
    dl.style.display = "inline-block";

    const dt = (performance.now() - t0).toFixed(0);
    const srtf = rtf ? String(rtf) : "?";
    const saudio = audioSec ? String(audioSec) : "?";
    const ssrv = srvMs ? String(srvMs) : "?";
    const sperf = perf ? String(perf) : "";
    const sseg = segs ? String(segs) : "?";

    msg.textContent = `OK client=${dt}ms | server=${ssrv}ms | audio=${saudio}s | RTF=${srtf} | seg=${sseg}` + (sperf ? ` | ${sperf}` : "");
    setStatus(rtf ? `RTF ${srtf}` : "done");
  }catch(err){
    msg.textContent = String(err);
    setStatus("error");
  }finally{
    go.disabled = false;
  }
}

el("go").addEventListener("click", synth);
loadSpeakers();
bindRanges();
</script>
</body>
</html>
"""


class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = None
    speed: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8


def build_app(engine: MeloRKNNEngine) -> FastAPI:
    app = FastAPI(title="MeloTTS RKNN WebDemo")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/health")
    def health():
        return {"ok": True, "sr": engine.sr, "language": engine.language, "L": engine.L, "chunk_T": engine.chunk_T}

    @app.get("/api/speakers")
    def speakers():
        return {"speakers": engine.list_speakers()}


    @app.post("/api/tts")
    def tts(req: TTSRequest):
        try:
            t0 = time.perf_counter()
            audio, st = engine.synthesize_long_text(
                req.text,
                speaker_id=int(req.speaker_id) if req.speaker_id is not None else 0,
                speed=float(req.speed),
                noise_scale=float(req.noise_scale),
                noise_scale_w=float(req.noise_scale_w),
                silence_ms=60.0,
            )
            t1 = time.perf_counter()

            wav = wav_bytes_i16(audio, engine.sr)

            audio_sec = (float(audio.shape[0]) / float(engine.sr)) if engine.sr > 0 else 0.0
            server_ms = (t1 - t0) * 1000.0
            rtf = (server_ms / 1000.0) / audio_sec if audio_sec > 1e-6 else 0.0

            headers = {
                "X-Gen-ms": f"{server_ms:.2f}",
                "X-Audio-sec": f"{audio_sec:.3f}",
                "X-RTF": f"{rtf:.3f}",
                "X-Front-ms": f"{float(st.get('front_ms', 0.0)):.2f}",
                "X-Enc-ms": f"{float(st.get('enc_ms', 0.0)):.2f}",
                "X-CPU-ms": f"{float(st.get('cpu_ms', 0.0)):.2f}",
                "X-Dec-ms": f"{float(st.get('dec_ms', 0.0)):.2f}",
                "X-Segments": str(int(st.get('segments', 1))),
            }
            return Response(content=wav, media_type="audio/wav", headers=headers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.on_event("shutdown")
    def _shutdown():
        engine.shutdown()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-rknn", required=True)
    ap.add_argument("--dec-rknn", required=True)

    ap.add_argument("--bert-rknn", default=None)
    ap.add_argument("--bert-tokenizer", default="bert-base-uncased")
    ap.add_argument("--bert-seq", type=int, default=256)

    ap.add_argument("--language", default="EN")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--chunk-T", type=int, default=256)

    ap.add_argument("--idx-dtype", default="int32", choices=["int32", "int64"])
    ap.add_argument("--core-mask", default="012", choices=["012", "01", "0", "1", "2"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    engine = MeloRKNNEngine(
        enc_rknn_path=args.enc_rknn,
        dec_rknn_path=args.dec_rknn,
        bert_rknn_path=args.bert_rknn,
        bert_tokenizer=args.bert_tokenizer,
        bert_seq=args.bert_seq,
        language=args.language,
        L=args.L,
        chunk_T=args.chunk_T,
        idx_dtype=args.idx_dtype,
        core_mask=args.core_mask,
        seed=args.seed,
        torch_threads=args.torch_threads,
        warmup=args.warmup,
    )

    app = build_app(engine)

    import uvicorn

    _fix_broken_logging_levels()

    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
        log_config=_UVICORN_LOG_CONFIG_NUMERIC,
    )


if __name__ == "__main__":
    main()
