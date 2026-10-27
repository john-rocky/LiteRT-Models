"""Host-side mirror of the Ming-3.0-tts-0.5B pipeline (numpy + regex only) = the device host spec.

Ports, from the upstream inference code (GitHub inclusionAI/Ming-3.0-tts @028c1eace193, MIT,
Copyright (c) Ant Group): tts_tasks.build_task_generation_kwargs / build_structured_instruction,
prompt_contract.build_input_ids, the Qwen2 byte-level BPE of tokenizer.json, transformers' Qwen2
RoPE (via the stored inv_freq), CFM's EPSS + sway times, the decode loop's stop rule and latent
history, and audio_tokenizer/istft.py (ISTFTHead mag/phase + "same" ISTFT, overlap-add).
Everything runs in float32 except where a comment says otherwise.
"""
import json
import unicodedata
from collections import OrderedDict

import numpy as np
import regex

import common as C

# ------------------------------------------------------------------ tokenizer


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class QwenBPE:
    """tokenizer.json: added tokens split first, NFC, the Split regex, ByteLevel, BPE merges."""

    def __init__(self, path=C.HF_DIR / "tokenizer.json"):
        t = json.load(open(path, encoding="utf-8"))
        assert t["normalizer"]["type"] == "NFC"
        pre = t["pre_tokenizer"]["pretokenizers"]
        assert pre[0]["type"] == "Split" and pre[0]["behavior"] == "Isolated"
        assert pre[1]["type"] == "ByteLevel" and not pre[1]["use_regex"]
        self.split = regex.compile(pre[0]["pattern"]["Regex"])
        m = t["model"]
        assert m["type"] == "BPE" and not m["byte_fallback"]
        self.vocab = m["vocab"]
        self.ranks = {tuple(p) if isinstance(p, list) else tuple(p.split(" ")): i
                      for i, p in enumerate(m["merges"])}
        self.added = {a["content"]: a["id"] for a in t["added_tokens"]}
        alts = sorted(self.added, key=len, reverse=True)
        self.added_re = regex.compile("|".join(regex.escape(a) for a in alts))
        self.b2u = _bytes_to_unicode()
        self.cache = {}

    def _bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts = parts[:bi] + [parts[bi] + parts[bi + 1]] + parts[bi + 2:]
        ids = [self.vocab[p] for p in parts]
        self.cache[word] = ids
        return ids

    def _encode_plain(self, text):
        text = unicodedata.normalize("NFC", text)
        out = []
        for piece in self.split.findall(text):
            word = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            out.extend(self._bpe(word))
        return out

    def encode(self, text):
        out, pos = [], 0
        for m in self.added_re.finditer(text):
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.added[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def token_id(self, token):
        return self.added[token] if token in self.added else self.vocab[token]


# ------------------------------------------------------------ prompt contract

DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"
LANG_NAMES = {"zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese", "en": "English",
              "english": "English", "ja": "Japanese", "japanese": "Japanese", "ko": "Korean",
              "korean": "Korean", "de": "German", "german": "German", "fr": "French",
              "french": "French", "es": "Spanish", "spanish": "Spanish", "yue": "Chinese",
              "zh-yue": "Chinese", "cantonese": "Chinese"}
ZH_FIELDS = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
BGM_FIELDS = ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")


def tts_prompt(language_id):
    if language_id is None or not str(language_id).strip():
        return DEFAULT_TTS_PROMPT
    lang = str(language_id).strip()
    return DEFAULT_TTS_PROMPT + f"Target language: {LANG_NAMES.get(lang.lower(), lang)}.\n"


def ip_instruction(ip):
    """tts_tasks.build_structured_instruction(schema='zh', ip=...) (IP task default schema)."""
    item = OrderedDict(zip(ZH_FIELDS, (1, "speaker_1", None, None, None, None, None, None)))
    item["BGM"] = OrderedDict((k, None) for k in BGM_FIELDS)
    item["IP"] = ip
    return json.dumps(OrderedDict((("audio_sequence", [item]),)), ensure_ascii=False)


def build_input_ids(tok, prompt, text, num_speakers, instruction=None, prompt_text=None,
                    prompt_latent_steps=0):
    """prompt_contract.build_input_ids (dense): (ids, speaker pad positions, latent start)."""
    enc = tok.encode
    speaker = []
    for i in range(num_speakers):
        speaker += enc(f"  speaker_{i + 1}:") + enc("<|vision_start|>") + enc("<|vision_pad|>") + \
            enc("<|vision_end|>\n")
    instr = enc(instruction) + enc("<|im_end|>") if instruction is not None else []
    ptext, lat = [], []
    if prompt_latent_steps > 0 and prompt_text is not None:
        ptext = enc(prompt_text)
        lat = enc("<audioPatch>") * prompt_latent_steps
    bgm = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")
    text_prefix = [] if all(m in text for m in bgm) else enc(" Text input:\n")
    head = enc("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") + \
        enc("<|im_start|>user\n") + enc(prompt)
    turn_end = enc("<|im_end|>\n") + enc("<|im_start|>assistant\n")
    ids = head + speaker + text_prefix + ptext + enc(text) + turn_end + instr + enc("<audio>")
    start_id = tok.token_id("<|vision_start|>")
    spk_pos = [i + 1 for i, t in enumerate(ids) if t == start_id]
    assert len(spk_pos) == num_speakers
    lat_pos = len(ids) if lat else -1
    return ids + lat, spk_pos, lat_pos


def fixture_request(fx, voice_meta=None):
    """The fields synthesize() hands to generate(), for one fixture row (raw_text=True)."""
    text = str(fx["text"]).strip()
    req = {"prompt": tts_prompt(fx["language_id"]), "text": text, "instruction": None,
           "num_speakers": 1, "zero_speaker": False, "prompt_text": None}
    if "ip" in fx:
        req["instruction"] = ip_instruction(fx["ip"])
        req["zero_speaker"] = True        # use_zero_spk_emb=True
    else:
        req["prompt_text"] = voice_meta["prompt_text"]
    return req


class EmbedTable:
    """LM input embedding rows (host lookup), float32 or float16 storage."""

    def __init__(self, dtype=np.float32):
        # The checkpoint stores BF16; safetensors' numpy reader has no bf16, so read it via torch.
        from safetensors import safe_open
        with safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt") as f:
            w = f.get_tensor("model.model.embed_tokens.weight").float()
        self.table = w.numpy().astype(dtype)

    def rows(self, ids):
        return self.table[np.asarray(ids)].astype(np.float32)


def prompt_embeddings(table, ids, spk_pos, lat_pos, speaker_row=None, latent_rows=None):
    """prepare_input_embed: token rows, speaker pad rows, prompt-latent rows."""
    e = table.rows(ids)
    for p in spk_pos:
        e[p] = np.zeros(e.shape[1], np.float32) if speaker_row is None else speaker_row
    if latent_rows is not None:
        e[lat_pos:lat_pos + latent_rows.shape[0]] = latent_rows
    return e


# ------------------------------------------------------------------ RoPE

# transformers Qwen2RotaryEmbedding inv_freq (rope_theta 1e6, dim 64), float32 bits as torch
# computes them; the host does not recompute the power (one ulp here is 2e-5 at position 511).
QWEN_INV_FREQ = None


def qwen_inv_freq():
    global QWEN_INV_FREQ
    if QWEN_INV_FREQ is None:
        QWEN_INV_FREQ = np.array(json.load(open(C.K / "contract.json"))["constants"]["qwen_inv_freq"],
                                 np.float32)
    return QWEN_INV_FREQ


def qwen_rope(positions):
    """cos, sin [n, 64] float32: angle = float32(pos) * inv_freq (float32), cos/sin in float64."""
    inv = qwen_inv_freq()
    ang = np.asarray(positions, np.float32)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1).astype(np.float64)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


# ------------------------------------------------------------------ masks

MASK_NEG = np.float32(-30000.0)


def lm_mask(start, nq, past_slots=511):
    """Additive mask [1, 1, nq, past_slots + nq] for nq new tokens at positions start..start+nq-1.

    Past slot s (position s) is valid for s < start; new key j is allowed for query i iff j <= i.
    """
    m = np.full((nq, past_slots + nq), MASK_NEG, np.float32)
    m[:, :start] = 0.0
    for i in range(nq):
        m[i, past_slots:past_slots + i + 1] = 0.0
    return m[None, None]


def dec_mask(p0, nq, keys=64, past=64):
    """Decoder chunk mask [1, 1, nq, past + nq]: query at absolute p0+i sees absolute j with
    p0+i-keys < j <= p0+i (keys=64: transformers 4.46.3 sdpa/eager; keys=65: flash-attn)."""
    m = np.full((nq, past + nq), MASK_NEG, np.float32)
    for i in range(nq):
        q = p0 + i
        for s in range(past):
            j = p0 - past + s
            if j >= 0 and q - keys < j <= q:
                m[i, s] = 0.0
        for s in range(nq):
            j = p0 + s
            if q - keys < j <= q:
                m[i, past + s] = 0.0
    return m[None, None]


# ------------------------------------------------------------------ flow schedule


def epss_times(steps=10):
    """CFM.sample's t (get_epss_timesteps + sway -1) as torch float32 computes it, read from the
    contract: torch's float32 cos at t = 4/32 is not the correctly rounded value, so a portable
    recompute (epss_times_recomputed) lands 1 ulp off there. The device never needs t at run
    time: the time embeddings (temb asset) and the Euler dt (baked) come from torch."""
    assert steps == 10, "contract stores the 10-step schedule; other step counts ship their own"
    return np.array(load_contract()["constants"]["epss_t"], np.float32)


def epss_times_recomputed(steps=10):
    """Portable float32 recompute (cos rounded from float64), for the record only."""
    table = {5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
             10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]}
    t = np.float32(1 / 32) * np.array(table[steps], np.float32)
    c = np.cos((np.float32(np.pi / 2) * t).astype(np.float64)).astype(np.float32)
    return (t + np.float32(-1.0) * (c - np.float32(1) + t)).astype(np.float32)


class StopHead:
    """stop_head (Linear 896 -> 2) on the host, float32: logits = W z + b."""

    def __init__(self):
        from safetensors import safe_open
        with safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt") as f:
            self.w = f.get_tensor("stop_head.weight").float().numpy()
            self.b = f.get_tensor("stop_head.bias").float().numpy()

    def __call__(self, z):
        z = np.asarray(z, np.float32).reshape(-1)
        return (self.w.astype(np.float64) @ z.astype(np.float64)).astype(np.float32) + self.b


def stop_decision(logits, step):
    """softmax(stop_logits)[1] > 0.5 and step > 3 (float32 softmax)."""
    l = np.asarray(logits, np.float32)
    e = np.exp(l - l.max())
    p1 = np.float32(e[1] / e.sum())
    return bool(p1 > 0.5 and step > 3), float(p1)


# ------------------------------------------------------------------ decoder windows + ISTFT


def dec_latent_window(lat, a, chunk):
    """[chunk+2, 64]: lat[a-1] | lat[a:a+chunk] | lat[a+chunk], edges replicated, tail padded."""
    S = lat.shape[0]
    idx = [max(a - 1, 0)] + [min(i, S - 1) for i in range(a, a + chunk)] + [min(a + chunk, S - 1)]
    return lat[idx].astype(np.float32)


class Istft:
    """ISTFTHead (mag = clip(exp(.), 1e2), phase) + ISTFT(padding="same"), whole utterance or
    streamed frame by frame with an overlap buffer (exact against the whole-utterance fold)."""

    def __init__(self, window, n_fft=3528, hop=882):
        self.w = np.asarray(window, np.float32)
        self.n, self.hop = n_fft, hop
        self.pad = (n_fft - hop) // 2

    def frames(self, head):
        """head [F, 3530] linear output -> windowed irfft frames [F, 3528] (float64 FFT, f32 out)."""
        h = np.asarray(head, np.float32)
        half = h.shape[1] // 2
        mag = np.minimum(np.exp(h[:, :half]), np.float32(1e2)).astype(np.float32)
        p = h[:, half:]
        re = (mag * np.cos(p)).astype(np.float32)
        im = (mag * np.sin(p)).astype(np.float32)
        spec = re.astype(np.float64) + 1j * im.astype(np.float64)
        x = np.fft.irfft(spec, n=self.n, axis=1).astype(np.float32)
        return x * self.w[None, :]

    def overlap_add(self, frames):
        F = frames.shape[0]
        size = (F - 1) * self.hop + self.n
        y = np.zeros(size, np.float32)
        env = np.zeros(size, np.float32)
        w2 = self.w * self.w
        for f in range(F):
            y[f * self.hop:f * self.hop + self.n] += frames[f]
            env[f * self.hop:f * self.hop + self.n] += w2
        y = y[self.pad:-self.pad]
        env = env[self.pad:-self.pad]
        assert (env > 1e-11).all()
        return y / env

    def __call__(self, head):
        return self.overlap_add(self.frames(head))


class IstftStream:
    """Streaming overlap-add of Istft: push windowed frames as they come, get final samples out.
    Equals Istft.overlap_add on the whole utterance (same float32 adds in the same order)."""

    def __init__(self, ist):
        self.ist = ist
        self.n, self.hop, self.pad = ist.n, ist.hop, ist.pad
        self.y = np.zeros(0, np.float32)
        self.env = np.zeros(0, np.float32)
        self.w2 = ist.w * ist.w
        self.f = 0              # frames pushed
        self.done = 0           # fold-coordinate samples already emitted (incl. the trimmed pad)
        self.out = []

    def push(self, frames):
        for fr in frames:
            need = self.f * self.hop + self.n - self.done
            if self.y.size < need:
                self.y = np.concatenate([self.y, np.zeros(need - self.y.size, np.float32)])
                self.env = np.concatenate([self.env, np.zeros(need - self.env.size, np.float32)])
            o = self.f * self.hop - self.done
            self.y[o:o + self.n] += fr
            self.env[o:o + self.n] += self.w2
            self.f += 1
        # samples before the next frame's start are final
        self._emit(self.f * self.hop)

    def _emit(self, upto):
        k = upto - self.done
        if k <= 0:
            return
        y, e = self.y[:k], self.env[:k]
        lo = max(0, self.pad - self.done)
        if lo < k:
            self.out.append(y[lo:] / e[lo:])
        self.y, self.env = self.y[k:], self.env[k:]
        self.done = upto

    def finish(self):
        total = (self.f - 1) * self.hop + self.n
        self._emit(total - self.pad)
        return np.concatenate(self.out) if self.out else np.zeros(0, np.float32)


def load_contract():
    return json.load(open(C.K / "contract.json"))
