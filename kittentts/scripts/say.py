"""KittenTTS nano on LiteRT — drop-in text-to-speech module (Piper replacement).

    from say import KittenTTS
    tts = KittenTTS(models_dir=".", voices_path="voices.npz")
    pcm = tts.say("Hello! How can I help you today?")        # float32 @ 24 kHz
    for sentence, pcm in tts.stream(long_text):              # sentence-streaming
        play(pcm)

CLI:
    python say.py "Hello there." -o hello.wav [--voice Jasper] [--speed 1.0]
                  [--precision fp16] [--threads 4] [--g2p auto|lib|cli]

Runtime deps: numpy + ai-edge-litert, plus ONE of:
  - phonemizer (+ espeakng-loader or a system espeak-ng library)  [--g2p lib]
    exact match to the reference tokenization; espeak-ng is GPL-3.0 and is
    loaded in-process (the same situation as Piper's frontend), or
  - the espeak-ng command-line binary                              [--g2p cli]
    GPL isolation via subprocess; tokenization is equivalent in practice but
    not guaranteed byte-identical to the phonemizer library.

The fused-LSTM graphs keep hidden state in variable tensors across invoke();
this module resets them before every utterance (required — reusing the
interpreter without a reset corrupts the second synthesis).
"""
import argparse
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

SR = 24000

# StyleTTS2 / Kokoro 178-symbol table (matches the official KittenTTS package)
_PAD = "$"
_PUNCT = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_IPA = ("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
SYMBOL_INDEX = {s: i for i, s in enumerate([_PAD] + list(_PUNCT) + list(_LETTERS) + list(_IPA))}

VOICE_ALIASES = {"Bella": "expr-voice-2-f", "Jasper": "expr-voice-2-m",
                 "Luna": "expr-voice-3-f", "Bruno": "expr-voice-3-m",
                 "Rosie": "expr-voice-4-f", "Hugo": "expr-voice-4-m",
                 "Kiki": "expr-voice-5-f", "Leo": "expr-voice-5-m"}
SPEED_PRIORS = {"expr-voice-2-f": 0.8, "expr-voice-2-m": 0.8, "expr-voice-3-m": 0.8,
                "expr-voice-3-f": 0.8, "expr-voice-4-m": 0.9, "expr-voice-4-f": 0.8,
                "expr-voice-5-m": 0.8, "expr-voice-5-f": 0.8}


# ------------------------------------------------------------------ frontend
class Phonemizer:
    """espeak-ng IPA via the phonemizer library (exact) or the CLI (isolated)."""

    def __init__(self, mode="auto"):
        self.backend = None
        if mode in ("auto", "lib"):
            try:
                import phonemizer
                try:
                    import espeakng_loader
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                    EspeakWrapper.set_library(espeakng_loader.get_library_path())
                    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                except ImportError:
                    pass  # fall back to a system espeak-ng library
                self.backend = phonemizer.backend.EspeakBackend(
                    language="en-us", preserve_punctuation=True, with_stress=True)
            except Exception:
                if mode == "lib":
                    raise
        if self.backend is None:
            if not shutil.which("espeak-ng"):
                raise RuntimeError(
                    "no G2P available: pip install phonemizer espeakng-loader, "
                    "or install the espeak-ng binary")

    def __call__(self, text):
        if self.backend is not None:
            return self.backend.phonemize([text])[0]
        # CLI path: phonemize the runs between punctuation, re-insert punctuation
        parts = re.split(r"([;:,.!?¡¿—…\"«»“”]+)", text)
        out = []
        for i, part in enumerate(parts):
            if i % 2:  # punctuation
                out.append(part)
                continue
            if not part.strip():
                continue
            ipa = subprocess.run(
                ["espeak-ng", "-q", "--ipa", "-v", "en-us", part.strip()],
                capture_output=True, text=True, check=True).stdout
            out.append(" ".join(ipa.split()))
        return " ".join(o for o in out if o)


def tokenize(phonemes):
    """Match the official package: re-space the phoneme string, map to ids."""
    spaced = " ".join(re.findall(r"\w+|[^\w\s]", phonemes))
    ids = [SYMBOL_INDEX[c] for c in spaced if c in SYMBOL_INDEX]
    return np.array([[0] + ids + [0]], dtype=np.int32)


def split_sentences(text, max_len=400):
    chunks = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_len:
            cut = sentence.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        chunks.append(sentence if sentence[-1] in ".!?,;:" else sentence + ",")
    return chunks


# ------------------------------------------------------------------- engine
class _Graph:
    def __init__(self, path, threads):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()

    @staticmethod
    def _canon(name):
        name = name.split(":")[0]
        return name[len("serving_default_"):] if name.startswith("serving_default_") else name

    def __call__(self, feeds):
        for d in self.inputs:
            self.it.resize_tensor_input(d["index"], list(feeds[self._canon(d["name"])].shape))
        self.it.allocate_tensors()
        self.it.reset_all_variables()  # fused-LSTM state persists otherwise
        for d in self.inputs:
            self.it.set_tensor(d["index"], feeds[self._canon(d["name"])])
        self.it.invoke()
        return [self.it.get_tensor(o["index"]) for o in self.outputs]


class KittenTTS:
    def __init__(self, models_dir=".", voices_path=None, voice="expr-voice-2-m",
                 precision="fp32", threads=4, g2p="auto"):
        models_dir = Path(models_dir)
        suffix = "" if precision == "fp32" else "_fp16"
        self.predictor = _Graph(models_dir / f"kitten_predictor{suffix}.tflite", threads)
        self.prosody = _Graph(models_dir / f"kitten_prosody{suffix}.tflite", threads)
        self.vocoder = _Graph(models_dir / f"kitten_vocoder{suffix}.tflite", threads)
        self.voices = np.load(voices_path or models_dir / "voices.npz")
        self.voice = VOICE_ALIASES.get(voice, voice)
        self.phonemize = Phonemizer(g2p)

    def _synthesize(self, sentence, speed):
        ids = tokenize(self.phonemize(sentence))
        ref_id = min(len(sentence), self.voices[self.voice].shape[0] - 1)
        style = self.voices[self.voice][ref_id:ref_id + 1].astype(np.float32)
        eff_speed = np.array([speed * SPEED_PRIORS.get(self.voice, 1.0)], dtype=np.float32)

        p_out = self.predictor({"input_ids": ids, "style": style, "speed": eff_speed})
        d = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 256][0]
        t_en = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 128][0]
        dur = [o for o in p_out if o.dtype in (np.int32, np.int64)][0]
        en = np.repeat(d[0], dur, axis=0)[None].astype(np.float32)
        asr = np.repeat(t_en[0], dur, axis=0)[None].astype(np.float32)
        pr = self.prosody({"en": en, "style": style})
        har = [o for o in pr if o.ndim == 3][0]
        f0, n = [o for o in pr if o.ndim == 2]
        v = self.vocoder({"asr": asr, "f0": f0, "n": n, "har": har, "style": style})
        return v[0][0]

    def stream(self, text, speed=1.0):
        """Yield (sentence, float32 PCM @ 24 kHz) as each sentence is ready."""
        for sentence in split_sentences(text):
            yield sentence, self._synthesize(sentence, speed)

    def say(self, text, speed=1.0):
        pieces = [pcm for _, pcm in self.stream(text, speed)]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="KittenTTS nano on LiteRT")
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="say.wav")
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parent.parent / "out"))
    ap.add_argument("--voices", default=None)
    ap.add_argument("--voice", default="expr-voice-2-m")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--g2p", choices=["auto", "lib", "cli"], default="auto")
    args = ap.parse_args()

    tts = KittenTTS(args.models_dir, args.voices, args.voice,
                    args.precision, args.threads, args.g2p)
    t0 = time.perf_counter()
    pieces = []
    for sentence, pcm in tts.stream(args.text, args.speed):
        print(f"  [{time.perf_counter()-t0:6.2f}s] {len(pcm)/SR:5.2f}s  {sentence[:60]}")
        pieces.append(pcm)
    wav = np.concatenate(pieces)
    dt = time.perf_counter() - t0
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"wrote {args.output}: {len(wav)/SR:.2f}s audio in {dt:.2f}s (RTF {dt/(len(wav)/SR):.3f})")


if __name__ == "__main__":
    main()
