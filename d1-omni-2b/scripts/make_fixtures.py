"""Fixtures for the JVM tests HostParityTest and AudioTest: the model's own Python host (host/ of
litert-community/d1-Omni-2B-LiteRT) answers requests on the LiteRT graphs and records, for each request, the clips
(WAV files) and the picture it read, every clip's log-mel features and audio encoder output, the sequences it built
(with the order of each sequence's clips), every hidden state and tower output it read, and its answers. A request
the host refuses records the refusal. The WAV cases record what the model's own reader (`media.read_audio`) makes of
16-bit PCM files and of files it refuses.

    pip install ai-edge-litert numpy pillow tokenizers
    python -B scripts/make_fixtures.py --repo /path/to/d1-Omni-2B-LiteRT --out /tmp/d1-omni-fixtures [--accel gpu|cpu]

The repository folder needs provider/, tables/, host/, fixtures/media/, the per-length decision files
(d1-omni-2b_decide_L<L>_fp16fc.tflite) the requests need, the four audio files d1-omni-2b_audio_T<T>_fp16fc.tflite
and d1-omni-2b_tower_G24x24_fp16fc.tflite. -B keeps Python from writing bytecode next to the host.

The requests: the app's examples (app/src/main/assets/examples.json), test requests of the model repository (fixtures/requests_public.json: a
voicemail, a doorbell, a stereo 44.1 kHz clip, a text state with a clip and reads, two clips in one state, spoken
questions over a text and over a clip, 18 questions over a clip in two sequences, a JSON state with a 24 s clip), a
photo with a clip, and two clips made here at the audio files' limit: 480,240 samples (3,000 feature frames,
30.015 s, answered) and 480,241 samples (3,001 frames, refused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import wave

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "app", "src", "main", "assets")

# Test requests of the model repository, by record id (fixtures/requests_public.json), with this file's names.
HUB_REQUESTS = {
    "voicemail": "aud_voicemail",
    "doorbell": "sound_doorbell",
    "stereo44k": "sound_beeps_4_stereo44k",
    "ticket_call": "aud_text_ticket_call",
    "two_clips": "aud_two_clips",
    "ticket_spoken": "spk_ticket",
    "supply_18q": "spk_aud_supply_18q",
    "sensor_json": "aud_text_sensor",
}
BEEP_QUESTIONS = {
    "beeps": {"type": "number", "instructions": "How many beeps are there?", "range": [0, 10], "step": 1},
    "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
}


class Recorded:
    """A backend that keeps every input and output it sees, in call order."""

    def __init__(self, inner, keep_inputs: bool = False):
        self.inner = inner
        for name in ("max_len", "max_frames"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))
        self.keep_inputs = keep_inputs
        self.inputs, self.outputs, self.calls = [], [], []

    def __call__(self, *args):
        out = np.asarray(self.inner(*args), dtype=np.float32)
        if self.keep_inputs:
            self.inputs.append(np.asarray(args[0], dtype=np.float32).copy())
        if len(args) == 3:  # decide(embeds, state_len, question)
            self.calls.append({"tokens": int(np.asarray(args[0]).shape[0]), "state_len": int(args[1]),
                               "question": [int(x) for x in args[2]]})
        self.outputs.append(out.copy())
        return out

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()
        self.calls.clear()

    def close(self):
        self.inner.close()


def write_wav(path: str, pcm: np.ndarray, rate: int, channels: int = 1) -> None:
    with wave.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def edge_clip(samples: int, seed: int) -> np.ndarray:
    """Quiet noise with an 880 Hz beep every 4 s, 16 kHz, as int16 (the repository's test sounds' scaling)."""
    rate = 16000
    x = np.random.default_rng(seed).normal(0, 0.003, samples).astype(np.float32)
    for start in np.arange(1.0, samples / rate - 0.3, 4.0):
        at = int(start * rate)
        x[at:at + 3200] += 0.3 * np.sin(2 * np.pi * 880 * np.arange(min(3200, samples - at)) / rate)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def wav_bytes(fmt: int, channels: int, rate: int, bits: int, data: bytes, extra_chunks: bytes = b"",
              data_size: int | None = None, extensible: bool = False) -> bytes:
    """A RIFF / WAVE file of `data` with one fmt chunk (WAVE_FORMAT_EXTENSIBLE with the PCM sub-format when
    `extensible`), optional chunks before the data, and a data chunk header of `data_size` bytes (default: the data)."""
    align = channels * bits // 8
    if extensible:
        guid = struct.pack("<H", 1) + bytes([0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
                                              0x9B, 0x71])
        fmt_body = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * align, align, bits)
        fmt_body += struct.pack("<HHI", 22, bits, 4 if channels == 1 else 3) + guid
    else:
        fmt_body = struct.pack("<HHIIHH", fmt, channels, rate, rate * align, align, bits)
    size = len(data) if data_size is None else data_size
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body + extra_chunks
    body += b"data" + struct.pack("<I", size) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def wav_cases(out_dir: str, media_module) -> list[dict]:
    """WAV files the app may be given, and what the model's own reader makes of each (read_audio at 16 kHz)."""
    rng = np.random.default_rng(7)
    tone = (np.sin(2 * np.pi * 440 * np.arange(8000) / 16000) * 12000).astype("<i2")
    stereo = np.stack([tone, (tone // 2).astype("<i2")], 1).reshape(-1).astype("<i2")
    cases = {
        "pcm16_mono_16k": wav_bytes(1, 1, 16000, 16, tone.tobytes()),
        "pcm16_stereo_44k": wav_bytes(1, 2, 44100, 16, stereo.tobytes()),
        "pcm16_mono_8k_list_chunk": wav_bytes(1, 1, 8000, 16, tone[:4000].tobytes(),
                                              extra_chunks=b"LIST" + struct.pack("<I", 5) + b"INFOx\x00"),
        "pcm16_extensible": wav_bytes(1, 1, 16000, 16, tone.tobytes(), extensible=True),
        "pcm16_empty": wav_bytes(1, 1, 16000, 16, b""),
        "pcm8_mono": wav_bytes(1, 1, 16000, 8, (rng.integers(0, 256, 4000, dtype=np.uint8)).tobytes()),
        "pcm24_mono": wav_bytes(1, 1, 16000, 24, rng.integers(0, 256, 3 * 4000, dtype=np.uint8).tobytes()),
        "float32_mono": wav_bytes(3, 1, 16000, 32, (tone.astype(np.float32) / 32768).tobytes()),
        "not_riff": b"ID3\x04\x00\x00\x00\x00\x00\x00" + rng.integers(0, 256, 500, dtype=np.uint8).tobytes(),
        "truncated_odd": wav_bytes(1, 1, 16000, 16, tone[:250].tobytes() + b"\x01", data_size=1000),
    }
    out = []
    for name, data in cases.items():
        path = os.path.join(out_dir, f"{name}.wav")
        with open(path, "wb") as f:
            f.write(data)
        record = {"name": name, "file": f"{name}.wav", "sha256": hashlib.sha256(data).hexdigest()}
        try:
            wave_, rate = media_module.read_audio(path, 16000)
            arr = np.asarray(wave_, dtype=np.float32)
            np.save(os.path.join(out_dir, f"{name}.npy"), arr)
            record.update(ok=True, rate=int(rate), shape=list(arr.shape), samples_file=f"{name}.npy")
        except Exception as e:  # what the provider refuses, and how
            record.update(ok=False, error=type(e).__name__, message=str(e))
        out.append(record)
    return out


def requests(hub: dict, media: dict) -> list[dict]:
    """Every request as {id, state, questions, media: [{type, path}], refused?}: paths are names in the fixture's
    media folder."""
    with open(os.path.join(ASSETS, "examples.json")) as f:
        examples = json.load(f)["examples"]
    out = []
    for e in examples:
        items = [{"type": "audio", "path": c} for c in e.get("clips", [])]
        if e.get("picture"):
            items.insert(0, {"type": "image", "path": e["picture"]})
        out.append({"id": e["id"], "state": e["state"], "questions": e["questions"], "media": items})

    def file_of(mid: str) -> str:
        return os.path.basename(media[mid]["file"])

    for rid, hub_id in HUB_REQUESTS.items():
        req = json.loads(json.dumps(hub[hub_id]["request"]))
        for q in req["questions"].values():
            if isinstance(q.get("instructions"), dict):
                q["instructions"] = {"type": "audio", "path": file_of(q["instructions"]["media"])}
        items = [{"type": media[m]["kind"] if media[m]["kind"] == "audio" else "image", "path": file_of(m)}
                 for m in req.get("media", [])]
        out.append({"id": rid, "state": req["state"], "questions": req["questions"], "media": items,
                    "source": hub_id})
    out.append({
        "id": "photo_clip",
        "state": None,
        "questions": {"bicycle": {"type": "box", "instructions": "Where is the bicycle?"},
                      "doorbell": {"type": "noul", "instructions": "Is a doorbell ringing in the recording?"},
                      "snow": {"type": "noul", "instructions": "Is there snow on the ground?"}},
        "media": [{"type": "image", "path": "commons_img_bike_03.png"}, {"type": "audio", "path": "snd_doorbell.wav"}],
    })
    out.append({"id": "edge_3000", "state": None, "questions": BEEP_QUESTIONS,
                "media": [{"type": "audio", "path": "edge_480240.wav"}]})
    out.append({"id": "edge_3001", "state": None, "questions": BEEP_QUESTIONS,
                "media": [{"type": "audio", "path": "edge_480241.wav"}], "refused": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the downloaded litert-community/d1-Omni-2B-LiteRT folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--accel", default="gpu", choices=("gpu", "cpu"))
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(repo, "host"))
    from d1_host import D1Host
    from lrt_backend import LiteRTAudio, LiteRTDecide, LiteRTTower, audio_paths, graph_paths

    out_root = os.path.abspath(a.out)
    media_dir = os.path.join(out_root, "media")
    os.makedirs(media_dir, exist_ok=True)
    with open(os.path.join(repo, "fixtures", "requests_public.json")) as f:
        hub = {r["id"]: r for r in json.load(f)["records"]}
    with open(os.path.join(repo, "fixtures", "media.json")) as f:
        media = json.load(f)["media"]
    reqs = requests(hub, media)
    # The clips and pictures, copied next to the fixtures (the JVM test reads them with the app's own reader).
    names = {m["path"] for r in reqs for m in r["media"]}
    names |= {q["instructions"]["path"] for r in reqs for q in r["questions"].values()
              if isinstance(q.get("instructions"), dict)}
    for name in sorted(names):
        if name.startswith("edge_"):
            continue
        source = os.path.join(ASSETS, name)
        if not os.path.isfile(source):
            source = os.path.join(repo, "fixtures", "media", name)
        shutil.copyfile(source, os.path.join(media_dir, name))
    write_wav(os.path.join(media_dir, "edge_480240.wav"), edge_clip(480240, 3000), 16000)
    write_wav(os.path.join(media_dir, "edge_480241.wav"), edge_clip(480241, 3001), 16000)
    sha = {n: hashlib.sha256(open(os.path.join(media_dir, n), "rb").read()).hexdigest()
           for n in sorted(os.listdir(media_dir))}

    dec_paths, tow_paths = graph_paths(repo, "fp16fc")
    decide = Recorded(LiteRTDecide(dec_paths, a.accel, "fp32", 8, keep=1))
    tower = Recorded(LiteRTTower(tow_paths, a.accel, "fp32", 8))
    audio = Recorded(LiteRTAudio(audio_paths(repo, "fp16fc"), a.accel, "fp32", 8), keep_inputs=True)
    host = D1Host(os.path.join(repo, "provider"), decide, tower, embed_table=os.path.join(repo, "tables", "embed_table.npy"),
                  audio=audio)
    cases_dir = os.path.join(out_root, "wav_cases")
    os.makedirs(cases_dir, exist_ok=True)
    cases = wav_cases(cases_dir, host.p.media)
    with open(os.path.join(cases_dir, "cases.json"), "w") as f:
        json.dump({"cases": cases}, f, indent=1)
    print(json.dumps({"wav_cases": [(c["name"], c["ok"]) for c in cases]}), flush=True)

    def item(m: dict):
        path = os.path.join(media_dir, m["path"])
        return Image.open(path) if m["type"] == "image" else {"type": "audio", "path": path}

    index = []
    for req in reqs:
        rid = req["id"]
        own = os.path.join(out_root, rid)
        os.makedirs(own, exist_ok=True)
        questions = json.loads(json.dumps(req["questions"]))
        for q in questions.values():
            if isinstance(q.get("instructions"), dict):
                q["instructions"] = {"type": "audio", "path": os.path.join(media_dir, q["instructions"]["path"])}
        items = [item(m) for m in req["media"]]
        picture = next((m["path"] for m in req["media"] if m["type"] == "image"), None)
        if picture:
            np.save(os.path.join(own, "picture.npy"),
                    np.asarray(Image.open(os.path.join(media_dir, picture)).convert("RGB"), dtype=np.uint8))
        for backend in (decide, tower, audio):
            backend.clear()
        t0 = time.time()
        refusal = None
        try:
            response = host.system_one(req["state"], questions, media=items or None)
        except ValueError as e:
            response, refusal = None, str(e)
        wall = time.time() - t0
        if req.get("refused") and refusal is None:
            raise SystemExit(f"{rid}: the host answered a request it should refuse")
        if refusal is not None and not req.get("refused"):
            raise SystemExit(f"{rid}: the host refused: {refusal}")
        record = {"id": rid, "source": req.get("source"),
                  "request": {"state": req["state"], "questions": req["questions"], "media": req["media"]},
                  "media_sha256": {m["path"]: sha[m["path"]] for m in req["media"]},
                  "accel": a.accel, "wall_s": wall}
        if refusal is not None:
            record.update(refused=True, message=refusal,
                          backend_calls={"decide": len(decide.outputs), "tower": len(tower.outputs),
                                         "audio": len(audio.outputs)})
        else:
            seqs = host.last["seqs"]
            calls = host.last["audio_features"]
            for n, (x, y) in enumerate(zip(audio.inputs, audio.outputs)):
                np.save(os.path.join(own, f"audio_in_{n}.npy"), x)
                np.save(os.path.join(own, f"audio_out_{n}.npy"), y)
            seq_audio = [[next(i for i, c in enumerate(calls) if c is o) for o in outs] for outs in host.last["seq_audio"]]
            for n, h in enumerate(decide.outputs):
                np.save(os.path.join(own, f"hidden_{n}.npy"), h)
            if tower.outputs:
                np.save(os.path.join(own, "tower.npy"), tower.outputs[0])
            pixel_values = next((s["pixel_values"] for s in seqs if s.get("pixel_values") is not None), None)
            if pixel_values is not None:
                np.save(os.path.join(own, "pixel_values.npy"), np.asarray(pixel_values, np.float32))
            record.update(
                refused=False,
                response=response,
                sequences=[{"first_question": s["first_question"], "ids": s["ids"], "state_len": s["state_len"],
                            "question": s["question"], "ends": s["ends"], "image_grid_thw": s["image_grid_thw"],
                            "clips": s["clips"], "audio_calls": seq_audio[n]} for n, s in enumerate(seqs)],
                audio_calls=[{"frames": int(x.shape[1]), "tokens": int(y.shape[0]), "in": f"audio_in_{n}.npy",
                              "out": f"audio_out_{n}.npy"} for n, (x, y) in enumerate(zip(audio.inputs, audio.outputs))],
                decide_calls=[dict(c, file=f"hidden_{n}.npy") for n, c in enumerate(decide.calls)],
                tower_file="tower.npy" if tower.outputs else None,
                picture_file="picture.npy" if picture else None,
                pixel_values_file="pixel_values.npy" if pixel_values is not None else None,
                timing=host.last["timing"],
            )
        with open(os.path.join(own, "expected.json"), "w") as f:
            json.dump(record, f, ensure_ascii=False)
        index.append({"id": rid, "refused": refusal is not None,
                      "tokens": None if response is None else response["usage"]["input_tokens"],
                      "sequences": None if response is None else len(host.last["seqs"]),
                      "clips": len(audio.outputs), "wall_s": round(wall, 2)})
        print(json.dumps(index[-1]), flush=True)
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump({"requests": index, "accel": a.accel, "media_sha256": sha}, f, indent=1)
    for backend in (decide, tower, audio):
        backend.close()


if __name__ == "__main__":
    main()
