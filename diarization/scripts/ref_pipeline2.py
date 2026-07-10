#!/usr/bin/env python3
"""Pipeline check on a clearly-two-voice construction (LibriSpeech male + Kokoro TTS female),
with short-clip clustering params (min_cluster_size small)."""
import os
import torch
import torchaudio
import torchaudio.functional as AF

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
torchaudio.AudioMetaData = getattr(torchaudio, "AudioMetaData", type("AudioMetaData", (), {}))
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization

male, sr_m = torchaudio.load(os.path.expanduser("~/Downloads/meeting/wav2vec2-work/sample_speech.wav"))
female, sr_f = torchaudio.load(os.path.expanduser("~/Downloads/meeting/A_true_kokoro.wav"))
male = AF.resample(male.mean(0, keepdim=True), sr_m, SR)
female = AF.resample(female.mean(0, keepdim=True), sr_f, SR)
male = male / male.abs().max() * 0.7
female = female / female.abs().max() * 0.7
gap = torch.zeros(1, int(0.4 * SR))
clip = torch.cat([male[:, :5 * SR], gap, female[:, :5 * SR], gap, male[:, 5 * SR:9 * SR]], dim=1)
torchaudio.save(os.path.join(HERE, "case_mf.wav"), clip, SR)
print(f"clip {clip.shape[1]/SR:.1f}s — truth: male 0-5, female 5.4-10.4, male 10.8-14.8")

seg = Model.from_pretrained(os.path.join(HERE, "seg30", "pytorch_model.bin")).eval()
emb = Model.from_pretrained(os.path.join(HERE, "wespeaker", "pytorch_model.bin")).eval()

for mcs, thr in [(12, 0.7045654963945799), (2, 0.7045654963945799), (2, 0.6)]:
    pipe = SpeakerDiarization(segmentation=seg, embedding=emb,
                              embedding_exclude_overlap=True,
                              clustering="AgglomerativeClustering")
    pipe.instantiate({
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"method": "centroid", "min_cluster_size": mcs, "threshold": thr},
    })
    out = pipe({"waveform": clip, "sample_rate": SR})
    print(f"-- min_cluster_size={mcs} threshold={thr:.2f}")
    for turn, _, spk in out.itertracks(yield_label=True):
        print(f"   {turn.start:5.1f} - {turn.end:5.1f}  {spk}")
