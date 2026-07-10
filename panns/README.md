# PANNs CNN14 — On-Device Audio Tagging (AudioSet 527 classes)

[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) **CNN14** (`Cnn14_mAP=0.431`) general sound-event tagging on Android, with the CNN running **fully on the LiteRT `CompiledModel` GPU (ML Drift)**. Given ~10 s of audio it predicts probabilities over the **527 AudioSet classes** — speech, music, instruments, animals, vehicles, alarms, household sounds, and so on. AudioSet tagging is **multi-label**: several tags can be high at once.

```
waveform[320000] --[Kotlin log-mel]--> logmel[1,1,1001,64] --[GPU CNN14]--> probs[527] (sigmoid)
```

## Why the log-mel is host-side

PANNs builds its spectrogram with **torchlibrosa**, whose STFT is a *DFT-as-Conv1d* — so there is **no FFT op** and the entire raw-audio→tags graph is almost GPU-clean. The only GPU blocker is the STFT centering **reflect-pad** (one `GATHER_ND`), removable by switching `pad_mode='reflect'→'constant'` (zero-pad), which is corr 1.0 (only the first/last frame differ).

**But** the converted spectral front-end is unusable: litert-torch lowers the giant 1024-tap DFT-conv incorrectly (fp32 tflite corr ≈ 0.19), and the power spectrum `|STFT|²` reaches ~1e6, which **overflows fp16 on Mali → NaN**. So the spectral front-end is computed on the CPU in Kotlin (the Whisper/Kokoro pattern), matched to torchlibrosa exactly, and only the CNN body rides the GPU. The CNN body (`bn0` + 6 conv blocks + pooling + 2 FC + sigmoid) is a pure CNN and converts at **corr 1.000000 in both fp32 and fp16**.

## Model

| Stage | File | Size | Input → Output | Placement |
| ----- | ---- | ---- | -------------- | --------- |
| log-mel | (Kotlin `MelSpectrogram.kt`) | — | waveform [320000] → logmel [1,1,1001,64] | CPU |
| CNN14 | `cnn14_audioset_fp16.tflite` | 162 MB FP16 | logmel [1,1,1001,64] → probs [1,527] | CompiledModel GPU |

Desktop validation: host log-mel vs torchlibrosa corr 1.000000 (max\|d\| 0.0017); fp16 CNN tflite vs PyTorch corr 1.000000; op-check banned NONE, >4D 0 (GPU-clean).

**On-device (Pixel 8a, Tensor G3 — verified):** CNN body **45/45** nodes on the LiteRT GPU delegate (`LITERT_CL`), 1 partition (single graph, no CPU fallback); ~124 ms GPU + ~99 ms host log-mel ≈ 0.22 s per 10 s clip; bundled-clip self-test → top tag "Speech".

## Audio front-end (must match torchlibrosa exactly)

`Spectrogram(n_fft=1024, hop=320, win='hann', center=True, pad_mode='reflect', power=2)` + `LogmelFilterBank(sr=32000, n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None)`, i.e. `log_mel = 10·log10(max(mel_power, 1e-10))`. The mel basis (`librosa.filters.mel`, slaney norm) is exported to `assets/mel_basis.bin` [64, 513]; the periodic Hann window and a radix-2 FFT are in Kotlin. 10 s @ 32 kHz → 1001 frames.

## Build & run

The 162 MB tflite is **not bundled** — it is pushed into the app's private `filesDir`.

```bash
cd scripts/
# get the model defs (models.py -> panns_models.py, pytorch_utils.py) from
#   github.com/qiuqiangkong/audioset_tagging_cnn
# and the checkpoint Cnn14_mAP=0.431.pth from
#   https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth   (CC-BY-4.0)
~/clipconv/bin/python build_panns.py          # -> cnn14_audioset_fp16.tflite + mel_basis.bin + fixtures
./install_to_device.sh                        # adb push the tflite into com.panns/files/
cd .. && ./gradlew :app:installDebug
```

On launch the app self-tests on a bundled clip (synthesized speech → top tag "Speech"), then the **Record 10 s & tag** button captures from the mic and lists the top tags with a bar chart.

## Original project & license

[qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) — code [Apache-2.0](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE), weights `Cnn14_mAP=0.431.pth` on [Zenodo](https://zenodo.org/record/3987831) [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). AudioSet ontology © Google, [CC-BY-4.0](https://research.google.com/audioset/).
