import numpy as np, torch, whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES
from ai_edge_litert.interpreter import Interpreter, OpResolverType

PD = '/Users/majimadaisuke/Downloads/meeting/npubench-phase9-probes'
model = whisper.load_model('tiny', device='cpu'); model.eval()

def tflite_feats(path, mel):
    it = Interpreter(model_path=path,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    want = it.get_input_details()[0]['shape'][-1]
    x = np.zeros((1, 80, want), np.float32); x[:, :, :mel.shape[-1]] = mel
    it.set_tensor(it.get_input_details()[0]['index'], x); it.invoke()
    return torch.from_numpy(it.get_tensor(it.get_output_details()[0]['index']))

class Fixed(torch.nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, mel): return self.f

opts = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)
for wav in ['/Users/majimadaisuke/Downloads/depthanything-android/voice_demo/out/asr_test_24k.wav',
            '/Users/majimadaisuke/Downloads/depthanything-android/voice_demo/out/demo_text_kitten.wav']:
    audio = whisper.load_audio(wav)
    mel = log_mel_spectrogram(pad_or_trim(audio, N_SAMPLES))[None].numpy().astype(np.float32)
    real_enc = model.encoder
    with torch.no_grad():
        texts = {}
        texts['torch'] = whisper.decode(model, torch.from_numpy(mel)[0], opts).text
        for tag, p in [('ctrl', f'{PD}/probe_wh_ctrl.tflite'), ('pad1536', f'{PD}/probe_wh_pad1536.tflite')]:
            model.encoder = Fixed(tflite_feats(p, mel))
            texts[tag] = whisper.decode(model, torch.from_numpy(mel)[0], opts).text
            model.encoder = real_enc
    print(wav.split('/')[-1])
    for k, v in texts.items():
        print(f'  {k:8s}: {v}')
    print('  ctrl==pad1536:', texts['ctrl'] == texts['pad1536'], '| torch==pad1536:', texts['torch'] == texts['pad1536'])
