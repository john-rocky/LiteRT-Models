"""User-manifest torch oracle with a deterministic ten-second reference bucket.

Manifest: references [{id, source_path}], sentences [{id, text, lang}], and
optional seed_base (default1000). Audio paths resolve against the manifest.
Generated fixtures/audio/results are private work-directory artifacts.
"""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
from sopro import audio as audio_ops
from sopro.model import Reference
from sopro.text import split_text, language_tag
from common import ROOT, bounded_run, load_tts, metadata, sha256, write_json


def freeze_dsp(tts):
    from torchaudio.functional.functional import _get_sinc_resample_kernel
    bank = {}
    frontends = {'speaker': tts.speaker_encoder.frontend.mel,
                 'semantic': tts.semantic_encoder.frontend.mel,
                 'acoustic': tts.vocoder.feature_extractor.mel_spec}
    for name, mel in frontends.items():
        bank[name+'_window'] = mel.spectrogram.window.detach().numpy()
        bank[name+'_melbank'] = mel.mel_scale.fb.detach().numpy()
    bank['istft_window'] = tts.vocoder.head.istft.window.detach().numpy()
    kernel, width = _get_sinc_resample_kernel(24000, 16000, 8000, dtype=torch.float32, device=torch.device('cpu'))
    bank['resample_24_16_kernel'] = kernel.numpy()
    bank['resample_24_16_width'] = np.array(width, dtype=np.int32)
    bank['mel_mean'], bank['mel_std'] = tts.mel_mean.numpy(), tts.mel_std.numpy()
    np.savez(ROOT/'fixtures/dsp_constants.npz', **bank)


@torch.inference_mode()
def fixed_reference(tts, source_path):
    wav = audio_ops.load_audio(source_path, tts.sample_rate).unsqueeze(0)
    original_samples = wav.shape[-1]
    if original_samples == 0:
        raise ValueError('Reference audio must contain samples')
    # This order preserves the measured >=ten-second path exactly. The new
    # short-input convenience branch is explicit zero padding AFTER source
    # normalization; its downstream quality has not been validated.
    wav, level_db = audio_ops.normalize_reference(wav, tts.sample_rate)
    padded = max(0, 240000-original_samples)
    wav = torch.nn.functional.pad(wav, (0, padded))[..., :240000].contiguous()
    wav16 = torchaudio.functional.resample(wav, 24000, 16000)
    spk = tts.speaker_encoder(wav16)
    cond = tts.model.build_condition(*(spk[k] for k in ('id_emb', 'style_emb', 'style_ctrl')))
    tokens = tts.semantic_encoder.encode(wav)
    mel = (tts.vocoder.mel(wav)-tts.mel_mean)/tts.mel_std
    sem_mel, sem_frames = tts.semantic_encoder.frontend(wav16)
    arrays = {'reference_wav24': wav.numpy(), 'reference_wav16': wav16.numpy(),
              'speaker_mel': tts.speaker_encoder.frontend(wav16).numpy(),
              'semantic_mel': sem_mel.numpy(), 'ref_mel_normalized': mel.numpy(),
              'cond_vec': cond.numpy(), 'ref_semantic_tokens': tokens.numpy(),
              'reference_level_db': np.array(level_db, dtype=np.float64),
              **{k: v.numpy() for k, v in spk.items()}}
    expected = {'reference_wav24': (1,240000), 'reference_wav16': (1,160000),
                'speaker_mel': (1,80,1001), 'semantic_mel': (1,80,1002),
                'ref_semantic_tokens': (1,235), 'ref_mel_normalized': (1,100,938)}
    for key, shape in expected.items():
        assert arrays[key].shape == shape, (key, arrays[key].shape, shape)
    assert sem_frames == 1000
    preparation = {'source_samples_after_resample': original_samples, 'padded_zero_samples': padded,
                   'cropped_samples': max(0, original_samples-240000),
                   'short_reference_padding_quality_validated': False if padded else None}
    return Reference(cond, tokens, mel, float(level_db)), arrays, preparation


def checked_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Identifiers must use letters, digits, underscores or hyphens')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    opts = parser.parse_args()
    bounded_run('oracle', 6600)
    manifest_path = opts.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    references, sentences = manifest['references'], manifest['sentences']
    if not references or not sentences:
        raise ValueError('Manifest requires at least one reference and one sentence')
    for entries in (references, sentences):
        ids = [checked_id(item['id']) for item in entries]
        if len(ids) != len(set(ids)):
            raise ValueError('Identifiers must be unique within references/sentences')
    for sentence in sentences:
        if not 0 < len(sentence['text']) <= 300 or len(split_text(sentence['text'],300)) != 1:
            raise ValueError('Each oracle sentence must be one segment of at most300 characters')
        language_tag(sentence['lang'])
    seed_base = int(manifest.get('seed_base', 1000))
    max_seconds = float(manifest.get('max_seconds', 30.0))
    tts = load_tts()
    freeze_dsp(tts)
    captured = {}
    original_solve = tts.model.acoustic_head.solve
    def solve(*args, **kwargs):
        captured['x0'] = args[0].detach().clone()
        out = original_solve(*args, **kwargs)
        captured['solved_mel_normalized'] = out.detach().clone()
        return out
    tts.model.acoustic_head.solve = solve
    original_segment = tts._synthesize_segment
    def segment(*args, **kwargs):
        wav, tokens = original_segment(*args, **kwargs)
        captured['raw_segment_wav'] = wav.detach().clone()
        captured['sampled_semantic_tokens'] = tokens.detach().clone()
        return wav, tokens
    tts._synthesize_segment = segment
    summary = {**metadata(), 'runtime': 'torch eager', 'torch_threads': torch.get_num_threads(),
               'sample_rate_hz': 24000, 'seed_rule': f'torch.manual_seed({seed_base}+i), reference-major sentence order',
               'manifest_sha256': sha256(manifest_path), 'reference_seconds': 10.0,
               'crop_on_pause': 'bypassed for deterministic static reference',
               'short_reference_padding': 'Zeros after normalization; quality unvalidated for padded references.',
               'utterances': [], 'status': 'IN_PROGRESS'}
    frozen = {'dsp_constants_sha256': sha256(ROOT/'fixtures/dsp_constants.npz'), 'references': []}
    with torch.inference_mode():
        for fixture in references:
            source = manifest_path.parent / fixture['source_path']
            ref, arrays, preparation = fixed_reference(tts, source)
            ref_path = ROOT/'fixtures'/(fixture['id']+'.npz')
            np.savez_compressed(ref_path, **arrays)
            frozen['references'].append({'id': fixture['id'], 'path': str(ref_path.relative_to(ROOT)),
                'sha256': sha256(ref_path), 'source_sha256': sha256(source), 'preparation': preparation,
                'license': fixture.get('license'), 'shapes': {k:list(v.shape) for k,v in arrays.items()}})
            write_json(ROOT/'fixtures/frozen_inputs.json', frozen)
            for sentence in sentences:
                i = len(summary['utterances'])
                name = fixture['id']+'_'+sentence['id']
                folder = ROOT/'results/oracle'/name
                folder.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                captured.clear()
                torch.manual_seed(seed_base+i)
                final = tts.synthesize(sentence['text'], ref=ref, lang=sentence['lang'], max_seconds=max_seconds)
                dump = {**arrays, **{k:v.numpy() for k,v in captured.items()}, 'final_wav': final.numpy(),
                        'text_ids': tts.tokenizer.encode_tensor(sentence['text'],sentence['lang'],tts.device).numpy(),
                        'seed':np.array(seed_base+i), 'text':np.array(sentence['text']), 'lang':np.array(sentence['lang']),
                        'lang_tag':np.array(language_tag(sentence['lang']))}
                prompt = ref.mel.shape[-1]
                dump['decode_mel_normalized'] = dump['solved_mel_normalized'][:,:,prompt-min(32,prompt):]
                out16 = torchaudio.functional.resample(final[None],24000,16000)
                out_id = tts.speaker_encoder(out16)['id_emb']
                cosine = float(torch.nn.functional.cosine_similarity(out_id,torch.from_numpy(arrays['id_emb']),dim=-1)[0])
                dump['output_id_emb'] = out_id.numpy()
                wav_metrics = {}
                for key in ('reference_wav24','reference_wav16','raw_segment_wav','final_wav'):
                    wav = dump[key]
                    wav_metrics[key] = {'finite':bool(np.isfinite(wav).all()),
                        'rms':float(np.sqrt(np.mean(wav.astype(np.float64)**2))), 'absmax':float(np.abs(wav).max())}
                finite = all(np.isfinite(a).all() for a in dump.values() if a.dtype.kind in 'fc')
                path = folder/'oracle.npz'
                np.savez_compressed(path, **dump)
                sf.write(folder/'final.wav',final.numpy(),24000,subtype='FLOAT')
                entry = {'id':name,'seed':seed_base+i,'lang':sentence['lang'],'text':sentence['text'],
                         'token_count':int(dump['sampled_semantic_tokens'].size), 'text_token_count':int(dump['text_ids'].size),
                         'seconds':final.numel()/24000,'raw_seconds':dump['raw_segment_wav'].size/24000,
                         'speaker_cosine':cosine,'wav_metrics':wav_metrics,'all_arrays_finite':bool(finite),
                         'reference_preparation':preparation,'max_steps':tts._steps(max_seconds),
                         'hit_max_steps':dump['sampled_semantic_tokens'].size>=tts._steps(max_seconds),
                         'path':str(path.relative_to(ROOT)),'sha256':sha256(path),
                         'elapsed_seconds_contended':time.monotonic()-started,
                         'pass':bool(finite and all(m['finite'] and m['rms']>1e-3 for m in wav_metrics.values()))}
                write_json(folder/'metrics.json',entry)
                summary['utterances'].append(entry)
                write_json(ROOT/'results/oracle/summary.json',summary)
                print(json.dumps(entry,ensure_ascii=False),flush=True)
    summary['status'] = 'PASS' if all(u['pass'] for u in summary['utterances']) else 'FAIL'
    write_json(ROOT/'results/oracle/summary.json',summary)


if __name__ == '__main__':
    main()
