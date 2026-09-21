"""CPU speaker/Whisper quality on private, regenerated oracle and inference audio."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
import jiwer
from whisper.normalizers import EnglishTextNormalizer,BasicTextNormalizer
from common import ROOT,bounded_run,load_tts,metadata,sha256,write_json

ASR_SHA256='aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'
ASR_OPTIONS={'task':'transcribe','fp16':False,'temperature':0.0,'beam_size':5,'condition_on_previous_text':False,'verbose':None}


def summarize(rows):
    return {'count':len(rows),'mean_speaker_cosine':float(np.mean([r['speaker_cosine'] for r in rows])),
            'min_speaker_cosine':min(r['speaker_cosine'] for r in rows),
            'mean_wer_percent':100*float(np.mean([r['wer'] for r in rows])),
            'corpus_wer_percent':100*jiwer.wer([r['normalized_reference'] for r in rows],[r['normalized_hypothesis'] for r in rows]),
            'all_audio_pass':all(r['audio_pass'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--oracle-summary',default='results/oracle/summary.json')
    p.add_argument('--generated-summary',required=True);p.add_argument('--output',default='results/quality.json')
    p.add_argument('--speaker-mean-drop',type=float,default=.03);a=p.parse_args();bounded_run('quality',14400)
    oracle=json.loads((ROOT/a.oracle_summary).read_text())['utterances']
    generation=json.loads((ROOT/a.generated_summary).read_text());actual={v['utterance']:v for v in generation['rows']}
    assert set(actual)=={v['id'] for v in oracle}
    tts=load_tts();model=whisper.load_model('turbo',device='cpu',download_root=str(ROOT/'cache/whisper')).eval()
    assert sha256(ROOT/'cache/whisper/large-v3-turbo.pt')==ASR_SHA256
    assert next(model.parameters()).dtype==torch.float32
    normals={'en':EnglishTextNormalizer()};basic=BasicTextNormalizer();groups={}
    with torch.inference_mode():
        for kind in ('oracle','generated'):
            rows=[]
            for u in oracle:
                assert sha256(ROOT/u['path'])==u['sha256']
                with np.load(ROOT/u['path']) as z:ident=torch.from_numpy(z['id_emb'])
                if kind=='oracle':path=(ROOT/u['path']).parent/'final.wav'
                else:
                    entry=actual[u['id']];path=ROOT/entry['wav_path'];assert sha256(path)==entry['wav_sha256']
                wav,sr=sf.read(path,dtype='float32');assert sr==24000 and wav.ndim==1 and np.isfinite(wav).all()
                audio16=torchaudio.functional.resample(torch.from_numpy(wav)[None],24000,16000)
                output_id=tts.speaker_encoder(audio16)['id_emb']
                cosine=float(torch.nn.functional.cosine_similarity(output_id,ident,dim=-1)[0])
                result=model.transcribe(np.ascontiguousarray(audio16[0].numpy(),dtype=np.float32),language=u['lang'],**ASR_OPTIONS)
                normal=normals.get(u['lang'],basic);reference,hypothesis=normal(u['text']),normal(result['text'])
                scores=jiwer.process_words(reference,hypothesis)
                rms=float(np.sqrt(np.mean(wav.astype(np.float64)**2)));peak=float(np.abs(wav).max())
                rows.append({'utterance':u['id'],'lang':u['lang'],'wav_path':str(path.relative_to(ROOT)),'wav_sha256':sha256(path),
                    'oracle_sha256':u['sha256'],'normalized_reference':reference,'normalized_hypothesis':hypothesis,
                    'transcript':result['text'],'speaker_cosine':cosine,'wer':scores.wer,
                    'word_counts':{'hits':scores.hits,'substitutions':scores.substitutions,'deletions':scores.deletions,'insertions':scores.insertions},
                    'rms':rms,'peak':peak,'audio_pass':bool(rms>1e-3 and peak<=1)})
                print(kind,u['id'],'WER',scores.wer,'speaker_cosine',cosine,flush=True)
            groups[kind]={'rows':rows,'summary':summarize(rows)}
    ref,got=groups['oracle']['summary'],groups['generated']['summary']
    gates={'speaker_mean':got['mean_speaker_cosine']>=ref['mean_speaker_cosine']-a.speaker_mean_drop,
           'speaker_min':got['min_speaker_cosine']>=.80,'mean_wer':got['mean_wer_percent']<=ref['mean_wer_percent']+3,
           'audio':got['all_audio_pass']}
    report={**metadata(),'runtime':'torch fp32 CPU speaker and openai-whisper turbo','asr_sha256':ASR_SHA256,
        'asr_options':ASR_OPTIONS,'audio_input':'16 kHz float32 NumPy array; no ffmpeg','groups':groups,'gates':gates,
        'generated_summary_sha256':sha256(ROOT/a.generated_summary),'oracle_summary_sha256':sha256(ROOT/a.oracle_summary),
        'acceptance':{'speaker_mean_drop':a.speaker_mean_drop,'speaker_min':.80,'mean_wer_increase_points':3.0},
        'status':'PASS' if all(gates.values()) else 'FAIL'}
    write_json(ROOT/a.output,report)
    assert report['status']=='PASS', 'Quality acceptance failed; saved report contains every measured row'


if __name__=='__main__':main()
