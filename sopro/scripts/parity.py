"""CPU graph and chain gates over user-supplied references and regenerated oracles."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, bounded_run, load_tts, metadata, compare, float_gate, sha256, write_json
from litert_utils import CM


def metric(reference, actual):
    result=compare(reference,actual);result['pass']=float_gate(result);return result


def speaker(references):
    import torch
    import torchaudio
    from sopro import audio as audio_ops
    from graphs import SpeakerGraph
    from oracle import freeze_dsp
    from host_dsp import HostDSP
    tts=load_tts();freeze_dsp(tts);dsp=HostDSP(ROOT/'fixtures/dsp_constants.npz')
    model=SpeakerGraph(tts,rewrite=False).eval()
    path=ROOT/'exports/sopro_speaker_encoder_fp32.tflite';cm=CM(path);rows=[]
    with torch.no_grad():
        for i,reference in enumerate(references):
            reference=Path(reference).resolve()
            wav=audio_ops.load_audio(reference,24000).unsqueeze(0)
            assert wav.shape[-1]>=240000,'This verification uses references of at least ten seconds'
            wav,_=audio_ops.normalize_reference(wav,24000);wav=wav[...,:240000].contiguous()
            wav16=torchaudio.functional.resample(wav,24000,16000)
            source_mel=tts.speaker_encoder.frontend(wav16).numpy()
            host_mel=dsp.mel(wav16.numpy(),'speaker')
            source_outputs=model(torch.from_numpy(source_mel))
            host_outputs=model(torch.from_numpy(host_mel))
            actual=cm.call_ordered(host_mel)
            results={name:metric(a.numpy(),b) for name,a,b in zip(model.output_names,host_outputs,actual)}
            frontend=metric(source_outputs[0].numpy(),host_outputs[0].numpy())
            frontend['max_abs_diff_limit']=1e-4;frontend['pass']=frontend['pass'] and frontend['max_abs_diff']<=1e-4
            norm=float(np.linalg.norm(actual[0]));passed=all(v['pass'] for v in results.values()) and frontend['pass'] and abs(norm-1)<=1e-4
            evidence=ROOT/'results'/f'speaker_reference_{i}.npz'
            np.savez_compressed(evidence,host_mel=host_mel,source_mel=source_mel,**{name:a for name,a in zip(model.output_names,actual)})
            rows.append({'reference_index':i,'reference_sha256':sha256(reference),'frontend_downstream':frontend,
                         'outputs':results,'id_norm':norm,'pass':bool(passed),'evidence':str(evidence.relative_to(ROOT)),'sha256':sha256(evidence)})
    cm.close();report={**metadata(),'status':'PASS' if rows and all(r['pass'] for r in rows) else 'FAIL',
        'count':len(rows),'export':{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'size_bytes':path.stat().st_size},'rows':rows}
    write_json(ROOT/'results/speaker_cpu_parity.json',report);print(json.dumps(report),flush=True)
    assert report['status']=='PASS'


def graphs(oracle_path=None,bucket=None):
    import torch
    from convert import NAMES,modules,first_oracle
    from semantic_graph import native_digit_logits,tokens_from_logits
    tts=load_tts();rows=[]
    paths=[Path(oracle_path)] if oracle_path else [ROOT/u['path'] for u in json.loads((ROOT/'results/oracle/summary.json').read_text())['utterances']]
    for case in paths:
        data=first_oracle(case)
        for name in NAMES:
            if name=='ar_merged':continue  # Both signatures are exercised by inference.py.
            if bucket==4096 and name in ('acoustic_condition','acoustic_velocity'):continue
            if bucket==2048 and name.endswith('_t4096'):continue
            signatures=modules(tts,name,data);_,model,args=signatures[0]
            path=ROOT/'exports'/f'sopro_{name}_fp32.tflite';cm=CM(path)
            with torch.no_grad():reference=model(*args)
            reference=reference if isinstance(reference,tuple) else (reference,)
            actual=cm.call_ordered(*(x.detach().numpy() for x in args));outputs=[]
            if name=='semantic_encoder':
                native=native_digit_logits(tts.semantic_encoder,args[0])
                expected=tokens_from_logits(native,tts.semantic_encoder.levels,tts.semantic_encoder._bases.to(torch.int32)).detach().numpy();got=actual[0]
                logits=native.detach().numpy()
                positions=np.flatnonzero(expected.ravel()!=got.ravel());parts=np.split(logits,np.cumsum(tts.semantic_encoder.levels)[:-1],axis=-1)
                margins=[]
                for p in positions:
                    digits=[float(np.sort(part[0,p])[-1]-np.sort(part[0,p])[-2]) for part in parts]
                    bases=tts.semantic_encoder._bases.detach().numpy().ravel()
                    changed=[j for j,(base,level) in enumerate(zip(bases,tts.semantic_encoder.levels)) if (int(expected[0,p])//int(base))%level != (int(got[0,p])//int(base))%level]
                    margins.append({'position':int(p),'torch_token':int(expected[0,p]),'litert_token':int(got[0,p]),'digit_top1_top2_gaps':digits,'changed_digits':changed,'all_changed_digits_near_tie':all(digits[j]<.05 for j in changed)})
                exact=int((expected==got).sum());outputs=[{'exact':exact,'total':235,'mismatches':margins,'pass':exact>=233 and all(v['all_changed_digits_near_tie'] for v in margins)}]
                diagnostic=CM(ROOT/'results/semantic_diagnostic_fp32.tflite')
                dtokens,dlogits=diagnostic.call_ordered(args[0].numpy());diagnostic.close()
                outputs.append(metric(logits,dlogits));outputs.append({'public_diagnostic_tokens_exact':bool(np.array_equal(dtokens,got)),'pass':bool(np.array_equal(dtokens,got))})
            else:
                for a,b in zip(reference,actual):
                    x=a.detach().numpy()
                    if name.startswith('acoustic_'):x,b=x[...,:data['x0'].shape[-1]],b[...,:data['x0'].shape[-1]]
                    if name=='vocoder':n=int(args[1].sum());x,b=x[:,:n],b[:,:n]
                    if name=='ar_prefill' and x.ndim==4:n=int(args[2][0])+1;x,b=x[:,:,:n],b[:,:,:n]
                    outputs.append(metric(x,b))
            rows.append({'oracle':str(case),'oracle_sha256':sha256(case),'graph':name,'sha256':sha256(path),'outputs':outputs,'pass':all(o['pass'] for o in outputs)})
            cm.close()
    report={**metadata(),'scope':'fp32 rewritten module vs exported graph on all regenerated oracles by default; --oracle selects one; --bucket limits the acoustic pair; semantic uses native digit logits; full AR replay and chain gates are separate','oracle_count':len(paths),'bucket':bucket,'rows':rows,'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL'}
    write_json(ROOT/'results/graph_cpu_parity.json',report);print(json.dumps(report),flush=True)
    assert report['status']=='PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('speaker','graphs'));p.add_argument('--reference',action='append');p.add_argument('--oracle');p.add_argument('--bucket',type=int,choices=(2048,4096));a=p.parse_args()
    bounded_run('parity_'+a.mode,7200)
    if a.mode=='speaker':
        assert a.reference,'Supply one or more --reference audio paths';speaker(a.reference)
    else:graphs(a.oracle,a.bucket)
