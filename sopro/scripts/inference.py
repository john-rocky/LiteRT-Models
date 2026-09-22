"""Portable CPU teacher forcing and free sampling with the exact host formulas."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import soundfile as sf
import ai_edge_litert
from ai_edge_litert.environment import Environment,EnvironmentOptions
from common import ROOT,bounded_run,metadata,sha256,compare,write_json
from litert_utils import CM,MergedCM
from host_dsp import HostDSP
from host_sampler import generate_tokens
from host_postprocess import postprocess_segment
from host_stream_istft import StreamingISTFT,decode_static
from perceptual_proxies import paired_proxies
import ar_host
import acoustic_host


class Signature:
    def __init__(self,model,key):self.model,self.key=model,key
    def call_ordered(self,*arrays):return self.model.call(self.key,*arrays)
    def call_named(self,arrays):
        out=self.call_ordered(*(arrays[f'args_{i}'] for i in range(len(arrays))))
        return {f'output_{i}':v for i,v in enumerate(out)}
    def close(self):pass


class Timed:
    def __init__(self,model,name,pipeline):self.model,self.name,self.pipeline=model,name,pipeline
    def call_ordered(self,*arrays):
        start=time.perf_counter();out=self.model.call_ordered(*arrays)
        self.pipeline.timings.setdefault(self.name,[]).append((time.perf_counter()-start)*1000)
        return out
    def call_named(self,arrays):
        out=self.call_ordered(*(arrays[f'args_{i}'] for i in range(len(arrays))))
        return {f'output_{i}':v for i,v in enumerate(out)}
    def close(self):self.model.close()


class Pipeline:
    def __init__(self,precision='fp32',bucket=2048,stream=False,merged=False):
        self.precision,self.bucket,self.stream=precision,bucket,stream
        self.timings,self.paths={},{}
        self.environment=Environment.create(options=EnvironmentOptions(runtime_path=str(Path(ai_edge_litert.__file__).parent)))
        self.merged=None
        def factory(path,threads=4,environment=None):
            name=Path(path).stem.removeprefix('sopro_').removesuffix('_fp32')
            suffix='fp32' if precision=='fp32' else 'wfp16'
            if name in ('ar_prefill','ar_step') and (merged or precision=='ship'):
                variant='i8native' if precision=='ship' else suffix
                file=ROOT/'exports'/f'sopro_ar_merged_{variant}.tflite'
                if self.merged is None:self.merged=MergedCM(file,threads,self.environment)
                model=Signature(self.merged,'prefill' if name=='ar_prefill' else 'step')
            else:
                actual=name+'_t4096' if bucket==4096 and name in ('acoustic_condition','acoustic_velocity') else name
                file=ROOT/'exports'/f'sopro_{actual}_{suffix}.tflite';model=CM(file,threads,self.environment)
            self.paths[name]={'path':str(file.relative_to(ROOT)),'sha256':sha256(file),'size_bytes':file.stat().st_size}
            return Timed(model,name,self)
        acoustic_host.N_MAX,acoustic_host.T_MAX=bucket//4,bucket
        with ExitStack() as stack:
            for module in (ar_host,acoustic_host):stack.enter_context(patch.object(module,'CM',factory))
            self.ar=ar_host.ARRunner(environment=self.environment)
            self.acoustic=acoustic_host.AcousticRunner(environment=self.environment)
        self.speaker=factory(ROOT/'exports/sopro_speaker_encoder_fp32.tflite')
        self.semantic=factory(ROOT/'exports/sopro_semantic_encoder_fp32.tflite')
        self.vocoders={mode:factory(ROOT/'exports'/f'sopro_vocoder_stream_{mode}_fp32.tflite') for mode in ('start','step','flush')} if stream else {}
        self.vocoder=None if stream else factory(ROOT/'exports/sopro_vocoder_fp32.tflite')
        # The numerical gates use these fp32 tables. Compact binary fp16 tables
        # are optional assets and are not silently substituted here.
        self.dsp=HostDSP(ROOT/'fixtures/dsp_constants.npz')

    def reference(self,data):
        wav=data['reference_wav24'];wav16=self.dsp.resample_24_16(wav)
        ident,style,control,cond=self.speaker.call_ordered(self.dsp.mel(wav16,'speaker'))
        tokens=self.semantic.call_ordered(self.dsp.mel(wav16,'semantic'))[0]
        mel=(self.dsp.mel(wav,'acoustic')-self.dsp.c['mel_mean'])/self.dsp.c['mel_std']
        return {'id_emb':ident,'cond_vec':cond,'ref_tokens':tokens,'ref_mel_normalized':mel}

    def decode(self,reference,tokens,x0,level):
        n=tokens.size;prompt=reference['ref_mel_normalized'].shape[-1]
        solved=self.acoustic.solve(reference['ref_tokens'],tokens,x0,reference['cond_vec'],reference['ref_mel_normalized'])
        ctx=min(32,prompt);mel=solved['solved_mel_normalized'][:,:,prompt-ctx:]*self.dsp.c['mel_std']+self.dsp.c['mel_mean']
        if self.stream:
            wav,features=decode_static(mel,lambda mode,args:self.vocoders[mode].call_ordered(*args),StreamingISTFT(self.dsp.c['istft_window']))
        else:
            valid=mel.shape[-1];assert valid<=1024,'Use --stream for longer waveforms'
            mask=np.zeros((1,1,1024),np.float32);mask[:,:,:valid]=1
            features=self.vocoder.call_ordered(np.pad(mel,((0,0),(0,0),(0,1024-valid))),mask)[0][:,:valid]
            wav=self.dsp.decode_features(features)
        raw=wav[0,ctx*256:ctx*256+n*1024];final,trim=postprocess_segment(raw,level)
        return {'raw_segment_wav':raw,'final_wav':final,'solved_mel_normalized':solved['solved_mel_normalized'],
                'istft_features':features,'trim':trim,'valid_acoustic_frames':solved['valid_frames']}

    def close(self):
        for item in (self.ar,self.acoustic,self.speaker,self.semantic,*self.vocoders.values()):item.close()
        if self.vocoder:self.vocoder.close()
        if self.merged:self.merged.close()
        self.environment.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('teacher','free'),default='teacher')
    p.add_argument('--precision',choices=('fp32','wfp16','ship'),default='fp32');p.add_argument('--bucket',type=int,choices=(2048,4096),default=2048)
    p.add_argument('--stream',action='store_true');p.add_argument('--merged',action='store_true');p.add_argument('--oracle-summary',default='results/oracle/summary.json')
    p.add_argument('--baseline-summary');p.add_argument('--output',default='results/inference');p.add_argument('--seed-base',type=int,default=2000);a=p.parse_args()
    bounded_run('inference_'+a.mode,12000);folder=ROOT/a.output;folder.mkdir(parents=True,exist_ok=True)
    oracle=json.loads((ROOT/a.oracle_summary).read_text())['utterances'];baseline={}
    if a.baseline_summary:baseline={v['utterance']:v for v in json.loads((ROOT/a.baseline_summary).read_text())['rows']}
    pipe=Pipeline(a.precision,a.bucket,a.stream,a.merged)
    report={**metadata(),'precision':a.precision,'mode':a.mode,'streaming':a.stream,'bucket':a.bucket,'exports':pipe.paths,
            'oracle_summary_sha256':sha256(ROOT/a.oracle_summary),'tables_sha256':sha256(ROOT/'exports/sopro_ar_tables_fp32.npz'),
            'dsp_sha256':sha256(ROOT/'fixtures/dsp_constants.npz'),'rows':[],'status':'ACTIVE'}
    for i,u in enumerate(oracle):
        assert sha256(ROOT/u['path'])==u['sha256'];data=dict(np.load(ROOT/u['path']));pipe.timings={};start=time.perf_counter()
        reference=pipe.reference(data);exact=int((reference['ref_tokens']==data['ref_semantic_tokens']).sum())
        if a.mode=='teacher':reference['ref_tokens']=data['ref_semantic_tokens'].astype(np.int32)
        prefix,_=pipe.ar.build_prefix(reference['ref_tokens'],data['text_ids']);initial,_,_=pipe.ar.start(prefix)
        if a.mode=='teacher':
            tokens=data['sampled_semantic_tokens'].astype(np.int32);logs=[initial]
            for token in tokens.ravel():logs.append(pipe.ar.advance(token)[0])
            logits=np.stack(logs);x0=data['x0'];stop='oracle teacher forcing'
        else:
            rng=np.random.default_rng(a.seed_base+i);sampled=generate_tokens(initial,pipe.ar.advance,rng)
            tokens,logits,stop=sampled['tokens'],sampled['logits'],sampled['stop_reason']
            x0=rng.standard_normal((1,100,938+4*tokens.size),dtype=np.float32)
        output=pipe.decode(reference,tokens,x0,float(data['reference_level_db']));elapsed=time.perf_counter()-start
        row={'utterance':u['id'],'text':u['text'],'lang':u['lang'],'reference_tokens_exact':exact,'teacher_ref_tokens_fixed':a.mode=='teacher',
             'token_count':tokens.size,'prefix_length':prefix.shape[1],'prefix_plus_steps':pipe.ar.position,'stop_reason':stop,
             'seed':int(data['seed']) if a.mode=='teacher' else a.seed_base+i,'oracle_sha256':u['sha256'],
             'graph_ms_contended':pipe.timings,'wall_seconds_contended':elapsed,'rtf_contended':elapsed/(output['final_wav'].size/24000)}
        finite=all(np.isfinite(output[k]).all() for k in ('raw_segment_wav','final_wav','solved_mel_normalized'))
        row['audio_pass']=bool(finite and np.sqrt(np.mean(output['final_wav'].astype(np.float64)**2))>1e-3 and np.max(np.abs(output['final_wav']))<=1)
        if a.mode=='teacher':
            raw=compare(data['raw_segment_wav'],output['raw_segment_wav']);mel=compare(pipe.dsp.mel(data['raw_segment_wav'][None],'acoustic'),pipe.dsp.mel(output['raw_segment_wav'][None],'acoustic'))
            row.update(raw_waveform=raw,chain_logmel=mel,solved_mel=compare(data['solved_mel_normalized'],output['solved_mel_normalized']))
            row['domain_pass']=bool(mel['finite'] and mel['corr']>=.99 and (a.precision!='fp32' or raw['corr']>=.99))
            if a.precision=='fp32' and a.bucket==2048:
                same_length=output['final_wav'].shape==data['final_wav'].shape
                final=compare(data['final_wav'],output['final_wav']) if same_length else {'finite':False}
                row['final_waveform']=final
                row['domain_pass']=bool(raw['finite'] and raw['corr']>=.999 and raw['max_abs_diff']<=.005 and mel['corr']>=.999 and final['finite'] and final['corr']>=.999)
            if baseline:
                b=baseline[u['id']];assert sha256(ROOT/b['evidence'])==b['sha256'];prior=dict(np.load(ROOT/b['evidence']))
                assert np.array_equal(prior['sampled_semantic_tokens'],tokens) and np.array_equal(prior['x0'],x0)
                row['proxies']=paired_proxies(prior['raw_segment_wav'],output['raw_segment_wav'])
                row['paired_fp32_raw']=compare(prior['raw_segment_wav'],output['raw_segment_wav'])
                base_logits=prior['logits'];assert base_logits.shape==logits.shape
                disagreements=[];agreements=0
                for step,(expected,got) in enumerate(zip(base_logits,logits)):
                    flat=expected.ravel();top=np.argsort(-flat)[:2];chosen=int(got.argmax())
                    same=chosen==int(top[0]);agreements+=same
                    if not same:
                        gap=float(flat[top[0]]-flat[top[1]])
                        disagreements.append({'prediction':step,'fp32_top1':int(top[0]),'actual_top1':chosen,
                            'fp32_top1_top2_gap':gap,'near_tie':gap<.05})
                row['ar_replay']={'predictions':len(logits),'agreements':agreements,'agreement_rate':agreements/len(logits),
                    'all_disagreements_near_tie':all(d['near_tie'] for d in disagreements),'disagreements':disagreements,
                    'logits':compare(base_logits,logits),'scope':'Teacher-forced replay including prefill and one prediction after each fixed token; integer/float metrics informational, free quality scored separately.'}
        npz=folder/(u['id']+'.npz');np.savez_compressed(npz,**{k:v for k,v in output.items() if isinstance(v,np.ndarray)},sampled_semantic_tokens=tokens,x0=x0,logits=logits,ref_tokens=reference['ref_tokens'])
        wav=folder/(u['id']+'.wav');sf.write(wav,output['final_wav'],24000,subtype='FLOAT')
        row.update(evidence=str(npz.relative_to(ROOT)),sha256=sha256(npz),wav_path=str(wav.relative_to(ROOT)),wav_sha256=sha256(wav))
        row['pass']=bool(row['audio_pass'] and row.get('domain_pass',True) and row.get('proxies',{}).get('pass',True))
        report['rows'].append(row);write_json(folder/'summary.json',report);print(u['id'],row['pass'],flush=True)
    pipe.close();report['status']='PASS' if all(v['pass'] for v in report['rows']) else 'FAIL';write_json(folder/'summary.json',report)
    assert report['status']=='PASS', 'Inference acceptance failed; saved report contains every measured row'


if __name__=='__main__':main()
