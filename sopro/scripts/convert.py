"""Reproduce one static graph or weight variant in the selected work directory."""
import argparse
import json
import time
import numpy as np
import torch
from common import ROOT, bounded_run, load_tts, contiguous_constants, compare, float_gate, sha256, write_json, metadata
from litert_utils import inspect_flatbuffer

NAMES = ('speaker_encoder', 'semantic_encoder', 'style_prefix', 'ar_prefill', 'ar_step',
         'acoustic_condition', 'acoustic_velocity', 'vocoder', 'ar_merged',
         'vocoder_stream_start', 'vocoder_stream_step', 'vocoder_stream_flush',
         'acoustic_condition_t4096', 'acoustic_velocity_t4096')


def first_oracle(path=None):
    if path is None:
        summary = json.loads((ROOT/'results/oracle/summary.json').read_text())
        path = ROOT/summary['utterances'][0]['path']
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


@torch.no_grad()
def tables(model):
    actual = model.sem_in_proj(model.semantic_tok_emb.weight)
    reference = model.embed_semantic(torch.arange(4377).reshape(1,-1))[0]
    metric = compare(reference.numpy(), actual.numpy())
    assert float_gate(metric) and metric['max_abs_diff'] <= 1e-5, metric
    np.savez(ROOT/'exports/sopro_ar_tables_fp32.npz', text_tok_emb=model.text_tok_emb.weight.numpy(),
             sem_emb512=actual.numpy(), bos_id=np.array(model.cfg.semantic_bos_id,np.int32),
             eos_id=np.array(model.cfg.semantic_eos_id,np.int32), max_text_len=np.array(model.cfg.max_text_len,np.int32))
    write_json(ROOT/'results/ar_table_fold.json', metric)


@torch.no_grad()
def ar_examples(tts, data):
    from ar_host import prefill_bias, step_bias, rotary_cos_sin
    refs, text = torch.from_numpy(data['ref_semantic_tokens']), torch.from_numpy(data['text_ids'])
    model = tts.model
    prefix = model.build_prefix(text, refs[:,:160], refs[:,:120]); n=prefix.shape[1]
    assert n <= 256
    cache = model.ar_prior.new_cache(1,1024,torch.device('cpu'),torch.float32)
    model.ar_prior(prefix,cache)
    pk,pv = torch.cat(cache.k,dim=1),torch.cat(cache.v,dim=1)
    pk[:,:,n:],pv[:,:,n:] = 0,0
    cos,sin=rotary_cos_sin([n]); token=int(data['sampled_semantic_tokens'].ravel()[0])
    prefill=(torch.nn.functional.pad(prefix,(0,0,0,256-n)),torch.from_numpy(prefill_bias(n)),torch.tensor([n-1],dtype=torch.int32))
    step=(model.embed_semantic(torch.tensor([[token]])),torch.from_numpy(cos[None,None]),
          torch.from_numpy(sin[None,None]),torch.from_numpy(step_bias(n)),pk,pv)
    return prefill,step


def modules(tts, name, data=None):
    if name=='speaker_encoder':
        from graphs import SpeakerGraph
        mel=torch.zeros(1,80,1001) if data is None else torch.from_numpy(data['speaker_mel'])
        return [('serving_default',SpeakerGraph(tts).eval(),(mel,))]
    if name=='semantic_encoder':
        from semantic_graph import SemanticDiagnosticGraph
        return [('serving_default',SemanticDiagnosticGraph(tts.semantic_encoder).eval(),(torch.from_numpy(data['semantic_mel']),))]
    if name in ('style_prefix','ar_prefill','ar_step','ar_merged'):
        from ar_graphs import StylePrefixGraph,ARPrefillGraph,ARStepGraph
        tables(tts.model)
        if name=='style_prefix':
            with torch.no_grad(): emb=tts.model.embed_semantic(torch.from_numpy(data['ref_semantic_tokens'][:,:160]))
            return [('serving_default',StylePrefixGraph(tts.model).eval(),(emb,))]
        pa,sa=ar_examples(tts,data)
        p,s=ARPrefillGraph(tts.model).eval(),ARStepGraph(tts.model).eval()
        if name=='ar_merged':
            s.ar=p.ar
            return [('prefill',p,pa),('step',s,sa)]
        return [('serving_default',p,pa)] if name=='ar_prefill' else [('serving_default',s,sa)]
    if name.startswith('acoustic_'):
        import acoustic_host
        from acoustic_graphs import AcousticConditionGraph,AcousticVelocityGraph
        frames=4096 if name.endswith('_t4096') else 2048
        acoustic_host.N_MAX,acoustic_host.T_MAX=frames//4,frames
        inputs=acoustic_host.prepare_inputs(data['ref_semantic_tokens'],data['sampled_semantic_tokens'],data['x0'],data['cond_vec'],data['ref_mel_normalized'])
        args={k:torch.from_numpy(v) for k,v in inputs.items() if isinstance(v,np.ndarray)}
        module=AcousticConditionGraph(tts.model.acoustic_head) if name.startswith('acoustic_condition') else AcousticVelocityGraph(tts.model.acoustic_head,frames)
        if name.startswith('acoustic_velocity'):
            head=tts.model.acoustic_head; valid=inputs['valid_frames']
            with torch.no_grad():
                latents=head.semantic_latents(args['semantic_tokens'][:,:inputs['valid_tokens']],torch.float32)
                mu=head.mu_proj(head.semantic_upsampler(head.semantic_prelook(latents),valid))
            args['mu']=torch.nn.functional.pad(mu,(0,frames-valid));args['t']=torch.tensor([acoustic_host.build_time_grid()[1]])
        return [('serving_default',module.eval(),tuple(args[k] for k in module.input_names))]
    mel=torch.from_numpy(data['decode_mel_normalized'])*tts.mel_std+tts.mel_mean
    if name=='vocoder':
        from graphs import VocoderGraph
        n=min(mel.shape[-1],1024);mask=torch.zeros(1,1,1024);mask[:,:,:n]=1
        return [('serving_default',VocoderGraph(tts.vocoder).eval(),(torch.nn.functional.pad(mel[:,:,:n],(0,1024-n)),mask))]
    from vocoder_stream_graphs import VocoderStreamGraph,source_examples
    mode=name.removeprefix('vocoder_stream_')
    assert mel.shape[-1]>=128,'Streaming static examples require at least 128 real mel frames'
    return [('serving_default',VocoderStreamGraph(tts.vocoder,mode).eval(),source_examples(tts.vocoder,mel)[mode])]


def public_tokens(diagnostic, public):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema
    model=schema.ModelT.InitFromPackedBuf(diagnostic.read_bytes(),0)
    sig=model.signatureDefs[0]; graph=model.subgraphs[sig.subgraphIndex]
    keep=[int(i) for i in graph.outputs if graph.tensors[i].type==schema.TensorType.INT32]
    assert len(keep)==1 and list(graph.tensors[keep[0]].shape)==[1,235]
    graph.outputs=np.array(keep,dtype=np.int32);sig.outputs=[o for o in sig.outputs if o.tensorIndex in keep]
    builder=flatbuffers.Builder(0);builder.Finish(model.Pack(builder),file_identifier=b'TFL3');public.write_bytes(builder.Output())


def half(source, target):
    from ai_edge_quantizer import quantizer,recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName,qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex='.*',operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    recipe=rm.get_quantization_recipe();q=quantizer.Quantizer(float_model=str(source));q.load_quantization_recipe(recipe)
    q.quantize().export_model(str(target),overwrite=target.exists())
    return recipe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('graph',choices=NAMES)
    parser.add_argument('--precision',choices=('fp32','wfp16','i8native'),default='fp32')
    parser.add_argument('--oracle');parser.add_argument('--expected-sha256');args=parser.parse_args()
    bounded_run('convert_'+args.graph,7200)
    path=ROOT/'exports'/f'sopro_{args.graph}_{args.precision}.tflite'
    report={**metadata(),'graph':args.graph,'precision':args.precision,'path':str(path.relative_to(ROOT))}
    started=time.monotonic()
    if args.precision=='wfp16':
        report['recipe']=half(ROOT/'exports'/f'sopro_{args.graph}_fp32.tflite',path)
        report['alias_probe']='Inherited fp32 audited constants; post-hoc FLOAT_CASTING'
    else:
        assert args.precision!='i8native' or args.graph=='ar_merged', 'Only merged AR native int8 is a shipping candidate'
        tts=load_tts();data=None if args.graph=='speaker_encoder' and not args.oracle else first_oracle(args.oracle)
        signatures=modules(tts,args.graph,data)
        container=torch.nn.ModuleDict({k:m for k,m,_ in signatures});contiguous_constants(container)
        signatures=[(k,m,tuple(x.detach().clone() for x in a)) for k,m,a in signatures]
        import litert_torch
        import alias_probe
        alias_probe.install()
        config=None
        if args.precision=='i8native':
            from litert_torch.quantize import pt2e_quantizer
            from litert_torch.quantize.quant_config import QuantConfig
            from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e,convert_pt2e
            recipe=pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True,is_dynamic=True)
            global_q=pt2e_quantizer.PT2EQuantizer().set_global(recipe);converted=[]
            for name,module,example in signatures:
                q=pt2e_quantizer.PT2EQuantizer().set_global(recipe)
                observed=prepare_pt2e(torch.export.export(module,example).module(),q)
                with torch.no_grad():observed(*example)
                converted.append((name,convert_pt2e(observed,fold_quantize=False),example))
            signatures=converted;contiguous_constants(torch.nn.ModuleDict({n:m for n,m,_ in signatures}))
            config=QuantConfig(pt2e_quantizer=global_q)
            report['recipe']='PT2E per-channel symmetric int8 weights; dynamic activation; fold_quantize=False'
        target=path if args.graph!='semantic_encoder' else ROOT/'results/semantic_diagnostic_fp32.tflite'
        if len(signatures)==1 and config is None:
            _,module,example=signatures[0];litert_torch.convert(module,example).export(str(target))
        else:
            converter=litert_torch.signature(*signatures[0])
            for sig in signatures[1:]:converter.signature(*sig)
            converter.convert(**({'quant_config':config} if config is not None else {})).export(str(target))
        if args.graph=='semantic_encoder':public_tokens(target,path)
        report['alias_probe']={'aliased_constants':len(alias_probe._collisions),'constants_lowered':alias_probe._order[0],'collisions':alias_probe._collisions}
        assert not alias_probe._collisions
    inspection=inspect_flatbuffer(path);assert inspection['pass']
    write_json(ROOT/'results'/f'{args.graph}_{args.precision}_opcheck.json',inspection)
    report.update(status='PASS',sha256=sha256(path),size_bytes=path.stat().st_size,seconds_contended=time.monotonic()-started)
    if args.expected_sha256:
        report['expected_sha256']=args.expected_sha256;report['byte_identical']=report['sha256']==args.expected_sha256
        assert report['byte_identical'],report
    write_json(ROOT/'results'/f'{args.graph}_{args.precision}_export.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
