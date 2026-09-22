"""Export exact static graph rewrites with the pinned conversion environment."""
import argparse
from pathlib import Path
import sys

DIRECTORY=Path(__file__).resolve().parent
sys.path.insert(0,str(DIRECTORY.parent));sys.path.insert(0,str(DIRECTORY/'modules'))

GRAPHS=('ar_merged','acoustic_condition','acoustic_condition_t4096','acoustic_velocity',
        'acoustic_velocity_t4096','semantic_encoder','style_prefix')


def examples(tts,graph,data):
    import numpy as np
    import torch
    from convert import ar_examples,modules
    if graph=='ar_merged':
        from ar_graphs import ARPrefillGraph,ARStepGraph
        pre,step=ar_examples(tts,data)
        onehot=torch.zeros((1,256),dtype=torch.float32);onehot[0,int(pre[2].item())]=1
        first=ARPrefillGraph(tts.model).eval();second=ARStepGraph(tts.model).eval();second.ar=first.ar
        return [('prefill',first,(*pre[:2],onehot)),('step',second,step)]
    if graph.startswith('acoustic_condition'):
        import acoustic_host
        from acoustic_condition_r6 import AcousticConditionGraphR6
        frames=4096 if graph.endswith('_t4096') else 2048
        acoustic_host.T_MAX,acoustic_host.N_MAX=frames,frames//4
        values=acoustic_host.prepare_inputs(*(data[k] for k in ('ref_semantic_tokens','sampled_semantic_tokens','x0','cond_vec','ref_mel_normalized')))
        vocab=int(tts.model.acoustic_head.semantic_token_emb.num_embeddings);capacity=frames//4
        tokens=np.zeros((1,capacity,vocab),np.float32);tokens[0,np.arange(capacity),values['semantic_tokens'][0]]=1
        mapping=np.zeros((frames,capacity),np.float32);mapping[np.arange(frames),values['frame_to_token']]=1
        args=tuple(torch.from_numpy(v) for v in (tokens,values['token_mask'],mapping))
        return [('serving_default',AcousticConditionGraphR6(tts.model.acoustic_head).eval(),args)]
    if graph.startswith('acoustic_velocity'):
        from acoustic_velocity_r6 import AcousticVelocityGraphR6
        _,_,args=modules(tts,graph,data)[0]
        frames=4096 if graph.endswith('_t4096') else 2048
        return [('serving_default',AcousticVelocityGraphR6(tts.model.acoustic_head,frames).eval(),args)]
    if graph=='semantic_encoder':
        from semantic_logits_r6 import SemanticLogitsGraphR6
        return [('serving_default',SemanticLogitsGraphR6(tts.semantic_encoder).eval(),(torch.from_numpy(data['semantic_mel']),))]
    from style_prefix_r6 import StylePrefixGraphR6
    with torch.no_grad():value=tts.model.embed_semantic(torch.from_numpy(data['ref_semantic_tokens'][:,:160]))
    return [('serving_default',StylePrefixGraphR6(tts.model).eval(),(value,))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph',choices=GRAPHS)
    parser.add_argument('--precision',choices=('fp32','wfp16','i8native'),default='fp32')
    parser.add_argument('--input-npz',type=Path,help='Prepared reference and synthesis example with canonical oracle.py keys')
    parser.add_argument('--checkpoint',type=Path,help='Local Sopro checkpoint directory; otherwise use source.py output')
    parser.add_argument('--output-dir',type=Path,default=Path('exports_r6'))
    args=parser.parse_args()
    if args.precision!='wfp16' and args.input_npz is None:parser.error('--input-npz is required for export examples')
    if args.precision=='i8native' and args.graph!='ar_merged':parser.error('Native int8 is supported only for merged AR')
    import numpy as np
    import torch
    from common import bounded_run,contiguous_constants,load_tts,sha256,write_json,versions
    from convert import half
    from flatbuffer_rewrites import promote_batch_matmul_rank4
    from ar_rank4 import promote_existing_reshapes
    from opscan import scan
    bounded_run('exact_reexport',5400)
    storage='int8' if args.precision=='i8native' else args.precision
    target=args.output_dir/storage/f'sopro_{args.graph}_{args.precision}.tflite';target.parent.mkdir(parents=True,exist_ok=True)
    fp32=args.output_dir/'fp32'/f'sopro_{args.graph}_fp32.tflite'
    report={'graph':args.graph,'precision':args.precision,'versions':versions()}
    if args.precision=='wfp16':
        report['recipe']=half(fp32,target)
        report['alias_probe']='Inherited from the audited fp32 export; direct FLOAT_CASTING without repacking external buffers.'
    else:
        if args.checkpoint:
            from sopro import SoproTTS
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            tts=SoproTTS.from_pretrained(str(args.checkpoint),device='cpu',dtype=torch.float32)
        else:tts=load_tts()
        with np.load(args.input_npz) as archive:data={name:archive[name] for name in archive.files}
        signatures=examples(tts,args.graph,data)
        contiguous_constants(torch.nn.ModuleDict({n:m for n,m,_ in signatures}))
        signatures=[(n,m,tuple(v.detach().clone().contiguous() for v in values)) for n,m,values in signatures]
        import alias_probe,litert_torch
        alias_probe.install();config=None
        if args.precision=='i8native':
            from litert_torch.quantize import pt2e_quantizer
            from litert_torch.quantize.quant_config import QuantConfig
            from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e,convert_pt2e
            recipe=pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True,is_dynamic=True)
            global_q=pt2e_quantizer.PT2EQuantizer().set_global(recipe);converted=[]
            for name,module,example in signatures:
                q=pt2e_quantizer.PT2EQuantizer().set_global(recipe)
                prepared=prepare_pt2e(torch.export.export(module,example).module(),q)
                with torch.no_grad():prepared(*example)
                converted.append((name,convert_pt2e(prepared,fold_quantize=False),example))
            signatures=converted;contiguous_constants(torch.nn.ModuleDict({n:m for n,m,_ in signatures}))
            config=QuantConfig(pt2e_quantizer=global_q)
        if len(signatures)==1 and config is None:
            _,module,example=signatures[0];litert_torch.convert(module,example).export(str(target))
        else:
            converter=litert_torch.signature(*signatures[0])
            for signature in signatures[1:]:converter.signature(*signature)
            converter.convert(**({'quant_config':config} if config is not None else {})).export(str(target))
        report['alias_probe']={'aliased_constants':len(alias_probe._collisions),'constants_lowered':alias_probe._order[0]}
        assert not alias_probe._collisions
        report['layout']=promote_existing_reshapes(target) if args.graph=='ar_merged' else promote_batch_matmul_rank4(target)
    inspection=scan(target);assert inspection['pass_'],inspection['forbidden']
    write_json(target.with_suffix('.opcheck.json'),inspection)
    report.update(status='EXPORTED',path=str(target),sha256=sha256(target),bytes=target.stat().st_size)
    write_json(target.with_suffix('.export.json'),report)
    print(target,report['sha256'],report['bytes'],flush=True)


if __name__=='__main__':main()
