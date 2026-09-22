"""Export exact style and streaming-vocoder layout variants."""
import argparse
import json
from pathlib import Path
import sys

DIRECTORY=Path(__file__).resolve().parent
SCRIPT_ROOT=DIRECTORY.parent
sys.path.insert(0,str(SCRIPT_ROOT))
sys.path.insert(0,str(SCRIPT_ROOT/'reexport_r6/modules'))
sys.path.insert(0,str(DIRECTORY/'modules'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph',choices=('style_prefix','vocoder_stream_start','vocoder_stream_step','vocoder_stream_flush'))
    parser.add_argument('--precision',choices=('fp32','wfp16'),default='fp32')
    parser.add_argument('--input-npz',type=Path,help='Prepared reference with ref_semantic_tokens for style conversion')
    parser.add_argument('--checkpoint',type=Path,help='Local Sopro checkpoint directory; otherwise use source.py output')
    parser.add_argument('--source-graph',type=Path,help='Published fp32 streaming graph for metadata-only promotion')
    parser.add_argument('--output-dir',type=Path,default=Path('exports_r9'))
    args=parser.parse_args()
    if args.precision=='fp32' and args.graph=='style_prefix' and args.input_npz is None:parser.error('--input-npz is required for style conversion')
    if args.precision=='fp32' and args.graph!='style_prefix' and args.source_graph is None:parser.error('--source-graph is required for streaming promotion')
    import numpy as np
    import torch
    from common import bounded_run,contiguous_constants,load_tts,sha256,write_json,versions
    from convert import half
    from flatbuffer_rewrites import promote_batch_matmul_rank4
    from fold_constant_reshapes import fold_constant_reshapes,constant_reshape_paths
    from opscan import scan
    bounded_run('exact_layout_reexport',3600)
    target=args.output_dir/args.precision/f'sopro_{args.graph}_{args.precision}.tflite';target.parent.mkdir(parents=True,exist_ok=True)
    fp32=args.output_dir/'fp32'/f'sopro_{args.graph}_fp32.tflite'
    report={'graph':args.graph,'precision':args.precision,'versions':versions()}
    if args.precision=='wfp16':
        report['recipe']=half(fp32,target)
        report['alias_probe']='Inherited from audited fp32 parameters; direct FLOAT_CASTING without external-buffer repacking.'
    elif args.graph=='style_prefix':
        from style_prefix import StylePrefixGraph
        if args.checkpoint:
            from sopro import SoproTTS
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            tts=SoproTTS.from_pretrained(str(args.checkpoint),device='cpu',dtype=torch.float32)
        else:tts=load_tts()
        with np.load(args.input_npz) as archive:tokens=archive['ref_semantic_tokens'][:,:160].copy()
        with torch.no_grad():example=tts.model.embed_semantic(torch.from_numpy(tokens)).detach().clone().contiguous()
        module=StylePrefixGraph(tts.model).eval();contiguous_constants(module)
        import alias_probe,litert_torch
        alias_probe.install();litert_torch.convert(module,(example,)).export(str(target))
        report['alias_probe']={'aliased_constants':len(alias_probe._collisions),'constants_lowered':alias_probe._order[0]}
        assert not alias_probe._collisions
        report['attention_layout']=promote_batch_matmul_rank4(target)
        report['constant_reshape_folding']=fold_constant_reshapes(target)
    else:
        from vocoder_rank4 import promote
        baseline=json.loads((DIRECTORY/'modules/vocoder_baseline.json').read_text())
        original=next(g for g in baseline['graphs'] if g['graph']==args.graph)
        assert sha256(args.source_graph)==original['sha256'] and args.source_graph.stat().st_size==original['bytes']
        report['alias_probe']=original['alias_probe'];report['source_graph_sha256']=original['sha256']
        report['layout']=promote(args.source_graph,target)
    inspection=scan(target);assert inspection['pass_'],inspection['forbidden']
    if args.graph=='style_prefix':
        paths=constant_reshape_paths(target);assert not paths,paths;report['constant_reshape_paths']=paths
    else:
        rank3=[t for t in inspection['tensors'] if t['shape'] is not None and len(t['shape'])==3]
        assert not rank3,rank3;report['rank3_tensor_count']=0
    write_json(target.with_suffix('.opcheck.json'),inspection)
    report.update(status='EXPORTED',path=str(target),sha256=sha256(target),bytes=target.stat().st_size)
    write_json(target.with_suffix('.export.json'),report)
    print(target,report['sha256'],report['bytes'],flush=True)


if __name__=='__main__':main()
