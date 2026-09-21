"""Download the pinned Apache-2.0 upstream source into the local work cache."""
import json
from pathlib import Path
from common import ROOT, bounded_run, sha256, write_json
from huggingface_hub import snapshot_download
REPO = 'samuel-vitorino/sopro-v2-turbo'
REVISION = 'f747f9edfb7b0233a3b7105af3a75603a7213d26'
EXPECTED = {'README.md': '0cd5af86ff15be2d0585fbda61f112ce94031108b0ff3f169477c7485dd0fca9', 'config.json': 'b8593ff96b2976215ff64e58e784f4d667f4f0f205ae8686b57e78c10f42d6d7', 'model.safetensors': '5ddc2e905ce68e015b3760f9437969f6fd7ef0af601b8e457fc18344bb908ffa', 'semantic_encoder.safetensors': 'd1d1bccdfffdcd6d01d802a3e5dbefa0674c660f5022d87ec14e3271596c5c5f', 'speaker_encoder.safetensors': '67422a45d801542ee0aa55ce824f570ceb1c6b2283b4e74978c1b4673b49acd7', 'tokenizer.model': '2d76e7a4e8dbd0a4d2137c13200ac9710ad3fd93fad20a61f71102ab7bae754d', 'vocoder.safetensors': '07e5561491cce41f7f90cfdb94b2ff263ff5742c3d89339db99b17ad82cc3f44'}

def main():
    bounded_run('source_download', 3600)
    path = Path(snapshot_download(repo_id=REPO, revision=REVISION, allow_patterns=list(EXPECTED),
        cache_dir=str(ROOT/'cache/huggingface/hub'), max_workers=4))
    rows=[]
    for name, expected in EXPECTED.items():
        file=path/name; actual=sha256(file)
        assert actual==expected, (name, actual, expected)
        rows.append({'name':name,'path':str(file.relative_to(ROOT)),'sha256':actual,'size_bytes':file.stat().st_size})
    assert not list((ROOT/'cache/huggingface').rglob('*.incomplete'))
    (ROOT/'results/model_path.txt').write_text(str(path.relative_to(ROOT))+'\n')
    report={'status':'PASS','repo':REPO,'revision':REVISION,'files':rows}
    write_json(ROOT/'results/sources.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
